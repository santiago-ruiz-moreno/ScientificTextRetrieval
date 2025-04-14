import polars as pl
import os
import re
import time
import sys

try:
    from langdetect import detect, LangDetectException
except ImportError:
    print("Error: 'langdetect' library not found.")
    print("Please install it using the PyCharm terminal:")
    print("pip install langdetect")
    sys.exit(1)

# --- Configuration ---
print("--- Configuration ---")
try:
    base_folder = os.getcwd()
    abstract_data_folder = os.path.join(base_folder, 'longeval_sci_training_2025_abstract')
    documents_path = os.path.join(abstract_data_folder, 'documents')
    print(f"Script base folder: {base_folder}")
    print(f"Looking for documents in: {documents_path}")
except Exception as e:
    print(f"Error setting up paths: {e}")
    sys.exit(1)

# --- Define Schema ---
print("\nDefining data schema...")
abstract_schema = {
    "id": pl.Utf8, "title": pl.Utf8, "abstract": pl.Utf8,
    "authors": pl.List(pl.Struct([pl.Field("name", pl.Utf8)])),
    "createdDate": pl.Utf8, "doi": pl.Utf8, "arxivId": pl.Utf8,
    "pubmedId": pl.Utf8, "magId": pl.Utf8, "oaiIds": pl.List(pl.Utf8),
    "links": pl.List(pl.Struct([
        pl.Field("type", pl.Utf8), pl.Field("url", pl.Utf8)
    ])),
    "publishedDate": pl.Utf8, "updatedDate": pl.Utf8
}

# --- Helper Functions ---
print("Defining helper functions...")


def clean_text_polars(text: str | None) -> str | None:
    if not isinstance(text, str):
        return None
    cleaned = re.sub(r'Comment:.*', '', text, flags=re.IGNORECASE | re.DOTALL)
    cleaned = cleaned.replace('\n', ' ').replace('\u00a0', ' ')
    cleaned = cleaned.strip()
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)
    return cleaned if cleaned else None


def detect_language_polars(text: str | None) -> str:
    if not isinstance(text, str) or len(text.strip()) < 10:
        return "null_or_short"
    try:
        return detect(text)
    except LangDetectException:
        return "undetermined"
    except Exception:
        return "detection_error"


# --- Language Detection Progress Tracker (Logging every 1000 abstracts) ---
class LanguageDetectorWithProgress:
    def __init__(self, total_count=None, print_step=1000):
        self.counter = 0
        self.total_count = total_count
        self.print_step = print_step

    def detect_and_count(self, text: str | None) -> str:
        lang = detect_language_polars(text)
        self.counter += 1
        if self.counter % self.print_step == 0:
            if self.total_count is not None:
                print(
                    f"[{time.strftime('%H:%M:%S')}] Processed abstract {self.counter}/{self.total_count} for language...",
                    flush=True)
            else:
                print(f"[{time.strftime('%H:%M:%S')}] Processed abstract {self.counter} for language...", flush=True)
        return lang

    def reset(self):
        self.counter = 0

    def get_count(self):
        return self.counter


# --- Main Processing Logic ---
print("\n--- Starting Main Processing Loop ---")
script_start_time = time.time()

if not os.path.isdir(documents_path):
    print(f"ERROR: Documents directory not found at: {documents_path}")
    sys.exit(1)

try:
    all_files = os.listdir(documents_path)
    jsonl_files = sorted([f for f in all_files if f.endswith(".jsonl")])
    print(f"Found {len(jsonl_files)} .jsonl files to process.")
except Exception as e:
    print(f"Error listing files in directory {documents_path}: {e}")
    sys.exit(1)

if not jsonl_files:
    print("No .jsonl files found in the directory.")
    sys.exit(0)

# Initialize overall counters
grand_total_docs = 0
grand_total_analyzed = 0
grand_total_english = 0
grand_total_non_english = 0
grand_total_null_short = 0
grand_total_undetermined = 0
grand_total_error = 0
files_processed_count = 0
files_error_count = 0

# --- Loop Through Files ---
for i, filename in enumerate(jsonl_files):
    current_file_path = os.path.join(documents_path, filename)
    file_start_time = time.time()
    print(f"\n{'=' * 15} Processing file {i + 1}/{len(jsonl_files)}: {filename} {'=' * 15}")

    # Detailed file reading update
    print(f"---> Starting to read {filename} ...")
    try:
        file_size_bytes = os.path.getsize(current_file_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        print(f"File Size: {file_size_mb:.2f} MB")

        read_start = time.time()
        df = pl.read_ndjson(current_file_path, schema=abstract_schema)
        read_end = time.time()
        num_docs = df.height
        print(f"<--- Finished reading {filename}. Read {num_docs} documents in {read_end - read_start:.2f}s.")
        grand_total_docs += num_docs

        # Cleaning Step Update
        print("---> Starting Cleaning Step...")
        cleaning_start = time.time()
        df = df.with_columns(
            pl.col("abstract")
            .map_elements(clean_text_polars, return_dtype=pl.Utf8, strategy="threading")
            .alias("abstract_cleaned")
        )
        cleaning_end = time.time()
        print(f"<--- Finished Cleaning Step. Cleaned abstracts in {cleaning_end - cleaning_start:.2f}s.")

        # Language Detection Step with Detailed Update (logging every 1000 abstracts)
        print("---> Starting Language Detection Step (logging every 1000 abstracts)...")
        detection_start = time.time()
        total_to_process = df.filter(pl.col("abstract_cleaned").is_not_null()).height
        print(f"Number of valid abstracts to process for language detection: {total_to_process}")

        lang_progress_tracker = LanguageDetectorWithProgress(total_count=total_to_process, print_step=1000)
        df = df.with_columns(
            pl.col("abstract_cleaned")
            .map_elements(lang_progress_tracker.detect_and_count, return_dtype=pl.Utf8)
            .alias("detected_language")
        )
        final_count_this_file = lang_progress_tracker.get_count()
        print(
            f"\n    Final count for language detection: Processed {final_count_this_file}/{total_to_process} abstracts.")
        detection_end = time.time()
        print(f"<--- Finished Language Detection Step. Time: {detection_end - detection_start:.2f}s.")

        # Analysis Step Update
        print("---> Starting Analysis Step for this file...")
        analysis_start = time.time()
        total_abstracts_analyzed = df.filter(pl.col("abstract_cleaned").is_not_null()).height
        special_codes = ["null_or_short", "undetermined", "detection_error"]
        non_english_filter = ~pl.col("detected_language").is_in(['en'] + special_codes)
        non_english_count = df.filter(non_english_filter).height
        english_count = df.filter(pl.col("detected_language") == "en").height
        null_short_count = df.filter(pl.col("detected_language") == "null_or_short").height
        undetermined_count = df.filter(pl.col("detected_language") == "undetermined").height
        error_count = df.filter(pl.col("detected_language") == "detection_error").height
        analysis_end = time.time()
        print(f"<--- Finished Analysis Step. Time: {analysis_end - analysis_start:.2f}s.")

        # Detailed Summary for This File
        print(f"\n--- Language Summary for: {filename} ---")
        print(f"   Total abstracts read:                 {df.height:>10}")
        print(f"   Total non-null cleaned abstracts:     {total_abstracts_analyzed:>10}")
        print(f"      - English ('en'):                  {english_count:>10}")
        print(f"      - Non-English (specific):          {non_english_count:>10}")
        print(f"      - Null/short/empty:                {null_short_count:>10}")
        print(f"      - Undetermined (langdetect err):   {undetermined_count:>10}")
        print(f"      - Detection error:                 {error_count:>10}")
        print(f"   --------------------------------------")

        # Update Grand Totals
        grand_total_analyzed += total_abstracts_analyzed
        grand_total_english += english_count
        grand_total_non_english += non_english_count
        grand_total_null_short += null_short_count
        grand_total_undetermined += undetermined_count
        grand_total_error += error_count

        files_processed_count += 1

    except pl.ComputeError as e:
        print(f"!!! Polars Compute Error processing {filename}: {e} !!!")
        files_error_count += 1
    except Exception as e:
        print(f"!!! An Unexpected Error Occurred processing {filename}: {type(e).__name__} - {e} !!!")
        files_error_count += 1

    file_end_time = time.time()
    print(f"--- Finished processing {filename}. Total time for file: {file_end_time - file_start_time:.2f}s ---")

# --- Finished All Files ---
print("\n\n--- Finished Processing All Files ---")
script_end_time = time.time()

# --- Grand Summary ---
print("\n--- Grand Total Language Summary ---")
print(f"   Total documents processed:            {grand_total_docs:>10}")
print(f"   Total non-null cleaned abstracts:     {grand_total_analyzed:>10}")
print(f"      - English ('en'):                  {grand_total_english:>10}")
print(f"      - Non-English (specific):          {grand_total_non_english:>10}")
print(f"      - Null/short/empty:                {grand_total_null_short:>10}")
print(f"      - Undetermined (langdetect err):   {grand_total_undetermined:>10}")
print(f"      - Detection error:                 {grand_total_error:>10}")

print("\n--- Script End ---")
print(f"Total execution time for all files: {script_end_time - script_start_time:.2f} seconds")
