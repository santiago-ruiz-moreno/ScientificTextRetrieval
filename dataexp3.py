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
    # Make sure this path is correct for your setup
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

# NOTE: clean_text_polars is kept if needed for abstracts elsewhere,
# but language detection will now run on the original 'title'.
def clean_text_polars(text: str | None) -> str | None:
    if not isinstance(text, str):
        return None
    cleaned = re.sub(r'Comment:.*', '', text, flags=re.IGNORECASE | re.DOTALL)
    cleaned = cleaned.replace('\n', ' ').replace('\u00a0', ' ')
    cleaned = cleaned.strip()
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)
    return cleaned if cleaned else None


def detect_language_polars(text: str | None) -> str:
    if not isinstance(text, str) or len(text.strip()) < 5: # Titles can be shorter, adjust threshold if needed
        return "null_or_short"
    try:
        return detect(text)
    except LangDetectException:
        # Sometimes langdetect fails on very short but non-empty strings
        if len(text.strip()) < 10:
             return "null_or_short"
        return "undetermined"
    except Exception as e:
        # Optional: Log the specific error for debugging
        # print(f"Language detection error on text '{text[:50]}...': {e}")
        return "detection_error"


# --- Language Detection Progress Tracker (Logging every 10000 items) --- ## MODIFIED HERE
class LanguageDetectorWithProgress:
    def __init__(self, total_count=None, print_step=10000): # MODIFIED: print_step default changed
        self.counter = 0
        self.total_count = total_count
        self.print_step = print_step # MODIFIED: set from argument

    def detect_and_count(self, text: str | None) -> str:
        lang = detect_language_polars(text)
        self.counter += 1
        if self.counter % self.print_step == 0:
            if self.total_count is not None:
                print(
                    f"[{time.strftime('%H:%M:%S')}] Processed title {self.counter}/{self.total_count} for language...", # MODIFIED: "abstract" -> "title"
                    flush=True)
            else:
                print(f"[{time.strftime('%H:%M:%S')}] Processed title {self.counter} for language...", flush=True) # MODIFIED: "abstract" -> "title"
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
grand_total_analyzed_titles = 0 # MODIFIED: Variable name
grand_total_english_titles = 0 # MODIFIED: Variable name
grand_total_non_english_titles = 0 # MODIFIED: Variable name
grand_total_null_short_titles = 0 # MODIFIED: Variable name
grand_total_undetermined_titles = 0 # MODIFIED: Variable name
grand_total_error_titles = 0 # MODIFIED: Variable name
files_processed_count = 0
files_error_count = 0

# --- Loop Through Files ---
# Initialize the progress tracker ONCE outside the loop, but reset it inside
# MODIFIED: Set print_step to 10000
lang_progress_tracker = LanguageDetectorWithProgress(print_step=10000)

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

        # Optional: Keep abstract cleaning if needed for other purposes
        # print("---> Starting Abstract Cleaning Step (Optional)...")
        # cleaning_start = time.time()
        # df = df.with_columns(
        #     pl.col("abstract")
        #     .map_elements(clean_text_polars, return_dtype=pl.Utf8, strategy="threading")
        #     .alias("abstract_cleaned")
        # )
        # cleaning_end = time.time()
        # print(f"<--- Finished Abstract Cleaning Step. Time: {cleaning_end - cleaning_start:.2f}s.")

        # Language Detection Step using TITLE with Detailed Update (logging every 10000 titles) ## MODIFIED HERE
        print("---> Starting Language Detection Step on TITLES (logging every 10000 titles)...") # MODIFIED: message
        detection_start = time.time()
        # MODIFIED: Calculate total based on non-null titles
        total_to_process = df.filter(pl.col("title").is_not_null()).height
        print(f"Number of valid titles to process for language detection: {total_to_process}")

        lang_progress_tracker.reset() # MODIFIED: Reset counter for each file
        lang_progress_tracker.total_count = total_to_process # Update total count for progress display

        df = df.with_columns(
            pl.col("title") # MODIFIED: Use 'title' column
            .map_elements(lang_progress_tracker.detect_and_count, return_dtype=pl.Utf8)
            .alias("title_language") # MODIFIED: Alias name
        )
        final_count_this_file = lang_progress_tracker.get_count()
        # Verify counts match, allowing for potential nulls filtered earlier
        if final_count_this_file != total_to_process:
             print(f"\n    Note: Processed {final_count_this_file} titles, expected {total_to_process} based on non-null check.")
        else:
             print(f"\n    Final count for language detection: Processed {final_count_this_file}/{total_to_process} titles.") # MODIFIED: "abstracts" -> "titles"

        detection_end = time.time()
        print(f"<--- Finished Title Language Detection Step. Time: {detection_end - detection_start:.2f}s.") # MODIFIED: message

        # Analysis Step Update (using title_language) ## MODIFIED HERE
        print("---> Starting Analysis Step for TITLE language...") # MODIFIED: message
        analysis_start = time.time()
        # MODIFIED: Calculate based on 'title' and 'title_language'
        total_titles_analyzed = df.filter(pl.col("title").is_not_null()).height
        special_codes = ["null_or_short", "undetermined", "detection_error"]
        # MODIFIED: Use 'title_language' column
        non_english_filter = ~pl.col("title_language").is_in(['en'] + special_codes)
        non_english_count = df.filter(non_english_filter).height
        english_count = df.filter(pl.col("title_language") == "en").height
        null_short_count = df.filter(pl.col("title_language") == "null_or_short").height
        undetermined_count = df.filter(pl.col("title_language") == "undetermined").height
        error_count = df.filter(pl.col("title_language") == "detection_error").height
        analysis_end = time.time()
        print(f"<--- Finished Analysis Step. Time: {analysis_end - analysis_start:.2f}s.")

        # Detailed Summary for This File (referring to Titles) ## MODIFIED HERE
        print(f"\n--- Title Language Summary for: {filename} ---") # MODIFIED: message
        print(f"   Total documents read:                 {df.height:>10}")
        #MODIFIED: Updated labels and counts
        print(f"   Total non-null titles analyzed:       {total_titles_analyzed:>10}")
        print(f"      - English ('en'):                  {english_count:>10}")
        print(f"      - Non-English (specific):          {non_english_count:>10}")
        print(f"      - Null/short/empty:                {null_short_count:>10}")
        print(f"      - Undetermined (langdetect err):   {undetermined_count:>10}")
        print(f"      - Detection error:                 {error_count:>10}")
        # Recalculate check sum based on new categories
        check_sum = english_count + non_english_count + null_short_count + undetermined_count + error_count
        print(f"   --------------------------------------")
        print(f"   Check sum (language categories):      {check_sum:>10}")
        if check_sum != total_titles_analyzed:
            print(f"   WARNING: Check sum ({check_sum}) does not match total analyzed titles ({total_titles_analyzed})")


        # Update Grand Totals ## MODIFIED HERE
        grand_total_analyzed_titles += total_titles_analyzed
        grand_total_english_titles += english_count
        grand_total_non_english_titles += non_english_count
        grand_total_null_short_titles += null_short_count
        grand_total_undetermined_titles += undetermined_count
        grand_total_error_titles += error_count

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

# --- Grand Summary (referring to Titles) --- ## MODIFIED HERE
print("\n--- Grand Total Title Language Summary ---") # MODIFIED: message
print(f"   Total documents processed:            {grand_total_docs:>10}")
# MODIFIED: Updated labels and variables
print(f"   Total non-null titles analyzed:       {grand_total_analyzed_titles:>10}")
print(f"      - English ('en'):                  {grand_total_english_titles:>10}")
print(f"      - Non-English (specific):          {grand_total_non_english_titles:>10}")
print(f"      - Null/short/empty:                {grand_total_null_short_titles:>10}")
print(f"      - Undetermined (langdetect err):   {grand_total_undetermined_titles:>10}")
print(f"      - Detection error:                 {grand_total_error_titles:>10}")
# Recalculate grand check sum
grand_check_sum = (grand_total_english_titles + grand_total_non_english_titles +
                   grand_total_null_short_titles + grand_total_undetermined_titles +
                   grand_total_error_titles)
print(f"   --------------------------------------")
print(f"   Check sum (language categories):      {grand_check_sum:>10}")
if grand_check_sum != grand_total_analyzed_titles:
     print(f"   WARNING: Grand check sum ({grand_check_sum}) does not match grand total analyzed titles ({grand_total_analyzed_titles})")

print(f"\nTotal files processed successfully: {files_processed_count}")
print(f"Total files with errors: {files_error_count}")

print("\n--- Script End ---")
print(f"Total execution time for all files: {script_end_time - script_start_time:.2f} seconds")