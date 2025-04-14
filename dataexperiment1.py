import polars as pl
import os
import time # Optional: To see timing


# --- Script Start ---
print("--- Script Start ---")
start_time = time.time() # Optional timing

# --- Setup Paths ---
print("Setting up paths...")
# @Santiago: I modified this line to use my local folder: 
# data_folder = os.getcwd()
data_folder = "C:/Users/jruizmor/Documents/WU/09 Courses/Advanced Information Retrieval/LonscievalDataset/"
# Construct paths using os.path.join for cross-platform compatibility
abstract_data_folder = os.path.join(data_folder, "longeval_sci_training_2025_abstract")
queries_path = os.path.join(abstract_data_folder, "queries.txt")
abstracts_path = os.path.join(abstract_data_folder, "documents")

print(f"Data Folder: {data_folder}")
print(f"Abstract Data Folder: {abstract_data_folder}")
print(f"Queries Path: {queries_path}") # Note: Queries are not used in this version
print(f"Abstracts Path: {abstracts_path}")

# --- Define Schema ---
# Define the expected schema for the abstract files
# Using Utf8 (String) for IDs/dates initially is safer due to nulls and format variations
abstract_schema = {
    "id": pl.Utf8,
    "title": pl.Utf8,
    "abstract": pl.Utf8,
    "authors": pl.List(pl.Struct([pl.Field("name", pl.Utf8)])),
    "createdDate": pl.Utf8,
    "doi": pl.Utf8,
    "arxivId": pl.Utf8,
    "pubmedId": pl.Utf8,
    "magId": pl.Utf8,
    "oaiIds": pl.List(pl.Utf8),
    "links": pl.List(pl.Struct([
        pl.Field("type", pl.Utf8),
        pl.Field("url", pl.Utf8)
    ])),
    "publishedDate": pl.Utf8,
    "updatedDate": pl.Utf8
}
print(f"\nDefined abstract schema.") # No need to print the whole schema now

# --- Initialize Counters ---
processed_files_count = 0
error_files_count = 0

# --- Start Abstract File Loop ---
print("\n--- Starting Abstract File Loop ---")
print(f"Processing abstract files from: {abstracts_path}")

# Check if the abstracts directory exists
if not os.path.isdir(abstracts_path):
    print(f"ERROR: Abstracts directory not found at {abstracts_path}")
else:
    # Process files in the documents folder.
    all_files = os.listdir(abstracts_path)
    print(f"Found {len(all_files)} items in the directory.")
    jsonl_files = [f for f in all_files if f.endswith(".jsonl")]
    print(f"Found {len(jsonl_files)} .jsonl files to process.")

    if not jsonl_files:
         print("No .jsonl files found in the directory to process.")
    else:
        for file in jsonl_files:
            file_path = os.path.join(abstracts_path, file)
            processed_files_count += 1
            print(f"\n--- Processing file ({processed_files_count}/{len(jsonl_files)}): {file} ---")

            try:
                # Read only the first 5 rows eagerly using the schema
                # Using n_rows is efficient if you only need the head
                df_head = pl.read_ndjson(
                    file_path,
                    schema=abstract_schema,
                    n_rows=5 # Read only the first 5 rows
                )

                # Print columns and the DataFrame (which is just the head)
                print(f"Columns in the file: {df_head.columns}")
                print(f"Shape: {df_head.shape}") # Will be (up to 5, 13)

                # Configure Polars display options for better readability
                with pl.Config(tbl_rows=10, tbl_cols=15, tbl_width_chars=120, fmt_str_lengths=50):
                     print(df_head)

            except Exception as e:
                # --- Handle Read Error ---
                error_files_count += 1
                print(f"\n   --- ERROR Processing {file} ---")
                print(f"   Error Type: {type(e).__name__}")
                print(f"   Error Message: {e}")
                print(f"   --- End Error for {file} ---")

                # Print the first few lines raw for inspection
                try:
                    print(f"\n   First 5 lines of raw content from {file} for inspection:")
                    with open(file_path, 'r', encoding="utf-8") as f:
                        for i in range(5):
                            line = f.readline()
                            if not line:
                                break
                            print(f"   {line.strip()}")
                except Exception as file_e:
                    print(f"   Additionally, error reading raw file contents for inspection: {file_e}")
                print("   --- End Raw Inspection ---")

# --- Finished Abstract File Loop ---
print("\n--- Finished Abstract File Loop ---")

# --- Print Summary ---
print(f"\nFinished processing.")
print(f"Total .jsonl files attempted: {processed_files_count}")
print(f"Successfully printed head for: {processed_files_count - error_files_count} files")
print(f"Files with errors during read: {error_files_count}")

# --- Script End ---
end_time = time.time()
print("\n--- Script End ---")
print(f"Total execution time: {end_time - start_time:.2f} seconds")
print("Printed head previews for individual files. No combined DataFrame was created.")