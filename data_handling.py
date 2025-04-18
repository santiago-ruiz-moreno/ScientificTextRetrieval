import polars as pl
import os

# Construct paths using os.path.join for safety and cross-platform compatibility.
data_folder = os.getcwd()
abstract_data_folder = os.path.join(data_folder, "longeval_sci_training_2025_abstract")
full_text_data_folder = os.path.join(data_folder, "longeval_sci_training_2025_fulltext")
queries_path = os.path.join(abstract_data_folder, "queries.txt")
abstracts_path = os.path.join(abstract_data_folder, "documents")

# Import queries from a tab-separated file.
queries = pl.scan_csv(
    queries_path,
    separator="\t",
    has_header=False,
    new_columns=["query_id", "query"]
)

# List to collect lazy DataFrames for abstracts.
abstract_pl_query = []

# Iterate through the JSONL files in the abstracts directory.
for file in os.listdir(abstracts_path):
    # Process only files with the .jsonl extension.
    if not file.endswith(".jsonl"):
        continue

    file_path = os.path.join(abstracts_path, file)
    try:
        # Read the JSONL file eagerly in order to inspect the data.
        df = pl.read_ndjson(file_path)

        # Debug: Print the columns and a 5-row preview of the file.
        print(f"Columns in {file}:", df.columns)
        print("Preview of the file:")
        print(df.head(5))

        # Convert the DataFrame to a lazy version and add it to the collection.
        abstract_pl_query.append(df.lazy())
    except Exception as e:
        print(f"Skipping file {file} due to error: {e}")

# Collect all lazy DataFrames into one final result if any valid ones were read.
if abstract_pl_query:
    pl_all_abstracts = pl.collect_all(abstract_pl_query)
else:
    pl_all_abstracts = []
    print("No valid abstracts were found.")
