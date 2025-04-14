import polars as pl
import os

# Paths

data_folder = "C:/Users/jruizmor/Documents/WU/09 Courses/Advanced Information Retrieval/LonscievalDataset/"
abstract_data_folder = data_folder + "longeval_sci_training_2025_abstract/"
full_text_data_folder = data_folder + "longeval_sci_training_2025_fulltext/"
queries_path = abstract_data_folder  + "queries.txt"
abstracts_path = abstract_data_folder + "documents/"


# Importing queries

queries = pl.scan_csv(queries_path, separator = "\t", has_header=False, new_columns=["query_id", "query"])

# TO BE DONE: @bubaltali @Aditiwien


# 6 hours: (Burak | Aditi )
# 1. Abstracts are in different languages. ()
# 2. Issues when importing the JSONL files, namely "Skipping file documents_000002.jsonl due to error: got non-null value for NULL-typed column:" 


# 6 Hours (Team A : Santiago | Saad | Ron   )
# 3. WORD-HANDLING FOR ABSTRACTS: 
# 4.  - TOKENIZATION: / AFTER TRANSLATION 
# 5.  - STEMMING
# 6.  - LEMMATIZATION

abstract_pl_query = []
for file in os.listdir(abstracts_path):
    file_path = abstracts_path + file
    try:
        pl_abstract = pl.read_ndjson(file_path).lazy()
        abstract_pl_query.append(pl_abstract)
    except Exception as e:
        print(f"Skipping file {file} due to error: {e}")

if abstract_pl_query:
    pl_all_abstracts = pl.collect_all(abstract_pl_query)
else:
    pl_all_abstracts = []
    print("No valid abstracts were found.")


# Queries = Abstracts (IDs.)
# IR
# TDF-IF (1ST Task.)




# Nerual Network. (2nd Week.)
