import polars as pl
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from incrementaltfidf import  IncrementalTfidf
from sklearn.metrics.pairwise import cosine_similarity
import json
import sys
from tqdm import tqdm

# Ensure incrementaltfidf is accessible
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from incrementaltfidf import IncrementalTfidf

# --- NLP Preprocessing ---
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import nltk


# Paths

data_folder = "C:/Users/jruizmor/Documents/WU/09 Courses/Advanced Information Retrieval/LonscievalDataset/"
abstract_data_folder = data_folder + "longeval_sci_training_2025_abstract/"
full_text_data_folder = data_folder + "longeval_sci_training_2025_fulltext/"
queries_path = abstract_data_folder  + "queries.txt"
abstracts_path = abstract_data_folder + "documents/"


# --- Adding queires to the Tf-IDF matrix ---


queries = pl.scan_csv(queries_path, separator = "\t", has_header=False, new_columns=["query_id", "query"])
queries_collected = pl.DataFrame(queries.collect())

queries_collected = queries_collected.with_columns(
    pl.col("query").map_elements(
        lambda x: [clean_text(word) for word in x.split(" ")] if x is not None else None,
        return_dtype =  pl.List(pl.Utf8)
    ).alias("query_cleaned")
)

# --- Adding current queries to the TF-IDF matrix ---

tfidf_model.add_documents(queries_collected["query_cleaned"], doc_ids=queries_collected["query_id"])

# --- Finding the most relevant documents for each query ---

def add_top_5_similar_docs(queries_collected: pl.DataFrame, tfidf_matrix, document_ids, doc_id_filter_fn=None):
    """
    For each query ID in queries_collected["query_id"], finds top 5 closest document IDs using cosine similarity.

    Args:
        queries_collected: Polars DataFrame with 'query_id' column.
        tfidf_matrix: Output from TfidfTransformer (sparse matrix).
        document_ids: List of IDs (strings or ints) for each row in the TF-IDF matrix.
        doc_id_filter_fn: Optional function to filter which IDs count as documents (e.g., lambda x: len(str(x)) == 4)

    Returns:
        Polars DataFrame with a new 'top_5' column listing top 5 closest document IDs for each query.
    """
    tfidf_dense = tfidf_matrix.toarray()
    id_to_index = {str(doc_id): idx for idx, doc_id in enumerate(document_ids)}

    # Filter document vectors
    if doc_id_filter_fn:
        doc_ids = [str(doc_id) for doc_id in document_ids if doc_id_filter_fn(str(doc_id))]
    else:
        doc_ids = [str(doc_id) for doc_id in document_ids]

    doc_indices = [id_to_index[doc_id] for doc_id in doc_ids]
    doc_vectors = tfidf_dense[doc_indices]

    top5_results = []

    for query_id in queries_collected["query_id"]:
        query_id = str(query_id)
        if query_id not in id_to_index:
            top5_results.append([])
            continue

        query_vec = tfidf_dense[id_to_index[query_id]].reshape(1, -1)
        sim_scores = cosine_similarity(query_vec, doc_vectors)[0]
        top_indices = np.argsort(sim_scores)[::-1][:5]
        top_docs = [doc_ids[i] for i in top_indices]
        top5_results.append(top_docs)

    return queries_collected.with_columns(
        pl.Series("top_5", top5_results)
    )
    
add_top_5_similar_docs(queries_collected, tfidf_model.get_tfidf_matrix(), tfidf_model.get_document_ids(), doc_id_filter_fn=lambda x: len(x) == 8)