#!/usr/bin/env python3
"""
batch_eval_docs_cache.py

Iterates through document files from 01 to 21 (documents_0000XX.jsonl).
For each file, it builds (or loads from cache) a per-language hashed TF-IDF
index and caches it in a dedicated folder (cache_docXX).
It then loads queries and qrels and inspects retrieval results for queries
with relevant documents within the currently processed document file.

This script helps manage indexing and caching for multiple document subsets
as part of a traditional IR approach with multilingual handling.
"""

import json
import time
import logging
import pickle
from pathlib import Path
from collections import defaultdict

import polars as pl
from langdetect import detect, DetectorFactory
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

# ─── Logging Configuration ────────────────────────────────────────────────────
# Set up basic logging to display timestamps, log level, and message
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger()

# ─── Configuration Variables ──────────────────────────────────────────────────
# Set a seed for the language detector for reproducibility
DetectorFactory.seed = 0

# Define base directory where data files are located
# IMPORTANT: Update this path to match your local file structure
BASE_DIR      = Path(r"C:\Users\bbalt\Downloads\endreport\ScientificTextRetrieval\longeval_sci_training_2025_abstract")

# Define paths to the queries and qrels files (assumed to be common for all document files)
QUERIES_PATH  = BASE_DIR / "queries.txt"
QRELS_PATH    = BASE_DIR / "qrels.txt"

# Define parameters for HashingVectorizer and retrieval
N_FEATURES     = 2**16 # Number of features (columns) for the hashed vectorizer
TOP_K          = 5     # Number of top documents to retrieve
NON_WORD_LANGS = {"zh-cn", "zh-tw", "ja", "ko"} # Languages where character n-grams are more suitable than word n-grams

# Define the range of document file numbers to process
DOC_FILE_RANGE_START = 1
DOC_FILE_RANGE_END   = 21 # Inclusive

# ─── Function to Process a Single Document File and Manage its Cache ──────────
def process_document_file(doc_number: int):
    """
    Processes a single document file, builds or loads its index, and performs
    retrieval inspection for relevant queries. Includes timing for the whole process.

    Args:
        doc_number (int): The sequential number of the document file (e.g., 19, 20).
    """
    start_time_file = time.time() # Start timer for this document file

    # Construct the document file path (e.g., documents_000019.jsonl)
    doc_filename = f"documents_{doc_number:06d}.jsonl"
    doc_path     = BASE_DIR / "documents" / doc_filename

    # Define the cache directory for this specific document file
    cache_dir    = Path(f"cache_doc{doc_number:02d}") # e.g., cache_doc19, cache_doc20

    # Define file paths for caching components within this cache directory
    hashers_file      = cache_dir / "hashers.pkl"
    transformers_file = cache_dir / "transformers.pkl"
    lang_ids_file     = cache_dir / "lang_ids.pkl"
    mats_dir          = cache_dir / "mats_npz"

    # Create cache directories if they don't exist
    cache_dir.mkdir(exist_ok=True)
    mats_dir.mkdir(exist_ok=True)

    logger.info(f"--- Processing {doc_filename} ---")

    # ─── Function to Build and Cache the Index for this file ──────────────────
    def build_and_cache_index():
        """
        Builds a per-language Hashed TF-IDF index from the current document file
        and caches the components.
        """
        logger.info(f"Building index for {doc_filename} (this will take some time)…")
        lang_texts, lang_ids = defaultdict(list), defaultdict(list)
        lang_detection_failures = 0 # Counter for language detection failures
        start = time.time() # Timer for the build process within this file
        total_docs_read = 0

        # Read documents line by line from the current file
        try:
            with doc_path.open(encoding="utf8") as f:
                for i, line in enumerate(f, start=1):
                    total_docs_read += 1
                    try:
                        # Load JSON record from each line
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        # Skip lines with JSON decoding errors
                        logger.warning(f"Skipping line {i} in {doc_filename} due to JSON decode error.")
                        continue

                    # Extract text to be indexed (abstract or title)
                    txt = (rec.get("abstract") or "").strip() or (rec.get("title") or "").strip()
                    if not txt:
                        # Skip documents with no abstract or title
                        continue

                    try:
                        # Detect language of the text
                        lang = detect(txt)
                    except:
                        # Default to English if language detection fails
                        lang = "en"
                        lang_detection_failures += 1 # Increment counter on failure
                        # logger.warning(f"Could not detect language for doc {rec.get('id', 'N/A')} in {doc_filename}, defaulting to 'en'.")
                        # Suppress individual warnings during batch processing for cleaner output, rely on the total count at the end.


                    # Store text and document ID based on detected language
                    lang_texts[lang].append(txt)
                    lang_ids[lang].append(rec["id"])

                    # Log progress periodically
                    if i % 10000 == 0:
                        logger.info(f"  {i:,} lines read from {doc_filename} (elapsed {time.time()-start:.1f}s)")
        except FileNotFoundError:
            logger.error(f"Document file not found: {doc_path}. Skipping.")
            return None, None, None, None # Return None if file not found

        logger.info(f"Finished reading {total_docs_read} lines from {doc_filename}.")
        logger.info(f"Bucketed {sum(len(v) for v in lang_texts.values())} texts in {time.time()-start:.1f}s")
        logger.info(f"Language detection failed for {lang_detection_failures} documents in {doc_filename}, defaulted to 'en'.") # Log the failure count

        hashers, transformers, mats = {}, {}, {}
        # Process texts for each language
        for lang, texts in lang_texts.items():
            # Determine analyzer and ngram range based on language type
            analyzer = "char" if lang in NON_WORD_LANGS else "word"
            ngram    = (2,4) if analyzer=="char" else (1,2)

            # Initialize HashingVectorizer
            hv      = HashingVectorizer(n_features=N_FEATURES,
                                        analyzer=analyzer,
                                        ngram_range=ngram,
                                        alternate_sign=False, # Use positive hash values
                                        norm=None) # Don't normalize counts yet

            # Transform texts into count vectors
            cnt     = hv.transform(texts)

            # Initialize and fit TfidfTransformer
            tf_tr   = TfidfTransformer(norm="l2", use_idf=True) # Apply L2 normalization and use inverse document frequency
            tfidf_m = tf_tr.fit_transform(cnt) # Fit and transform count vectors to TF-IDF

            logger.info(f"[{lang}] Indexed {len(texts)} docs; shape={tfidf_m.shape}")

            # Save the TF-IDF matrix for the current language
            sparse.save_npz(mats_dir / f"{lang}.npz", tfidf_m)

            # Store the vectorizer, transformer, and matrix
            hashers[lang]      = hv
            transformers[lang] = tf_tr
            mats[lang]         = tfidf_m

        # Cache the vectorizers, transformers, and language IDs
        with open(hashers_file, "wb")      as f: pickle.dump(hashers, f)
        with open(transformers_file, "wb") as f: pickle.dump(transformers, f)
        with open(lang_ids_file, "wb")     as f: pickle.dump(lang_ids, f)

        logger.info(f"Built & cached for {doc_filename} in {time.time()-start:.1f}s")
        return hashers, transformers, lang_ids, mats

    # ─── Load Cached Index or Build for this file ───────────────────────────────
    # Check if cached files exist for this document file
    if hashers_file.exists() and transformers_file.exists() and lang_ids_file.exists() and list(mats_dir.glob("*.npz")):
        logger.info(f"Loading cached index for {doc_filename}…")
        try:
            # Load cached components
            with open(hashers_file, "rb")      as f: hashers      = pickle.load(f)
            with open(transformers_file, "rb") as f: transformers = pickle.load(f)
            with open(lang_ids_file, "rb")     as f: lang_ids     = pickle.load(f)
            # Load sparse matrices for each language from the cache directory
            mats = {p.stem: sparse.load_npz(p) for p in mats_dir.glob("*.npz")}
            logger.info("Index loaded.")
            # Return loaded components
            loaded_hashers, loaded_transformers, loaded_lang_ids, loaded_mats = hashers, transformers, lang_ids, mats
        except Exception as e:
             logger.warning(f"Could not load cached index for {doc_filename}: {e}. Rebuilding index.")
             # If loading fails, proceed to build the index
             loaded_hashers, loaded_transformers, loaded_lang_ids, loaded_mats = build_and_cache_index()
    else:
        # If cache does not exist, build the index
        loaded_hashers, loaded_transformers, loaded_lang_ids, loaded_mats = build_and_cache_index()

    # Check if index building/loading was successful (in case file was not found)
    if loaded_hashers is None:
        end_time_file = time.time()
        logger.info(f"Skipped processing {doc_filename} due to file not found or build error.")
        logger.info(f"Total time for {doc_filename}: {end_time_file - start_time_file:.1f}s")
        return # Exit function if index is not available


    # ─── Retrieval Function (scoped to this file's index) ───────────────────────
    def retrieve(q, k=TOP_K, current_hashers=loaded_hashers, current_transformers=loaded_transformers, current_mats=loaded_mats, current_lang_ids=loaded_lang_ids):
        """
        Retrieves the top K documents for a given query using the language-specific
        TF-IDF indexes for the current document file.
        """
        try:
            # Detect language of the query
            lang = detect(q)
        except:
            # Default to English if language detection fails
            lang = "en"
            # logger.warning(f"Could not detect language for query '{q[:50]}...' when searching {doc_filename}, defaulting to 'en'.")
            # Suppress individual warnings during batch processing for cleaner output


        # Check if an index exists for the detected language in the current file's index
        if lang not in current_hashers:
            # If not, default to English index if available
            if "en" in current_hashers:
                lang = "en"
                logger.warning(f"No index for language '{lang}' in {doc_filename}, using 'en' index.")
            else:
                # If no English index either, return empty list
                logger.error(f"No index available for query language '{lang}' or 'en' in {doc_filename}.")
                return []

        # Get the vectorizer, transformer, matrix, and document IDs for the detected language
        hv, tf_tr, mat = current_hashers[lang], current_transformers[lang], current_mats[lang]
        ids            = current_lang_ids[lang]

        # Transform the query into a count vector and then a TF-IDF vector
        q_cnt   = hv.transform([q])
        q_tfidf = tf_tr.transform(q_cnt)

        # Calculate cosine similarity between the query vector and document vectors
        sims    = cosine_similarity(q_tfidf, mat).ravel()

        # Get the indices of the top K most similar documents
        idxs    = sims.argsort()[::-1][:k]

        # Return the document IDs corresponding to the top indices
        return [ids[i] for i in idxs]

    # ─── Load Queries and Filter Qrels (common for all files, but filtered per file) ──
    logger.info(f"Loading queries + filtering qrels for {doc_filename}…")

    # Load queries from the queries file (only need to do this once in the main loop if outside this function)
    # For simplicity here, we load it inside, but could optimize by loading once outside.
    try:
        qdf = pl.read_csv(str(QUERIES_PATH), has_header=False,
                          separator="\t", new_columns=["query_id","query"])
        queries = {str(r["query_id"]): r["query"] for r in qdf.iter_rows(named=True)}
    except Exception as e:
        logger.error(f"Error loading queries from {QUERIES_PATH}: {e}")
        queries = {}


    # Collect all document IDs from all languages in the index built for *this* file
    all_ids_in_this_file = {d for lst in loaded_lang_ids.values() for d in lst}

    # Load qrels and filter to include only documents present in the index *for this file*
    qrels_for_this_file = defaultdict(set)
    try:
        with open(QRELS_PATH, encoding="utf8") as f:
            for line in f:
                parts = line.split()
                if len(parts) == 4:
                    qid, snap, did, rel = parts
                    # Only keep relevant judgments (relevance > 0) for documents in our index *for this file*
                    if int(rel) > 0 and did in all_ids_in_this_file:
                        qrels_for_this_file[qid].add(did)
                else:
                    logger.warning(f"Skipping malformed qrels line in {QRELS_PATH}: {line.strip()}")
    except FileNotFoundError:
        logger.warning(f"Qrels file not found at {QRELS_PATH}. Skipping qrels loading for {doc_filename}.")
    except Exception as e:
        logger.error(f"Error loading qrels from {QRELS_PATH}: {e}")
        qrels_for_this_file = defaultdict(set)


    logger.info(f"Queries with ≥1 relevant in indexed documents for {doc_filename}: {len(qrels_for_this_file)}")

    # ─── Inspect Retrieval Results for Queries with Qrels for this file ─────────
    logger.info(f"Inspecting retrieval results for queries with qrels in {doc_filename}…")

    if not qrels_for_this_file:
        logger.warning(f"No queries with relevant documents found in qrels for the indexed documents in {doc_filename}. Cannot perform inspection.")
    else:
        # Iterate through queries that have relevant documents in the qrels for this file
        for qid, gold in qrels_for_this_file.items():
            # Get the query text
            qtxt  = queries.get(qid, f"Query text not found for QID {qid}")
            # Retrieve top documents using our system for this file's index
            preds = retrieve(qtxt) # Use the retrieve function with the loaded/built components

            # Print results for inspection
            print(f"\n--- Results for QueryID: {qid} in {doc_filename} ---")
            print(" Query Text    :", qtxt)
            print(" Gold Relevant :", sorted(gold)) # Relevant documents from qrels
            print(" Top-5 Retrieved:", preds)       # Documents retrieved by the system
            print("--------------------------------------------")

    end_time_file = time.time() # End timer for this document file
    logger.info(f"--- Finished processing {doc_filename} ---")
    logger.info(f"Total time for {doc_filename}: {end_time_file - start_time_file:.1f}s")


# ─── Main Execution Loop ──────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info(f"Starting batch processing for documents {DOC_FILE_RANGE_START:06d} to {DOC_FILE_RANGE_END:06d}")
    total_batch_start_time = time.time() # Start timer for the entire batch

    for doc_num in range(DOC_FILE_RANGE_START, DOC_FILE_RANGE_END + 1):
        process_document_file(doc_num)

    total_batch_end_time = time.time() # End timer for the entire batch
    logger.info("Batch processing finished.")
    logger.info(f"Total time for batch processing: {total_batch_end_time - total_batch_start_time:.1f}s")

