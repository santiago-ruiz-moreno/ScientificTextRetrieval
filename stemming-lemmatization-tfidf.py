import os
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

    
# Add custom path to NLTK data
nltk_data_path = r"C:\Users\jruizmor\Documents\WU\09 Courses\Advanced Information Retrieval\nltk_data"
if nltk_data_path not in nltk.data.path:
    nltk.data.path.insert(0, nltk_data_path)

# Check and download required resources
for resource in ["punkt", "stopwords", "wordnet","punkt_tab"]:
    try:
        nltk.data.find(resource)
    except LookupError:
        print(f"📥 Downloading {resource}...")
        nltk.download(resource, download_dir=nltk_data_path)


# Setup tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# pl_all_abstracts is the list of DataFrames loaded from the JSONL files
# in the importingabstracts.py script.

# --- Load, Preprocess, and Add to TF-IDF ---
documents = []
doc_ids = []
max_docs = 100000


for df in tqdm(pl_all_abstracts, desc="Loading documents"):
    for row in df.iter_rows(named=True):
        if len(documents) >= max_docs:
            break
        try:
            title = clean_text(row.get("title", ""))
            abstract = clean_text(row.get("abstract", ""))
            combined = f"{title} {abstract}".strip()
            if combined:
                documents.append(combined)
                doc_ids.append(row.get("id", f"doc_{len(documents)}"))
        except Exception as e:
            print(f"❌ Error processing row: {e}")
    if len(documents) >= max_docs:
        break
print(f"\n✅ Loaded and cleaned {len(documents)} documents.")

# --- TF-IDF Processing ---
tfidf_model = IncrementalTfidf()
tfidf_model.add_documents(documents, doc_ids=doc_ids)

# --- Example Output ---
print("\n🔍 Sample TF-IDF terms from the first document:")
feature_names = tfidf_model.get_feature_names()
tfidf_matrix = tfidf_model.get_tfidf_matrix()

if tfidf_matrix.shape[0] > 0:
    first_doc_vector = tfidf_matrix[0].toarray().flatten()
    top_indices = first_doc_vector.argsort()[-10:][::-1]
    for idx in top_indices:
        print(f"  {feature_names[idx]}: {first_doc_vector[idx]:.4f}")
else:
    print("TF-IDF matrix is empty.")
    
       
