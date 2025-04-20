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
nltk_data_path = r"C:\Users\Saad\AppData\Roaming\nltk_data"
if nltk_data_path not in nltk.data.path:
    nltk.data.path.insert(0, nltk_data_path)

# Check and download required resources
for resource in ["punkt", "stopwords", "wordnet"]:
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

# --- Paths ---
abstracts_path = r"C:\Users\Saad\Downloads\Porashuna TU WIEN Summer 2025\AIR\longeval_sci_training_2025_abstract\documents"

# --- Load, Preprocess, and Add to TF-IDF ---
documents = []
doc_ids = []
max_docs = 50  # Adjust for testing

for file_name in tqdm(os.listdir(abstracts_path), desc="Loading files"):
    if not file_name.endswith(".jsonl"):
        continue

    file_path = os.path.join(abstracts_path, file_name)
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(documents) >= max_docs:
                break
            try:
                data = json.loads(line)
                title = clean_text(data.get("title", ""))
                abstract = clean_text(data.get("abstract", ""))
                combined = f"{title} {abstract}".strip()
                if combined:
                    documents.append(combined)
                    doc_ids.append(data.get("id", f"doc_{len(documents)}"))
            except Exception as e:
                print(f"❌ Error processing line: {e}")
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
