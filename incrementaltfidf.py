from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from scipy.sparse import vstack
from tqdm import tqdm


# @Santiago Ruiz: These lines of code iterates through each document
#     Creates a token with maximam 100_000 and then sees whether
#      the abstract contains the word, and then adds a new column,
#      and updates the counts for the tokens analyzed previously.

class IncrementalTfidf:
    def __init__(self, max_features=100_000):
        self.vectorizer = CountVectorizer(max_features=max_features)
        self.tfidf_transformer = TfidfTransformer()
        self.count_matrix = None
        self.tfidf_matrix = None
        self.doc_ids = []
        self.fitted = False
        
    def _sanitize_documents(self, documents):
        """Convert all documents to string and handle None values."""
        sanitized = []
        for doc in documents:
            if doc is None:
                sanitized.append("")
            elif not isinstance(doc, str):
                sanitized.append(str(doc))
            else:
                sanitized.append(doc)
        return sanitized

    def add_documents(self, documents, doc_ids=None):
        """
        Add new documents to the corpus and update TF-IDF matrix.
        
        Args:
            documents (list of str): List of raw text documents.
            doc_ids (list): Optional list of IDs for the documents.
        """
        # Step 1: Vectorize the new batch of documents
        documents = self._sanitize_documents(documents)
        
        if not self.fitted:
            count_batch = self.vectorizer.fit_transform(documents)
            self.fitted = True
        else:
            count_batch = self.vectorizer.transform(documents)

        # Step 2: Update the count matrix
        if self.count_matrix is None:
            self.count_matrix = count_batch
        else:
            self.count_matrix = vstack([self.count_matrix, count_batch])

        # Step 3: Update TF-IDF matrix
        self.tfidf_matrix = self.tfidf_transformer.fit_transform(self.count_matrix)

        # Step 4: Track document IDs
        if doc_ids is None:
            doc_ids = list(range(len(self.doc_ids), len(self.doc_ids) + len(documents)))
        self.doc_ids.extend(doc_ids)

    def get_tfidf_matrix(self):
        return self.tfidf_matrix

    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()

    def get_document_ids(self):
        return self.doc_ids

