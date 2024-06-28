import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor
from scipy.sparse import csr_matrix


class Index:
    """
    A simple search index using TF-IDF and cosine similarity for text fields and exact matching for keyword fields.

    Attributes:
        text_fields (list): List of text field names to index.
        keyword_fields (list): List of keyword field names to index.
        vectorizers (dict): Dictionary of TfidfVectorizer instances for each text field.
        keyword_df (pd.DataFrame): DataFrame containing keyword field data.
        text_matrices (dict): Dictionary of TF-IDF matrices for each text field.
        docs (list): List of documents indexed.
    """
    def __init__(self, text_fields,
                 keyword_fields,
                 vectorizer_params = None):
        if vectorizer_params is None:
            vectorizer_params = {}
        self.text_fields = text_fields
        self.keyword_fields = keyword_fields
        self.vectorizers = {field: TfidfVectorizer(**vectorizer_params) for field in text_fields}
        self.keyword_df = None
        self.text_matrices = {}
        self.docs = []

    def fit(self, docs):
        """
            Fits the index with the provided documents.

            Args:
                docs (list of dict): List of documents to index. Each document is a dictionary.
        """
        self.docs = docs
        keyword_data = {field: [] for field in self.keyword_fields}

        def fit_text_field(field):
            texts = [doc.get(field, '') for doc in docs]
            return field, self.vectorizers[field].fit_transform(texts)

        with ThreadPoolExecutor() as executor:
            results = executor.map(fit_text_field, self.text_fields)
            for field, matrix in results:
                self.text_matrices[field] = matrix

        for doc in docs:
            for field in self.keyword_fields:
                keyword_data[field].append(doc.get(field, ''))

        self.keyword_df = pd.DataFrame(keyword_data)
        return self

    def search(self, query, filter_dict = {}, boost_dict = {}, num_results = 10):
        """
        Searches the index with the given query, filters, and boost parameters.

        Args:
            query (str): The search query string.
            filter_dict (dict): Dictionary of keyword fields to filter by. Keys are field names and values are the values to filter by.
            boost_dict (dict): Dictionary of boost scores for text fields. Keys are field names and values are the boost scores.
            num_results (int): The number of top results to return. Defaults to 10.

        Returns:
            list of dict: List of documents matching the search criteria, ranked by relevance.
        """
        query_vecs = {field: self.vectorizers[field].transform([query]) for field in self.text_fields}
        scores = np.zeros(len(self.docs))

        def compute_similarity(field, query_vec):
            sim = cosine_similarity(query_vec, self.text_matrices[field]).flatten()
            boost = boost_dict.get(field, 1)
            return sim * boost

        with ThreadPoolExecutor() as executor:
            similarities = executor.map(lambda field: compute_similarity(field, query_vecs[field]), self.text_fields)
            for sim in similarities:
                scores += sim

        for field, value in filter_dict.items():
            if field in self.keyword_fields:
                mask = self.keyword_df[field] == value
                scores = scores * mask.to_numpy()

        top_indices = np.argpartition(scores, -num_results)[-num_results:]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        top_docs = [self.docs[i] for i in top_indices if scores[i] > 0]

        return top_docs
