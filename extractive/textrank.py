"""
TextRank-based Extractive Summarizer
-----------------------------------
Classical NLP baseline for comparison with Transformer models.
"""

import nltk
import networkx as nx
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download tokenizer (first run only)
nltk.download("punkt", quiet=True)


def textrank_summarize(text, top_n=5):
    """
    Generate extractive summary using TextRank.

    Args:
        text (str): Full article text
        top_n (int): Number of sentences in summary

    Returns:
        str: Extractive summary
    """

    # Sentence segmentation
    sentences = sent_tokenize(text)

    if len(sentences) <= top_n:
        return text

    # TF-IDF sentence embeddings
    vectorizer = TfidfVectorizer(stop_words="english")
    sentence_vectors = vectorizer.fit_transform(sentences)

    # Similarity matrix
    similarity_matrix = cosine_similarity(sentence_vectors)

    # Build graph & apply PageRank
    graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(graph)

    # Rank sentences
    ranked_sentences = sorted(
        ((scores[i], s) for i, s in enumerate(sentences)),
        reverse=True
    )

    # Select top-n sentences
    selected_sentences = [s for _, s in ranked_sentences[:top_n]]

    # Preserve original order
    summary = " ".join([s for s in sentences if s in selected_sentences])

    return summary


# Simple CLI test
if __name__ == "__main__":
    sample_text = """
    Artificial intelligence is transforming industries across the world.
    Companies are investing heavily in machine learning solutions.
    However, ethical concerns remain about data privacy and bias.
    Governments are now introducing regulations to control AI development.
    Experts believe collaboration is key to responsible innovation.
    """

    print("🔹 Extractive Summary:\n")
    print(textrank_summarize(sample_text, top_n=3))
