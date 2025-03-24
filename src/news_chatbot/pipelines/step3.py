"""
How to rank the importance of the news in each category
- frequency of the article (duplicate_count)
- keywords that represent the importance of each category
- Reciprocal Rank Fusion (RRF): combine the frequency and keywords to rank the importance of the news
"""

# Standard libraries
import os
from datetime import datetime
from typing import List, Dict, Any

# Third-party libraries
import pandas as pd
from zenml import Model, step, pipeline

import json
from openai import OpenAI
import time

from sentence_transformers import SentenceTransformer
import numpy as np

from news_chatbot.datasets import DuckdbDataset


# Database configuration
DUCKDB_PATH = os.getenv("DUCKDB_PATH")

# Model configuration
MODEL_NAME = "high-importance-pipeline"
MODEL_VERSION = "v0.1"


@step
def high_importance(batch_query_time: int) -> Dict[str, pd.DataFrame]:
    """
    using faiss to generate embeddings for the news articles:
    """

    def normalize(vectors):
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def compute_rrf(df, rank_cols, k=60):
        """
        Add a new column 'rrf_score' to the dataframe using Reciprocal Rank Fusion.

        Parameters:
        - df: pandas DataFrame
        - rank_cols: list of column names containing ranks (int, 0-based)
        - k: RRF damping constant (default=60)

        Returns:
        - DataFrame with new column 'rrf_score'
        """
        df = df.copy()
        df["rrf_score"] = 0.0
        for col in rank_cols:
            df["rrf_score"] += 1 / (k + df[col])
        return df["rrf_score"]

    # retrieve dataset
    dataset = DuckdbDataset(DUCKDB_PATH, table_name="deduplicated_news")

    df = dataset.read_data()

    # cleaning format
    df.loc[:, "categories"] = df["categories"].apply(
        lambda x: x.replace("[", "").replace('"', "").replace("]", "").split(",")
    )

    # combine headlines + summary
    df.loc[:, "combined_text"] = df["headline"] + "\n\n" + df["summary"]

    # define keywords for different categories
    # Each topic is represented by 3 core keywords + 'breaking news'
    topic_keywords = {
        "sports": ["athlete", "tournament", "score", "breaking news"],
        "lifestyle": ["wellness", "fashion", "routine", "breaking news"],
        "music": ["concert", "album", "artist", "breaking news"],
        "finance": ["investment", "market", "economy", "breaking news"],
    }

    # create embedding
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Embed all titles
    combined_text_embeddings = model.encode(df["combined_text"], convert_to_numpy=True)

    # Create 1 embedding per topic by joining keywords into a string
    topic_texts = [" ".join(words) for words in topic_keywords.values()]
    topic_embeddings = model.encode(topic_texts, convert_to_numpy=True)

    combined_text_embeddings = normalize(combined_text_embeddings)
    topic_embeddings = normalize(topic_embeddings)

    # Create similarity matrix
    similarity_matrix = np.matmul(combined_text_embeddings, topic_embeddings.T)
    df.loc[
        :,
        [
            "sports_importance",
            "lifestyle_importance",
            "music_importance",
            "finance_importance",
        ],
    ] = similarity_matrix

    # compute rrf for each topic, combine their feq_rank and similarity_score to the topic keywords
    dfs = {}

    for t in topic_keywords.keys():

        df_topic = df[df["topic"].apply(lambda x: t in x)].copy()

        df_topic.loc[:, f"{t}_rank"] = df_topic[f"{t}_importance"].rank(
            ascending=False, method="min"
        )
        df_topic.loc[:, "feq_rank"] = df_topic["duplicate_count"].rank(
            ascending=False, method="min"
        )

        df_topic.loc[:, "rrf_score"] = compute_rrf(df_topic, [f"{t}_rank", "feq_rank"])
        df_topic.loc[:, "rrf_rank"] = df_topic["rrf_score"].rank(
            ascending=False, method="min"
        )

        df_topic.loc[:, "batch_query_time"] = batch_query_time

        dfs[t] = df_topic

    # store each dataframe to its own table

    for t, df_topic in dfs.items():
        dataset = DuckdbDataset(
            DUCKDB_PATH, df=df_topic, table_name=f"{t}_news", overwrite=True
        )
        dataset.write_data()

    return dfs


# Create model metadata for tracking
model = Model(
    name=MODEL_NAME,
    version=MODEL_VERSION,
    description="Process news data to rank the importance of the news in each category",
    tags=["news", "data", "high-importance"],
)


@pipeline
def highlight_news(batch_query_time: int):

    dfs = high_importance(batch_query_time)

    return dfs


if __name__ == "__main__":
    # Execute the pipeline with current timestamp
    _ = highlight_news.with_options(model=model)(
        batch_query_time=int(datetime.now().timestamp())
    )
