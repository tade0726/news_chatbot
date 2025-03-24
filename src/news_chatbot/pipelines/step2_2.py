# Standard libraries
import os
from datetime import datetime
from typing import List, Dict, Any

# Third-party libraries
import pandas as pd
import numpy as np
from zenml import Model, step, pipeline
import faiss


# Local imports
from news_chatbot.datasets import DuckdbDataset


# Model configuration
MODEL_NAME = "deduplication-pipeline"
MODEL_VERSION = "v0.1"

# API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Database configuration
DUCKDB_PATH = os.getenv("DUCKDB_PATH")

# TMP folder
TMP_FOLDER = os.getenv("TMP_FOLDER")


@step
def deduplicate_step(batch_query_time: int) -> pd.DataFrame:

    def deduplicate(embeddings, titles):

        # Build FAISS index
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        # Search for top-k similar titles
        D, I = index.search(embeddings, k=5)  # Top 5 similar items for each title

        # Post-process to remove near-duplicates
        threshold = 0.9
        visited = set()
        unique_indices = []
        duplicate_counts = []  # Track number of duplicates for each unique item

        for idx, neighbors in enumerate(I):
            if idx in visited:
                continue

            # Count similar items (including self)
            similar_count = 1  # Start with 1 for self
            for i in range(1, len(neighbors)):
                if D[idx][i] >= threshold and I[idx][i] != idx:
                    similar_count += 1

            unique_indices.append(idx)
            duplicate_counts.append(similar_count)

            # Mark all neighbors as visited
            visited.add(idx)
            for i in range(1, len(neighbors)):
                if D[idx][i] >= threshold:
                    visited.add(I[idx][i])

        deduplicated_titles = [titles[i] for i in unique_indices]

        return deduplicated_titles, unique_indices, duplicate_counts, D, I

    # Save the enriched data to DuckDB
    dataset = DuckdbDataset(DUCKDB_PATH, table_name="enriched_news")
    df = dataset.read_data()

    # remove simple url duplicates
    df2 = df.drop_duplicates(subset=["url"])
    df2.reset_index(drop=True, inplace=True)

    # vector conversion
    def vector(long_str):
        return np.array(
            [float(x) for x in long_str.lstrip("'[").rstrip("]'").split(",")]
        )

    df2.loc[:, "title_embedding"] = df2["title_embedding"].apply(vector)

    # deduplicate
    titles = df2.title
    embeddings = np.stack(df2["title_embedding"].values)

    _, unique_indices, duplicate_counts, _, _ = deduplicate(embeddings, titles)

    # save deduplicated data
    df3 = df2.iloc[unique_indices].reset_index(drop=True)

    # add batch query time
    df3["batch_query_time"] = batch_query_time

    # add duplicate counts
    df3["duplicate_count"] = duplicate_counts

    # add id
    df3.loc[:, "id"] = df3.index

    # save to duckdb
    dataset = DuckdbDataset(
        DUCKDB_PATH, df3, table_name="deduplicated_news", overwrite=True
    )
    dataset.write_data()

    return df3


# Create model metadata for tracking
model = Model(
    name=MODEL_NAME,
    version=MODEL_VERSION,
    description="Deduplicate news data",
    tags=["news", "deduplicated", "data"],
)


@pipeline
def deduplicate_news(batch_query_time: int):

    df = deduplicate_step(batch_query_time)

    return df


if __name__ == "__main__":
    # Execute the pipeline with current timestamp
    _ = deduplicate_news.with_options(model=model)(
        batch_query_time=int(datetime.now().timestamp())
    )
