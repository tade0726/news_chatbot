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
MODEL_NAME = "news-ui-metadata-pipeline"
MODEL_VERSION = "v0.1"


# data folder
DATA_FOLDER = os.getenv("DATA_FOLDER")


@step
def metadata(batch_query_time: int) -> pd.DataFrame:

    def curating_attr_build_json(df):

        columns = [
            "source",
            "author",
            "headline",
            "summary",
            "keywords",
            "rrf_rank",
            "sentiment",
            "publishedAt",
            "duplicate_count",
        ]

        df2 = df[columns]

        df2.loc[:, "source"] = df2["source"].apply(lambda x: x["name"])

        return df2.sort_values("rrf_rank", ascending=True)

    dfs = []

    for x in ["sports", "lifestyle", "music", "finance"]:

        dataset = DuckdbDataset(DUCKDB_PATH, table_name=f"{x}_news")

        df = dataset.read_data()

        df_clean = curating_attr_build_json(df)

        df_clean.loc[:, "topic"] = x

        dfs.append(df_clean)

    df_res = pd.concat(dfs, axis=0)

    df_res.reset_index(drop=True, inplace=True)

    df_ui_news = df_res[df_res["rrf_rank"] <= 15]

    df_ui_news.to_parquet(f"{DATA_FOLDER}/news_ui_metadata.parquet")
    df_res.to_parquet(f"{DATA_FOLDER}/news_metadata.parquet")

    return df_ui_news


# Create model metadata for tracking
model = Model(
    name=MODEL_NAME,
    version=MODEL_VERSION,
    description="Generate metadata for news articles",
    tags=["news", "data", "metadata"],
)


@pipeline
def metadata_pipeline(batch_query_time: int):
    return metadata(batch_query_time)


if __name__ == "__main__":
    # Execute the pipeline with current timestamp
    _ = metadata_pipeline.with_options(model=model)(
        batch_query_time=int(datetime.now().timestamp())
    )
