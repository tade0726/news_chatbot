"""
TODO
- a duckdb to store no sql json
- query data from https://serper.dev/
"""

# libs

import pandas as pd
from tavily import TavilyClient
from zenml import Model, step, pipeline
import os
from news_chatbot.datasets import DuckdbNewsDataset

from datetime import datetime

# %% parameters

## model parameters
MODEL_NAME = "news-query-pipeline"
MODEL_VERSION = "v0.1"


# tavily parameters
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# duckdb parameters
DUCKDB_PATH = os.getenv("DUCKDB_PATH")


# TOPICS
TOPICS = ["sports", "lifestyle", "music", "finance"]


# %%


@step(enable_cache=True)
def query_news(batch_query_time: int) -> pd.DataFrame:

    client = TavilyClient(TAVILY_API_KEY)
    
    dfs = []

    for topic in TOPICS:
        # query news
        query = f"Australian {topic}"
        
        response = client.search(
            query=query,
            topic="news",
            search_depth="advanced",
            max_results=50,
            time_range="month",
        )   

        # create dataframe
        df_result = pd.DataFrame(response["results"])
        df_result["topic"] = topic
        df_result["query"] = query
        df_result["model_name"] = MODEL_NAME
        df_result["model_version"] = MODEL_VERSION
        df_result["batch_query_time"] = batch_query_time
        
        dfs.append(df_result)
    
    return pd.concat(dfs)


@step(enable_cache=False)
def save_data_to_duckdb(df: pd.DataFrame, db_path: str) -> DuckdbNewsDataset:

    dataset = DuckdbNewsDataset(db_path, df)
    dataset.write_data()

    return dataset


# %%

model = Model(
    name=MODEL_NAME,
    version=MODEL_VERSION,
    description="query news from a unversal outlet",
    tags=["news", "raw", "data"],
)


@pipeline
def query_data_with_different_topics(batch_query_time: int):

    df_result = query_news(batch_query_time)
    save_data_to_duckdb(df_result, DUCKDB_PATH)


# %%

if __name__ == "__main__":
    _ = query_data_with_different_topics.with_options(model=model)(
        batch_query_time=int(datetime.now().timestamp())
    )
