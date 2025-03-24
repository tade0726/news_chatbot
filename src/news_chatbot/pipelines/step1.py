"""
News Query Pipeline

This module implements a pipeline for querying news data from NewsAPI and storing it in a DuckDB database.

Features:
- Fetches news articles across multiple topics using NewsAPI
- Stores results in a DuckDB database
- Implements ZenML pipeline for reproducible data processing
"""

# Standard libraries
import os
from datetime import datetime, timedelta

# Third-party libraries
import pandas as pd
from newsapi import NewsApiClient
from zenml import Model, step, pipeline

# Local imports
from news_chatbot.datasets import DuckdbDataset

# Model configuration
MODEL_NAME = "news-query-pipeline"
MODEL_VERSION = "v0.1"

# API configuration
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

# Database configuration
DUCKDB_PATH = os.getenv("DUCKDB_PATH")

# News topics to query
TOPICS = ["sports", "lifestyle", "music", "finance"]

# NewsAPI configuration
MAX_PAGES = 1  # Maximum number of pages to fetch per topic
ARTICLES_PER_PAGE = 100  # Maximum articles per page allowed by NewsAPI


@step(enable_cache=False)
def query_news(batch_query_time: int) -> pd.DataFrame:
    """
    Query news articles from NewsAPI for predefined topics with pagination.

    Args:
        batch_query_time: Unix timestamp for the current batch query

    Returns:
        DataFrame containing news articles with metadata
    """
    newsapi = NewsApiClient(api_key=NEWSAPI_KEY)
    dfs = []

    # Calculate date range (last 14 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=14)
    from_date = start_date.strftime("%Y-%m-%d")

    for topic in TOPICS:
        topic_articles = []

        # Construct query for current topic
        query = f"Australian {topic}"

        # Fetch multiple pages of results
        for page in range(1, MAX_PAGES + 1):
            try:
                # Fetch news articles from NewsAPI
                response = newsapi.get_everything(
                    q=query,
                    language="en",
                    sort_by="publishedAt",
                    from_param=from_date,
                    page=page,
                    page_size=ARTICLES_PER_PAGE,
                )

                # Check if we have articles
                if response["status"] == "ok" and response["articles"]:
                    topic_articles.extend(response["articles"])

                    # If we received fewer articles than requested, we've reached the end
                    if len(response["articles"]) < ARTICLES_PER_PAGE:
                        break
                else:
                    # No more articles or error
                    break

            except Exception as e:
                print(f"Error fetching page {page} for topic '{topic}': {e}")
                break

        if topic_articles:
            # Create dataframe and add metadata
            df_result = pd.DataFrame(topic_articles)
            df_result["topic"] = topic
            df_result["query"] = query
            df_result["model_name"] = MODEL_NAME
            df_result["model_version"] = MODEL_VERSION
            df_result["batch_query_time"] = batch_query_time

            dfs.append(df_result)

    return pd.concat(dfs) if dfs else pd.DataFrame()


@step(enable_cache=False)
def save_data_to_duckdb(df: pd.DataFrame, db_path: str) -> DuckdbDataset:
    """
    Save news data to DuckDB database.

    Args:
        df: DataFrame containing news articles
        db_path: Path to DuckDB database file

    Returns:
        DuckdbDataset instance with the saved data
    """
    dataset = DuckdbDataset(db_path, df, table_name="news")
    dataset.write_data()

    return dataset


# Create model metadata for tracking
model = Model(
    name=MODEL_NAME,
    version=MODEL_VERSION,
    description="Query news from various topics using NewsAPI",
    tags=["news", "raw", "data"],
)


@pipeline
def query_data_with_different_topics(batch_query_time: int):
    """
    Pipeline to query news data and save to DuckDB.

    Args:
        batch_query_time: Unix timestamp for the current batch query
    """
    df_result = query_news(batch_query_time)
    save_data_to_duckdb(df_result, DUCKDB_PATH)


if __name__ == "__main__":
    # Execute the pipeline with current timestamp
    _ = query_data_with_different_topics.with_options(model=model)(
        batch_query_time=int(datetime.now().timestamp())
    )
