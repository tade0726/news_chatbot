"""
Categories and summary pipeline

Features:
- Using OpenAI API to generate categories and summaries for news articles
- Stores results in a DuckDB database
"""

# Standard libraries
import os
from datetime import datetime
import asyncio

# Third-party libraries
import pandas as pd
from zenml import Model, step, pipeline

import json
from news_chatbot.llm import (
    process_article_content,
    process_article_embedding,
    process_items_with_semaphore,
    SYSTEM_PROMPT,
)
from news_chatbot.datasets import DuckdbDataset
from openai import AsyncOpenAI

# Model configuration
MODEL_NAME = "categories-and-summary-pipeline"
MODEL_VERSION = "v0.1"

# API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Database configuration
DUCKDB_PATH = os.getenv("DUCKDB_PATH")

# TMP folder
TMP_FOLDER = os.getenv("TMP_FOLDER")

# Concurrency settings
BATCH_SIZE = 50  # Adjust based on OpenAI rate limits

# INPUT_NUM
INPUT_NUM = -1


@step
def clean_data(batch_query_time: int) -> pd.DataFrame:

    # read duckdb
    dataset = DuckdbDataset(DUCKDB_PATH, "news")
    df = dataset.read_data()

    # date format
    df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")

    # If any dates failed to parse, try the ISO 8601 format explicitly
    mask = df["publishedAt"].isna()

    if mask.any():
        df.loc[mask, "publishedAt"] = pd.to_datetime(
            df.loc[mask, "publishedAt"], format="%Y-%m-%dT%H:%M:%SZ", utc=True
        )

    # convert batch query time to datetime
    df["step1_batch_query_time"] = df["batch_query_time"]

    # drop duplicates
    df = df.drop_duplicates(subset=["url", "topic"])

    # add batch query time
    df["batch_query_time"] = batch_query_time

    # add id
    df["id"] = df.reset_index(drop=True).index

    # save to duckdb
    dataset2 = DuckdbDataset(
        DUCKDB_PATH, df, table_name="processed_news", overwrite=True
    )
    dataset2.write_data()

    return df[:INPUT_NUM] if INPUT_NUM > 0 else df


@step
def categories_and_summary(df: pd.DataFrame, batch_query_time: int) -> pd.DataFrame:
    """
    Process news articles to extract categories and summaries using OpenAI API.

    Args:
        df: DataFrame containing news articles
        batch_query_time: Unix timestamp for the current batch query

    Returns:
        DataFrame with added categories, summaries, and other enrichments
    """
    # Initialize OpenAI client
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    # Create new columns for the enriched data
    df["categories"] = None
    df["summary"] = None
    df["keywords"] = None
    df["key_entities"] = None
    df["sentiment"] = None
    df["headline"] = None

    # Prepare content for each article
    content_items = {}
    for idx, row in df.iterrows():
        content = f"Title: {row['title']}\n\nDescription: {row['description']}\n\nContent: {row['content']}"
        content_items[str(idx)] = {"content": content}

    # Process articles in batches
    all_results = {}

    # Define async function to process all batches
    async def process_all_batches():
        nonlocal all_results
        # Split items into batches
        item_ids = list(content_items.keys())
        for i in range(0, len(item_ids), BATCH_SIZE):
            batch_ids = item_ids[i : i + BATCH_SIZE]
            batch_items = {item_id: content_items[item_id] for item_id in batch_ids}

            # Process the batch
            batch_results = await process_items_with_semaphore(
                batch_items,
                process_article_content,
                client,
                system_prompt=SYSTEM_PROMPT,
                max_concurrency=5,
            )
            all_results.update(batch_results)

            # Add a delay between batches to avoid rate limiting
            if i + BATCH_SIZE < len(item_ids):
                print(
                    f"Processed batch {i//BATCH_SIZE + 1}/{(len(item_ids) + BATCH_SIZE - 1)//BATCH_SIZE}, waiting before next batch..."
                )
                await asyncio.sleep(1)

    # Run the async function
    asyncio.run(process_all_batches())

    # Update the DataFrame with the results
    for idx_str, result in all_results.items():
        idx = int(idx_str)
        df.at[idx, "categories"] = result["categories"]
        df.at[idx, "summary"] = result["summary"]
        df.at[idx, "keywords"] = result["keywords"]
        df.at[idx, "key_entities"] = result["key_entities"]
        df.at[idx, "sentiment"] = result["sentiment"]
        df.at[idx, "headline"] = result["headline"]

    return df


@step
def embeddings(df: pd.DataFrame, batch_query_time: int) -> pd.DataFrame:
    """
    Generate embeddings for news article titles using OpenAI API.

    Args:
        df: DataFrame containing news articles
        batch_query_time: Unix timestamp for the current batch query

    Returns:
        DataFrame with added embedding column
    """
    # Initialize OpenAI client
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    # Create new column for embeddings
    df["title_embedding"] = None

    # Prepare titles for each article
    title_items = {}
    for idx, row in df.iterrows():
        title = row["title"]
        if title and isinstance(title, str):
            title_items[str(idx)] = {"title": title}

    # Process titles in batches
    all_results = {}

    # Define async function to process all batches
    async def process_all_embeddings():
        nonlocal all_results
        # Split items into batches
        item_ids = list(title_items.keys())
        for i in range(0, len(item_ids), BATCH_SIZE):
            batch_ids = item_ids[i : i + BATCH_SIZE]
            batch_items = {item_id: title_items[item_id] for item_id in batch_ids}

            # Process the batch
            batch_results = await process_items_with_semaphore(
                batch_items, process_article_embedding, client, max_concurrency=10
            )
            all_results.update(batch_results)

            # Add a delay between batches to avoid rate limiting
            if i + BATCH_SIZE < len(item_ids):
                print(
                    f"Processed embeddings batch {i//BATCH_SIZE + 1}/{(len(item_ids) + BATCH_SIZE - 1)//BATCH_SIZE}, waiting before next batch..."
                )
                await asyncio.sleep(1)

    # Run the async function
    asyncio.run(process_all_embeddings())

    # Update the DataFrame with the results
    for idx_str, embedding in all_results.items():
        if embedding is not None:
            idx = int(idx_str)
            df.at[idx, "title_embedding"] = json.dumps(embedding)

    # Save the enriched data to DuckDB
    dataset = DuckdbDataset(
        DUCKDB_PATH, df, table_name="enriched_news", overwrite=False
    )
    dataset.write_data()

    return df


# Create model metadata for tracking
model = Model(
    name=MODEL_NAME,
    version=MODEL_VERSION,
    description="Process news data using OpenAI API to generate categories, summaries, and other enrichments",
    tags=["news", "processed", "data", "enriched"],
)


@pipeline
def process_news_data(batch_query_time: int):

    df = clean_data(batch_query_time)
    df = categories_and_summary(df, batch_query_time)
    df = embeddings(df, batch_query_time)

    return df


if __name__ == "__main__":
    # Execute the pipeline with current timestamp
    _ = process_news_data.with_options(model=model)(
        batch_query_time=int(datetime.now().timestamp())
    )
