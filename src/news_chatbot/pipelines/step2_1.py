"""
Categories and summary pipeline

Features:
- Using OpenAI API to generate categories and summaries for news articles
- Stores results in a DuckDB database
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

# Local imports
from news_chatbot.datasets import DuckdbDataset


# Model configuration
MODEL_NAME = "categories-and-summary-pipeline"
MODEL_VERSION = "v0.1"

# API configuration
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Database configuration
DUCKDB_PATH = os.getenv("DUCKDB_PATH")

# TMP folder
TMP_FOLDER = os.getenv("TMP_FOLDER")

# Concurrency settings
BATCH_SIZE = 10  # Adjust based on OpenAI rate limits


## SYSTEM PROMPT
SYSTEM_PROMPT = """
You are an expert news analyst that provides concise yet comprehensive summaries of news articles.

Return your response in JSON format with the following structure:
{
    "categories": [list of categories that apply to this article from the following options: "sports", "lifestyle", "music", "finance"],
    "summary": "a detailed summary that covers the key points, main actors, important events, and significant implications of the article. Include relevant dates, statistics, and quotes if present",
    "keywords": [list of 5-8 important keywords from the article],
    "key_entities": [list of main people, organizations, or places mentioned],
    "sentiment": "overall sentiment of the article (positive, negative, or neutral)",
    "headline": "a concise headline that captures the essence of the article"
}

Do not include any explanations, just the JSON.
"""


# prepare batch files


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

    return df


def process_article_content(client: OpenAI, content: str) -> Dict[str, Any]:
    """
    Process a single article's content using OpenAI API.

    Args:
        client: OpenAI client
        content: Article content to process

    Returns:
        Dictionary with extracted information
    """
    try:
        # Make API request to OpenAI
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            temperature=0.3,
            max_tokens=2000,
        )

        # Extract the response
        result_text = response.choices[0].message.content

        result_text = result_text.lstrip("```json").rstrip("```")

        # Parse the JSON response
        try:
            result = json.loads(result_text)
            return {
                "categories": json.dumps(result.get("categories", [])),
                "summary": result.get("summary", ""),
                "keywords": json.dumps(result.get("keywords", [])),
                "key_entities": json.dumps(result.get("key_entities", [])),
                "sentiment": result.get("sentiment", ""),
                "headline": result.get("headline", ""),
                "success": True,
            }
        except json.JSONDecodeError:
            print(f"Failed to parse JSON: {result_text}")
            return {
                "categories": "[]",
                "summary": "",
                "keywords": "[]",
                "key_entities": "[]",
                "sentiment": "",
                "headline": "",
                "success": False,
            }
    except Exception as e:
        print(f"Error processing article: {e}")
        return {
            "categories": "[]",
            "summary": "",
            "keywords": "[]",
            "key_entities": "[]",
            "sentiment": "",
            "headline": "",
            "success": False,
        }


def process_article_embedding(client: OpenAI, title: str) -> List[float]:
    """
    Generate embedding for a single article title using OpenAI API.

    Args:
        client: OpenAI client
        title: Article title to embed

    Returns:
        List of embedding values or None if failed
    """
    try:
        if title and isinstance(title, str):
            # Generate embedding for the title
            response = client.embeddings.create(
                model="text-embedding-ada-002", input=title
            )

            # Extract the embedding
            return response.data[0].embedding
        return None
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None


def process_batch_of_items(items_to_process, process_func, client, **kwargs):
    """
    Process a batch of items using the provided function.

    Args:
        items_to_process: List of items to process
        process_func: Function to call for each item
        client: OpenAI client
        **kwargs: Additional arguments to pass to process_func

    Returns:
        Dictionary mapping item identifiers to results
    """
    results = {}

    for item_id, item_data in items_to_process.items():
        result = process_func(client=client, **item_data)
        results[item_id] = result

        # Add a small delay to avoid rate limiting
        time.sleep(0.2)

    return results


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
    client = OpenAI(api_key=OPENAI_API_KEY)

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

    # Split items into batches
    item_ids = list(content_items.keys())
    for i in range(0, len(item_ids), BATCH_SIZE):
        batch_ids = item_ids[i : i + BATCH_SIZE]
        batch_items = {item_id: content_items[item_id] for item_id in batch_ids}

        # Process the batch
        batch_results = process_batch_of_items(
            batch_items, process_article_content, client
        )
        all_results.update(batch_results)

        # Add a delay between batches to avoid rate limiting
        if i + BATCH_SIZE < len(item_ids):
            print(
                f"Processed batch {i//BATCH_SIZE + 1}/{(len(item_ids) + BATCH_SIZE - 1)//BATCH_SIZE}, waiting before next batch..."
            )
            time.sleep(1)

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
    client = OpenAI(api_key=OPENAI_API_KEY)

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

    # Split items into batches
    item_ids = list(title_items.keys())
    for i in range(0, len(item_ids), BATCH_SIZE):
        batch_ids = item_ids[i : i + BATCH_SIZE]
        batch_items = {item_id: title_items[item_id] for item_id in batch_ids}

        # Process the batch
        batch_results = process_batch_of_items(
            batch_items, process_article_embedding, client
        )
        all_results.update(batch_results)

        # Add a delay between batches to avoid rate limiting
        if i + BATCH_SIZE < len(item_ids):
            print(
                f"Processed embeddings batch {i//BATCH_SIZE + 1}/{(len(item_ids) + BATCH_SIZE - 1)//BATCH_SIZE}, waiting before next batch..."
            )
            time.sleep(1)

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
