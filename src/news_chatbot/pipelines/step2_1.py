"""
Categories and summary pipeline

Features:
- Using OpenAI API to generate categories and summaries for news articles
- Stores results in a DuckDB database
"""

# Standard libraries
import os
from datetime import datetime

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


## SYSTEM PROMPT
SYSTEM_PROMPT = """
You are an expert news analyst that provides concise yet comprehensive summaries of news articles.

Return your response in JSON format with the following structure:
{
    "categories": [list of categories that apply to this article from the following options: "sports", "lifestyle", "music", "finance", "politics", "technology", "health", "science", "entertainment", "education"],
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
    df["published_date"] = pd.to_datetime(
        df["published_date"], format="%a, %d %b %Y %H:%M:%S %Z", utc=True
    )

    # convert batch query time to datetime
    df["step1_batch_query_time"] = df["batch_query_time"]

    # keep only the nessary columns
    df = df[
        [
            "url",
            "raw_content",
            "title",
            "published_date",
            "content",
            "topic",
            "model_name",
            "model_version",
            "step1_batch_query_time",
        ]
    ]

    # drop duplicates
    df = df.drop_duplicates(subset=["url"])

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


@step
def categories_and_summary(df: pd.DataFrame, batch_query_time: int) -> str:

    rows = []

    for _, row in df.head(3).iterrows():
        template = {
            "custom_id": f"{row['id']}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {"role": "user", "content": f"{row['title']}\n{row['content']}"},
                ],
                "max_tokens": 2000,
            },
        }

        rows.append(template)

    with open(TMP_FOLDER + f"/batch_{batch_query_time}.jsonl", "w") as f:
        f.writelines([json.dumps(row) + "\n" for row in rows])

    return TMP_FOLDER + f"/batch_{batch_query_time}.jsonl"


@step
def request_openai_api(batch_file_path: str, batch_query_time: int):
    client = OpenAI()

    batch_input_file = client.files.create(
        file=open(batch_file_path, "rb"), purpose="batch"
    )

    batch_input_file_id = batch_input_file.id

    batch = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "categories and summary"},
    )

    while True:
        batch_status = client.batches.retrieve(batch.id)
        if batch_status.status == "completed" or batch_status.status == "failed":
            break
        time.sleep(30)

    if batch_status.status == "failed":
        raise Exception("Batch failed")

    file_response = client.files.content(batch_status.output_file_id)

    results_file_path = TMP_FOLDER + f"/summary_batch_results_{batch_query_time}.jsonl"
    with open(results_file_path, "w") as f:
        f.writelines(file_response.text)

    return results_file_path


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
    batch_file_path = categories_and_summary(df, batch_query_time)
    batch_results_path = request_openai_api(batch_file_path, batch_query_time)

    return batch_results_path


if __name__ == "__main__":
    # Execute the pipeline with current timestamp
    _ = process_news_data.with_options(model=model)(
        batch_query_time=int(datetime.now().timestamp())
    )
