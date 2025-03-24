from openai import AsyncOpenAI
import json
from typing import Dict, Any, List
import asyncio

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


async def process_article_content(
    client: AsyncOpenAI, content: str, system_prompt: str
) -> Dict[str, Any]:
    """
    Process a single article's content using OpenAI API asynchronously.

    Args:
        client: AsyncOpenAI client
        content: Article content to process
        system_prompt: System prompt to use

    Returns:
        Dictionary with extracted information
    """
    try:
        # Make API request to OpenAI
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
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


async def process_article_embedding(
    client: AsyncOpenAI, title: str, *args, **kwargs
) -> List[float]:
    """
    Generate embedding for a single article title using OpenAI API asynchronously.

    Args:
        client: AsyncOpenAI client
        title: Article title to embed
        *args: Additional arguments
        **kwargs: Additional keyword arguments

    Returns:
        List of embedding values or None if failed
    """
    try:
        if title and isinstance(title, str):
            # Generate embedding for the title
            response = await client.embeddings.create(
                model="text-embedding-ada-002", input=title
            )

            # Extract the embedding
            return response.data[0].embedding
        return None
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None


async def process_items_with_semaphore(
    items_to_process,
    process_func,
    client: AsyncOpenAI,
    system_prompt: str = None,
    max_concurrency: int = 5,
    **kwargs,
):
    """
    Process a batch of items using the provided function asynchronously with a semaphore
    to control concurrency.

    Args:
        items_to_process: Dictionary of items to process
        process_func: Async function to call for each item
        client: AsyncOpenAI client
        system_prompt: System prompt for the process function
        max_concurrency: Maximum number of concurrent requests
        **kwargs: Additional arguments to pass to process_func

    Returns:
        Dictionary mapping item identifiers to results
    """
    results = {}
    semaphore = asyncio.Semaphore(max_concurrency)

    async def process_with_semaphore(item_id, item_data):
        async with semaphore:
            result = await process_func(
                client=client, system_prompt=system_prompt, **item_data
            )
            # Add a small delay to avoid rate limiting
            await asyncio.sleep(0.2)
            return item_id, result

    # Create tasks for all items
    tasks = [
        asyncio.create_task(process_with_semaphore(item_id, item_data))
        for item_id, item_data in items_to_process.items()
    ]

    # Wait for all tasks to complete
    for task in asyncio.as_completed(tasks):
        item_id, result = await task
        results[item_id] = result

    return results
