from openai import OpenAI
import json
from typing import Dict, Any, List


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


def process_article_content(
    client: OpenAI, content: str, system_prompt: str
) -> Dict[str, Any]:
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


def process_batch_of_items(items_to_process, process_func, client: OpenAI, **kwargs):
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
