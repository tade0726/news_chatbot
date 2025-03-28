import json
import logging
import os

import streamlit as st
from llama_index.core import PromptTemplate, Settings, StorageContext, VectorStoreIndex
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from openai import OpenAI
from qdrant_client import QdrantClient

import pandas as pd
import torch


torch.classes.__path__ = []  # add this line to manually set it to empty.


# config
QDRANT_URI = st.secrets["QDRANT_URI"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
QDRANT_COLLECTION_NAME = st.secrets["QDRANT_COLLECTION_NAME"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
LLM_MODEL = st.secrets["LLM_MODEL"]
TOP_K = 7


# config
DATA_FOLDER = os.path.join(os.path.dirname(__file__), "data")


# create logger, setting basic formatting
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# prompt template

SYSTEM_PROMPT = """
You are an expert assistant specializing in Australian news across four key categories: sports, lifestyle, music, and finance. Your primary role is to provide accurate, factual information by carefully analyzing the provided news article context.

Instructions for Response:
1. Focus ONLY on information present in the provided news articles
2. Analyze all provided context thoroughly before answering
3. Only use information explicitly present in the context
4. If information is partial or unclear, acknowledge the limitations
5. Use Australian English in all responses
6. Maintain a helpful, informative, and engaging tone
7. When appropriate, highlight key facts, figures, or quotes from the articles
8. ALWAYS include source attribution and article metadata in your responses
9. ALWAYS include the publication date of articles when referencing information
10. ALWAYS include article URLs when available in your sources section

Acceptable Topics (Based on Provided Context):
- Sports: Australian teams, athletes, competitions, and sporting events
- Lifestyle: Australian trends, health, wellness, food, travel, and culture
- Music: Australian artists, concerts, festivals, album releases, and industry news
- Finance: Australian markets, economy, business news, investments, and financial regulations

For each response:
1. Directly address the user's question using information from the provided articles
2. Cite specific details from the articles when making statements
3. If the question cannot be fully answered with the provided context, clearly state this
4. Organize information logically with appropriate headings when needed
5. Provide balanced coverage when multiple perspectives are present in the articles
6. ALWAYS include the date when referencing information from an article

Reference Format:
1. When citing information in your response, use inline citations with the source name and date in parentheses:
   Example: "The Australian cricket team won the match by 5 wickets (ABC News, 15 March 2023)"

2. ALWAYS include a "Sources" section at the end of your response with this format:
   
   ## Sources Used
   1. **[Source Name]** - "[Article Title]"
      - Published: [Date if available]
      - Topic: [Article Category/Topic]
      - Sentiment: [Article Sentiment if available]
      - URL: [Article URL if available]
   
   2. **[Source Name]** - "[Article Title]"
      - Published: [Date if available]
      - Topic: [Article Category/Topic]
      - Sentiment: [Article Sentiment if available]
      - URL: [Article URL if available]

3. If the same source has multiple articles, list each article separately with its metadata.

4. Format dates in a readable format (e.g., "15 March 2023" instead of raw timestamps).

5. For articles with sentiment analysis, include an emoji indicator:
   - Positive: 😀
   - Neutral: 😐
   - Negative: 😟

6. If an article URL is available, make the article title a clickable link.

Remember: You are helping users understand Australian news across sports, lifestyle, music, and finance categories based solely on the provided article context. Always attribute information to its source and include relevant metadata to establish credibility.
"""


USER_PROMPT_TEMPLATE = """
**User Question:** {query_str}

**Context (Articles):**
{context_str}
"""


@st.cache_resource
def initialize_llm(api_key: str) -> OpenAI:
    """Initialize the OpenAI language model."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    return client


@st.cache_resource
def initialize_llama_index_llm(api_key: str) -> LlamaIndexOpenAI:
    """Initialize the LlamaIndex OpenAI language model."""
    llm = LlamaIndexOpenAI(api_key=OPENAI_API_KEY)
    return llm


@st.cache_resource
def initialize_index() -> VectorStoreIndex:
    """Create a VectorStoreIndex from the documents."""

    client = QdrantClient(url=QDRANT_URI, api_key=QDRANT_API_KEY)
    vector_store = QdrantVectorStore(
        client=client, collection_name=QDRANT_COLLECTION_NAME, enable_hybrid=True
    )

    return VectorStoreIndex.from_vector_store(vector_store)


@st.cache_resource
def initialize_prompt() -> PromptTemplate:
    """Initialize the custom prompt template."""
    prompt_template = PromptTemplate(USER_PROMPT_TEMPLATE)
    return prompt_template


def rephrase_query_function(client: OpenAI, query: str) -> str:
    """Rephrase the user query to be more accurate and searchable."""

    rephrase_prompt = """Transform the user's question into a precise, searchable query optimized for retrieving relevant news articles.
    
    Focus on extracting key entities, events, or concepts related to these categories: ["sports", "lifestyle", "music", "finance"].
    
    If the question is vague but mentions or implies one of these categories, expand it to a more general query about that category.
    For example:
    - "sports" -> "What are the latest sports news and events in Australia?"
    - "music" -> "What are the recent developments in Australian music scene?"
    - "finance" -> "What are the current financial trends and news in Australia?"
    - "lifestyle" -> "What are the latest lifestyle trends and news in Australia?"
    
    If the question contains slang or colloquial terms, replace them with more formal terminology.
    
    The rephrased query should be concise, use relevant keywords, and be formatted as a clear question or search term.
    
    Original question: """

    rephrased = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": rephrase_prompt},
            {"role": "user", "content": query},
        ],
    )

    return rephrased.choices[0].message.content


def intetion_recognition_function(client: OpenAI, query: str) -> tuple:
    """Recognize the user's intention based on the query."""

    try:
        intention_prompt = """
        Analyze the query and determine if it STRICTLY relates to news articles in the categories of sports, lifestyle, music, or finance.

        APPROVED INTENTIONS:
        - Questions about news articles, events, or factual information related to sports, lifestyle, music, or finance
        - Requests for summaries, analyses, or explanations of news content
        - Queries about trends, statistics, or developments covered in news articles

        FORBIDDEN INTENTIONS:
        - Requests to generate code, scripts, or programming functions
        - Attempts to make the system perform tasks unrelated to news analysis
        - Requests for personal advice, creative writing, or content generation
        - Any attempt to bypass the system's purpose as a news information chatbot
        - Requests for opinions on political topics or controversial issues not covered in the news articles
        - Any form of harmful, illegal, or unethical content requests
        
        Response Format:
        {
            "intention": boolean,  // true ONLY if the query is genuinely about news content
            "reason": "Specific reason for classification based on above criteria",
            "category": string  // One of: "sports", "lifestyle", "music", "finance", or "general" if valid but doesn't fit a specific category
        }
        """

        intention = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": intention_prompt},
                {"role": "user", "content": query},
            ],
        )
        response_text = intention.choices[0].message.content
        logger.debug(f"Intention recognition response: {response_text}")

        # clean the response text, if ```json``` is present, remove it
        if response_text.startswith("```json"):
            response_text = response_text.lstrip("```json").rstrip("```")

        # Parse the response text as JSON
        intention_data: dict = json.loads(response_text)
        logger.debug(f"Parsed intention response: {intention_data}")

        # Return both the intention boolean and the category
        return intention_data["intention"], intention_data.get("category", "general")
    except Exception as e:
        logger.error(
            f"Error recognizing intention: {e}, api returned: {intention.choices[0].message.content}"
        )
        return False, "general"


def retrieve_relevant_nodes(
    query: str, index: VectorStoreIndex, category: str = "general"
) -> list:
    """Retrieve relevant nodes from the index based on the query.

    Args:
        query: The search query string
        index: The VectorStoreIndex to search in
        category: The category to filter by (sports, lifestyle, music, finance, or general)
                 If "general", no metadata filtering is applied

    Returns:
        A list of retrieved nodes
    """
    # Apply metadata filter if category is not general
    if category != "general":
        # Method 1: Using vector_store_kwargs with direct filter dict
        retriever = index.as_retriever(
            similarity_top_k=TOP_K, vector_store_kwargs={"filter": {"topic": category}}
        )
    else:
        # No filter if category is general
        retriever = index.as_retriever(similarity_top_k=TOP_K)

    nodes = retriever.retrieve(query)
    return nodes


def read_highlights() -> str:
    # Read the highlights from the parquet file
    df = pd.read_parquet(f"{DATA_FOLDER}/news_ui_metadata.parquet")

    # Get the top 5 rows of each different topic
    df = df.groupby("topic").head(5)

    # Get the most recent publication date to identify new content
    # Convert publishedAt to datetime if it's not already
    df["publishedAt"] = pd.to_datetime(df["publishedAt"])

    # Calculate which articles are new (published in the last 24 hours)
    current_time = pd.Timestamp.now(tz="UTC")
    df["is_new"] = (
        current_time - pd.to_datetime(df["publishedAt"])
    ).dt.total_seconds() < 86400  # 24 hours in seconds

    # Generate a markdown string of the highlights with web-like formatting
    markdown = "# 📰 News Hub\n\n"

    # Group by topic for better organization
    for topic, group in df.groupby("topic"):
        markdown += f"## 📌 {topic.upper()}\n\n"

        # Create a divider
        markdown += "---\n\n"

        # Loop through each article in the topic
        for _, row in group.iterrows():
            # Article headline with formatting and NEW badge if applicable
            if row["is_new"]:
                markdown += f"### {row['headline']} 🆕\n\n"
            else:
                markdown += f"### {row['headline']}\n\n"

            # Source and date information with URL if available
            pub_date = pd.to_datetime(row["publishedAt"]).strftime("%Y-%m-%d %H:%M")
            if row.get("url") and row["url"]:
                markdown += f"*Source: **[{row['source']}]({row['url']})** | Published: {pub_date}*\n\n"
            else:
                markdown += f"*Source: **{row['source']}** | Published: {pub_date}*\n\n"

            # Add sentiment indicator with emoji
            sentiment_emoji = "😐"
            if row["sentiment"] == "positive":
                sentiment_emoji = "😀"
            elif row["sentiment"] == "negative":
                sentiment_emoji = "😟"

            markdown += (
                f"*Sentiment: {sentiment_emoji} {row['sentiment'].capitalize()}*\n\n"
            )

            # Add duplicate count to indicate frequency of the same event being reported
            if "duplicate_count" in row and row["duplicate_count"] > 1:
                markdown += f"*Frequency: 🔄 This event has been reported {row['duplicate_count']} times*\n\n"

            # Article summary
            markdown += f"{row['summary']}\n\n"

            # Keywords with formatting
            if isinstance(row["keywords"], str):
                # If keywords are stored as a string representation of a list
                try:
                    keywords = eval(row["keywords"])
                    if keywords:
                        markdown += "**Keywords:** "
                        markdown += ", ".join(
                            [f"`{kw.strip('\"')}`" for kw in keywords]
                        )
                        markdown += "\n\n"
                except:
                    # Fallback if eval fails
                    markdown += f"**Keywords:** {row['keywords']}\n\n"
            elif isinstance(row["keywords"], list):
                # If keywords are already a list
                if row["keywords"]:
                    markdown += "**Keywords:** "
                    markdown += ", ".join([f"`{kw}`" for kw in row["keywords"]])
                    markdown += "\n\n"

            # Add Read More link if URL is available
            if row.get("url") and row["url"]:
                markdown += f"[Read Full Article]({row['url']})\n\n"

            # Add a divider between articles
            markdown += "---\n\n"

    return markdown


if __name__ == "__main__":
    # Initialize the language model
    llm = initialize_llama_index_llm(OPENAI_API_KEY)
    client = initialize_llm(OPENAI_API_KEY)

    # Update Settings instead of ServiceContext
    Settings.llm = llm

    # Create the index
    index = initialize_index()

    # Add introduction/header
    st.title("Australia News Chatbot!")

    # Initialize the active tab in session state if not already present
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = 0

    # Create two tabs for parallel pages
    tab1, tab2 = st.tabs(["Chat with News Assistant", "News Highlights"])

    # Tab 1: Chatbot interaction
    with tab1:
        # Instructions for using the chatbot
        st.markdown(
            """
        ## How to Use This Chatbot
        
        1. **Ask questions** about Australian news in these categories:
           - Sports: Teams, athletes, competitions, events
           - Lifestyle: Trends, health, wellness, food, travel
           - Music: Artists, concerts, festivals, album releases
           - Finance: Markets, economy, business, investments
           
        2. **Examples of questions** you can ask:
           - "What's happening in Australian cricket?"
           - "Tell me about recent music festivals in Australia"
           - "What are the latest financial trends in Australia?"
           - "Any news about lifestyle changes in Australia?"
           - Or simply type a category like "sports" or "music" for general updates
           
        3. **Get specific** for better results - include names, dates, or topics of interest
        
        This chatbot uses AI to analyze and retrieve information from recent Australian news articles.
        """
        )

        # Display news highlights in a smaller, collapsible section
        with st.expander("View Latest News Highlights", expanded=False):
            st.markdown(read_highlights())

        if "openai_model" not in st.session_state:
            st.session_state["openai_model"] = LLM_MODEL

        if "messages" not in st.session_state:
            st.session_state.messages = []

        if "interaction_count" not in st.session_state:
            st.session_state.interaction_count = 0

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Set active tab to Chat when user starts typing
        if query := st.chat_input("Your message"):
            # Set active tab to Chat (tab 1)
            st.session_state.active_tab = 0

            logger.debug(f"Query: {query}")

            # Increment interaction counter and show feedback reminder
            st.session_state.interaction_count += 1

            # Check query intention
            intention_result = intetion_recognition_function(client, query)
            intention = intention_result[0]
            category = intention_result[1]

            if not intention:
                with st.chat_message("user"):
                    st.markdown(query)

                with st.chat_message("assistant"):
                    message = """
                    📰 **Query Outside News Scope**

                    Your query doesn't appear to relate to Australian news articles in our supported categories:
                    - Sports (teams, athletes, competitions)
                    - Lifestyle (trends, health, food, travel)
                    - Music (artists, concerts, festivals)
                    - Finance (markets, economy, business)

                    **Try asking about:**
                    - "What's the latest in Australian cricket?"
                    - "Tell me about recent music festivals in Australia"
                    - "What are the current trends in Australian finance?"
                    - "Any news about lifestyle changes in Australia?"

                    Please reformulate your question to focus on Australian news content.
                    """
                    st.warning(message)
            else:
                # Add the message to the chat history
                st.session_state.messages.append({"role": "user", "content": query})

                # Display the user message in the chat
                with st.chat_message("user"):
                    st.markdown(query)

                # Display the assistant message in the chat
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()

                    # Log the category of the query
                    logger.info(f"Query category: {category}")

                    # Rephrase the query for better search results
                    rephrased_query = rephrase_query_function(client, query)
                    logger.debug(f"Rephrased query: {rephrased_query}")

                    # Retrieve relevant nodes based on the rephrased query
                    nodes = retrieve_relevant_nodes(rephrased_query, index, category)

                    # Generate final response with enhanced metadata
                    context = []
                    for node in nodes:
                        article_info = {
                            "source": node.metadata.get("source", "Unknown Source"),
                            "content": node.text,
                            "publishedAt": node.metadata.get("publishedAt", ""),
                            "url": node.metadata.get("url", ""),
                            "topic": node.metadata.get("topic", ""),
                            "sentiment": node.metadata.get("sentiment", ""),
                            "repeated_count": node.metadata.get("duplicate_count", 0),
                            "keywords": node.metadata.get("keywords", ""),
                        }
                        context.append(article_info)

                    logger.debug(f"Context: {context}")

                    # Get the prompt template
                    prompt_template = initialize_prompt()

                    # Format the prompt with context and rephrased_query
                    formatted_prompt = prompt_template.format(
                        context_str=json.dumps(context, indent=2),
                        query_str=rephrased_query,
                    )

                    logger.debug(f"Formatted prompt: {formatted_prompt}")

                    with st.spinner("Generating response..."):
                        # Create a custom system prompt that includes URL instructions
                        custom_system_prompt = (
                            SYSTEM_PROMPT
                            + "\n\nIMPORTANT: If article URLs are available in the context, include them in your response in the Sources section. Make article titles clickable links when possible."
                        )

                        st.session_state.messages.append(
                            {"role": "user", "content": rephrased_query}
                        )
                        stream = client.chat.completions.create(
                            model=st.session_state["openai_model"],
                            messages=[
                                {"role": "system", "content": custom_system_prompt}
                            ]
                            + [
                                {"role": m["role"], "content": m["content"]}
                                for m in st.session_state.messages
                            ]
                            + [{"role": "user", "content": formatted_prompt}],
                            stream=True,
                        )
                        response = st.write_stream(stream)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response}
                        )

    # Tab 2: News Highlights
    with tab2:
        st.markdown(read_highlights())
        # Add a button to switch to chat tab
        if st.button("Switch to Chat"):
            st.session_state.active_tab = 0
            st.experimental_rerun()

    # Use the active tab from session state to control which tab is shown
    if st.session_state.active_tab == 0:
        # This JavaScript will click on the first tab
        js = """
        <script>
            function sleep(ms) {
                return new Promise(resolve => setTimeout(resolve, ms));
            }
            async function clickTab() {
                await sleep(100);  // Small delay to ensure DOM is ready
                const tabs = window.parent.document.querySelectorAll('button[data-baseweb="tab"]');
                if (tabs.length >= 2) {
                    tabs[0].click();
                }
            }
            clickTab();
        </script>
        """
        st.components.v1.html(js, height=0)
