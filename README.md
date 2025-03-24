

##  News Chatbot Pipeline & UI Documentation

### Overview

The News Chatbot is a comprehensive system that processes news articles using OpenAI's API to generate categories, summaries, and embeddings. It uses ZenML for pipeline management and DuckDB for data storage. The system provides a chat interface for users to query and interact with the processed news data.

### Pipeline Architecture

The pipeline consists of several key steps:

1. **Data Acquisition (Step 1)**
   - Fetches news articles from NewsAPI across multiple topics (sports, lifestyle, music, finance)
   - Stores raw data in DuckDB database
   - Implements pagination to handle API limits

2. **Data Cleaning (Step 2.1)**
   - Processes raw news data
   - Standardizes date formats
   - Removes duplicates
   - Assigns unique IDs to articles

3. **Content Enrichment (Step 2.1)**
   - Uses OpenAI API to generate:
     - Categories (sports, lifestyle, music, finance)
     - Detailed summaries
     - Keywords
     - Key entities
     - Sentiment analysis
     - Alternative headlines
   - Processes articles in batches to manage API rate limits

4. **Embedding Generation (Step 2.1)**
   - Creates vector embeddings for article titles using OpenAI's embedding API
   - Enables semantic search capabilities
   - Processes in batches with rate limiting

5. **News Importance Ranking (Step 3)**
   - Implements Reciprocal Rank Fusion (RRF) to score article importance
   - Combines frequency metrics (duplicate count) with keyword relevance
   - Uses SentenceTransformer to compute similarity between articles and topic keywords
   - Creates separate ranked datasets for each news category
   - Stores results in category-specific tables in DuckDB

6. **UI Metadata Generation (Step 4)**
   - Curates and formats article data for UI presentation
   - Selects top-ranked articles (based on RRF score) for each category
   - Normalizes source and author information
   - Generates two output datasets:
     - A focused dataset with top 15 articles per category for UI display
     - A complete dataset with all processed articles for comprehensive access
   - Exports data to Parquet files for efficient storage and retrieval

7. **Vector Index Building (Step 5)**
   - Creates searchable vector index using LlamaIndex and Qdrant
   - Implements hybrid search capabilities (dense and sparse vectors)
   - Processes documents with:
     - Sentence splitting for chunking content
     - Title extraction for improved search relevance
     - OpenAI embeddings for semantic understanding
   - Stores article metadata alongside vector representations
   - Configurable parameters for chunk size, overlap, and processing parallelism

### Chat UI Implementation

The chat interface provides a user-friendly way to interact with the news database:

1. **Core Components**
   - Built with Streamlit for the web interface
   - Uses LlamaIndex for vector search capabilities
   - Integrates with Qdrant vector database for efficient similarity search
   - Leverages OpenAI's GPT models for natural language understanding

2. **Key Features**
   - Query rephrasing to optimize search results
   - Intention recognition to ensure queries are relevant to news content
   - Category-based filtering (sports, lifestyle, music, finance)
   - Source attribution in responses
   - Sentiment indicators for articles (positive, neutral, negative)

3. **User Experience**
   - Provides detailed, contextual responses based on relevant news articles
   - Includes metadata and publication dates for transparency
   - Formats responses with clear structure and source attribution
   - Supports general queries about news categories or specific information requests

### Data Storage

- **DuckDB**: Used as the primary database for structured news data
- **Qdrant**: Vector database for storing and retrieving embeddings
- **Data Model**: Articles are stored with rich metadata including categories, summaries, keywords, and vector embeddings

### System Requirements

- OpenAI API key for content processing and chat functionality
- NewsAPI key for data acquisition
- Qdrant instance for vector storage
- Python environment with required dependencies

### Future Enhancements

- Implement deduplication across categories
- Generate highlights per category based on frequency and keywords
- Enhance UI to show categorized news with key details
- Improve chatbot response quality and context awareness