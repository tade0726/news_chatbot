##  News Chatbot Pipeline & UI Documentation

### Overview

The News Chatbot is a comprehensive system that processes news articles using OpenAI's API to generate categories, summaries, and embeddings. It uses ZenML for pipeline management and DuckDB for data storage. The system provides a chat interface for users to query and interact with the processed news data.

### Technology Stack Selection

The project leverages several key technologies chosen for their specific strengths:

1. **ZenML**
   - Selected for creating reproducible data science pipelines
   - Provides versioning for models and pipeline steps
   - Enables tracking of pipeline runs and metadata
   - Facilitates easy experimentation with different pipeline configurations
   - Supports caching of intermediate results for faster iteration
   - Ensures reproducibility across different environments

2. **DuckDB**
   - Chosen for its simplicity and portability
   - Stores data as a single file that can be easily shared within the repository
   - Provides SQL capabilities with minimal setup
   - Offers excellent performance for analytical queries
   - Requires no separate database server installation
   - Supports direct integration with pandas DataFrames

3. **OpenAI API**
   - Powers the natural language understanding components
   - Generates high-quality summaries and categorizations
   - Creates embeddings for semantic search capabilities

4. **Qdrant**
   - Vector database optimized for similarity search
   - Supports hybrid search combining dense and sparse vectors
   - Enables efficient retrieval of relevant news articles

5. **Streamlit**
   - Provides a simple yet powerful framework for building the UI
   - Enables rapid development of interactive web applications
   - Integrates well with Python data science tools

### Environment Setup

Before running the project, you need to set up the required environment variables:

1. **Create Environment File**:
   - Copy the provided `.env.sample` file to a new file named `.env`
   - Update the values with your own API keys and configuration

   ```bash
   cp .env.sample .env
   ```

2. **Required API Keys**:
   - **OpenAI API Key**: Required for content processing and embeddings
   - **NewsAPI Key**: Required for fetching news articles
   - **Tavily API Key**: Used for additional search capabilities (optional)
   - **Qdrant API Key**: If using a secured Qdrant instance

3. **File Paths**:
   - Update the file paths in the `.env` file to match your local setup:
     - `DUCKDB_PATH`: Path to the DuckDB database file
     - `TMP_FOLDER`: Directory for temporary files
     - `DATA_FOLDER`: Directory for storing processed data

4. **Service URLs**:
   - Ensure the ZenML and Qdrant URLs match your Docker Compose configuration
   - Default values should work if using the provided docker-compose.yml

Example `.env` file structure:
```
# API Keys
TAVILY_API_KEY="your_tavily_api_key_here"
OPENAI_API_KEY="your_openai_api_key_here"
NEWSAPI_KEY="your_newsapi_key_here"

# File paths
DUCKDB_PATH="./duckdb/duckdb.db"
TMP_FOLDER="./tmp"
DATA_FOLDER="./src/news_chatbot/data"

# ZenML configuration
OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
ZENML_SERVER_URL=http://127.0.0.1:8080

# Qdrant configuration
QDRANT_URI=http://127.0.0.1:6333
QDRANT_COLLECTION_NAME="build-index-pipeline-v1"
QDRANT_API_KEY="your_qdrant_api_key_here"
```

### Project Structure and Development Environment

The project follows a modular structure for maintainability and scalability:

```
news_chatbot/
├── .env                  # Environment variables configuration
├── .env.sample           # Template for environment variables
├── .streamlit/           # Streamlit configuration
├── Makefile              # Commands for running pipelines and services
├── README.md             # Project documentation
├── docker-compose.yml    # Local development environment setup
├── docs/                 # Additional documentation
├── duckdb/               # DuckDB database files
├── pyproject.toml        # Project dependencies and metadata
├── src/                  # Source code
│   └── news_chatbot/     # Main package
│       ├── __init__.py
│       ├── chats_ui.py   # Streamlit chat interface
│       ├── data/         # Processed data storage
│       ├── datasets/     # Data handling modules
│       ├── llm/          # LLM integration components
│       └── pipelines/    # ZenML pipeline implementations
│           ├── step1.py  # Data acquisition
│           ├── step2_1.py # Data cleaning and enrichment
│           ├── step2_2.py # Additional processing
│           ├── step3.py  # News importance ranking
│           ├── step4.py  # UI metadata generation
│           └── step5.py  # Vector index building
└── tmp/                  # Temporary files
```

#### Development Environment Setup

The project uses Docker Compose to create a consistent local development environment:

1. **Services Included**:
   - **MySQL**: Database for ZenML metadata storage
   - **ZenML Server**: Pipeline orchestration and tracking
   - **Qdrant**: Vector database for similarity search

2. **Getting Started**:
   ```bash
   # Start all services
   make up
   
   # Stop services
   make stop
   
   # Remove containers
   make down
   ```

#### Pipeline Execution

The Makefile provides commands to run each pipeline step in the correct order:

```bash
# Run data acquisition pipeline
make step1

# Run data cleaning and enrichment pipeline
make step2_1

# Run additional processing
make step2_2

# Run news importance ranking
make step3

# Run UI metadata generation
make step4

# Run vector index building
make step5

# Start the Streamlit UI
make streamlit
```

The pipelines should be run in sequence as each step depends on the output of the previous steps. The Makefile simplifies this process by providing standardized commands for each operation.

### Pipeline Architecture

The pipeline consists of several key steps:

1. **Data Acquisition (Step 1)**
   - Fetches news articles from NewsAPI across multiple topics (sports, lifestyle, music, finance)
   - Stores raw data in DuckDB database
   - Implements pagination to handle API limits

2. **Content Enrichment (Step 2.1)**
   - Uses OpenAI API to generate:
     - Categories (sports, lifestyle, music, finance)
     - Detailed summaries
     - Keywords
     - Key entities
     - Sentiment analysis
     - Alternative headlines
   - Processes requests in async to improve performance

3. **Semantic Deduplication (Step 2.2)**
   - Implements k-nearest neighbors (KNN) algorithm using FAISS (Facebook AI Similarity Search) for efficient similarity search
   - Utilizes title embeddings to identify semantically similar news articles
   - Process:
     - Builds a FAISS index with title embeddings (vector representations of titles)
     - Performs KNN search to find the top 5 most similar titles for each article
     - Applies a similarity threshold (0.9) to identify near-duplicates
     - Keeps only one representative article from each cluster of similar articles
     - Tracks the number of duplicates for each unique article as `duplicate_count`
     - This count represents the frequency of reporting for the same event across sources
   - Benefits:
     - Identifies semantically equivalent titles even with different wording
     - Reduces redundancy in the news feed while preserving diversity
     - Provides frequency metrics that indicate the importance/popularity of news events
     - Maintains the most representative version of each news story

4. **News Importance Ranking (Step 3)**
   - Implements Reciprocal Rank Fusion (RRF) to score article importance
   - Combines frequency metrics (duplicate count) with keyword relevance
   - Uses SentenceTransformer to compute similarity between articles and topic keywords
   - Creates separate ranked datasets for each news category
   - Stores results in category-specific tables in DuckDB
   
   **How Importance Ranking Works:**
   - Each news category (sports, lifestyle, music, finance) has a set of predefined keywords that represent what's important in that domain (e.g., "athlete", "tournament", "score" for sports)
   - Keywords for each category are combined into a single text string (e.g., "athlete tournament score breaking news")
   - Both the combined article content (headline + summary) and the category keyword strings are converted into vector embeddings using SentenceTransformer
   - Cosine similarity is calculated between each article's embedding and each category's keyword embedding
   - Articles receive a similarity score for each category, measuring how well they align with the importance criteria
   - The final importance ranking combines two factors using Reciprocal Rank Fusion (RRF):
     1. Frequency rank: How often the article appears across sources (duplicate count)
     2. Semantic relevance: How closely the article matches the category's importance keywords
   - This approach balances popularity metrics with content relevance to identify truly important news

5. **UI Metadata Generation (Step 4)**
   - Curates and formats article data for UI presentation
   - Selects top-ranked articles (based on RRF score) for each category
   - Normalizes source and author information
   - Generates two output datasets:
     - A focused dataset with top 15 articles per category for UI display
     - A complete dataset with all processed articles for comprehensive access
   - Exports data to Parquet files for efficient storage and retrieval

6. **Vector Index Building (Step 5)**
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