# Australian News Aggregator with AI-Powered Chatbot

## Project Overview

This project involves developing an AI-powered system that:
1. Extracts news articles from multiple Australian news outlets
2. Categorizes and summarizes the content
3. Presents daily highlights in a user-friendly interface
4. Provides a chatbot for users to query the news using Retrieval-Augmented Generation (RAG)

## Technical Requirements

### 1. News Extraction & Data Processing
- Extract news from multiple Australian news outlets
- Focus on four main categories: Sports, Lifestyle, Music, and Finance
- Implement an optimized pipeline for efficient data collection

### 2. Content Classification & Deduplication
- **Categorization**: Accurately classify articles into the four main categories
- **Deduplication**: Identify and group similar stories reported across multiple sources
  - Implement clustering, indexing, or other appropriate techniques
  - Avoid presenting duplicate content to users

### 3. Highlight Generation
- Create a system to identify important news highlights
- Prioritization criteria:
  - Important keywords (e.g., "Breaking News")
  - Frequency of coverage across sources
  - Relevance within each category

### 4. User Interface Development
- Build a web-based dashboard for presenting daily news highlights
- Organize content by categories
- For each article, display:
  - Title
  - Author
  - Source(s)
  - Frequency/popularity metrics
  - Other relevant metadata

### 5. Chatbot Implementation
- Create an interactive chatbot interface
- Allow users to ask questions about daily news highlights
- Implement Retrieval-Augmented Generation (RAG) to:
  - Understand user queries in context
  - Retrieve relevant news information
  - Generate accurate, contextual responses

## Deliverables
- Functional news aggregation system
- Web-based UI for browsing highlights
- Interactive chatbot with RAG capabilities
- Documentation of system architecture and components
