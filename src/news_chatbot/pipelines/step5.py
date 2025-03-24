# Standard libraries
import os
from datetime import datetime

# Third-party libraries
import pandas as pd
from zenml import Model, get_step_context, log_step_metadata, pipeline, step


# Llama Index
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models


# Model configuration
MODEL_NAME = "build-index-pipeline"
MODEL_VERSION = None


# data folder
DATA_FOLDER = os.getenv("DATA_FOLDER")


## INDEX parameters
QDRANT_URI = os.getenv("QDRANT_URI")
NUM_DOCS = -1
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
NUM_WORKERS = 10


@step
def load_data(batch_query_time: int) -> pd.DataFrame:

    # load all news data
    df = pd.read_parquet(f"{DATA_FOLDER}/news_metadata.parquet")
    return df


@step
def build_index(
    df: pd.DataFrame,
    chunk_size: int,
    chunk_overlap: int,
    num_workers: int,
    num_docs: int,
):
    """build the index and store the index Qdrant, expose chunk_size, chunk_overlap, and num_docs"""

    # limit for test
    if num_docs > 0:
        df = df.head(num_docs)

    # access the model name from the pipeline step
    model_name = get_step_context().model.name

    # version of model
    model_version = get_step_context().model.version

    # using model name and version to create a collection name
    collection_name = f"{model_name}-{model_version}"

    # client
    client = QdrantClient(url=QDRANT_URI)

    # create two sets of vectors fro hybrid search
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "text-dense": models.VectorParams(
                size=1536,  # openai vector size
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "text-sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )

    vector_store = QdrantVectorStore(
        client=client, collection_name=collection_name, enable_hybrid=True
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # create the pipeline with transformations
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
            TitleExtractor(),
            OpenAIEmbedding(),
        ]
    )

    # creating documents
    documents = [
        Document(
            text=f"{row['headline']}\n\n{row['summary']}",
            metadata={
                "source": row["source"],
                "keywords": row["keywords"],
                "duplicate_count": row["duplicate_count"],
                "rrf_rank": row["rrf_rank"],
                "sentiment": row["sentiment"],
                "topic": row["topic"],
            },
        )
        for _, row in df.iterrows()
    ]

    # build index from nodes
    nodes = pipeline.run(documents=documents, num_workers=num_workers)

    # add nodes to vector store
    _ = VectorStoreIndex(nodes=nodes, storage_context=storage_context)

    # metadata
    log_step_metadata(
        metadata={
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "num_docs": df.shape[0],
            "qdrant_collection_name": collection_name,
        },
    )


# Create model metadata for tracking
model = Model(
    name=MODEL_NAME,
    version=MODEL_VERSION,
    description="Generate metadata for news articles",
    tags=["news", "data", "metadata"],
)


@pipeline
def build_index_pipeline(batch_query_time: int):

    df = load_data(batch_query_time)

    build_index(
        df,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        num_workers=NUM_WORKERS,
        num_docs=NUM_DOCS,
    )


if __name__ == "__main__":
    # Execute the pipeline with current timestamp
    _ = build_index_pipeline.with_options(model=model)(
        batch_query_time=int(datetime.now().timestamp())
    )
