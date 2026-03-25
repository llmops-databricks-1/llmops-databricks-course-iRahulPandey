# Databricks notebook source
"""
Week 2 — Chunk book summaries and sync to Mosaic AI Vector Search

Pipeline:
  raw_documents (Delta)
    └─► chunk_text()  →  chunked_documents (Delta, CDC enabled)
         └─► VS Delta Sync index  (embeddings computed by Databricks
                                   using databricks-gte-large-en)

Run after 1_book_summaries_ingestion.py.
Only un-chunked documents are processed on each run — safe to re-run.
"""

# COMMAND ----------

import time
import uuid
from datetime import datetime

from loguru import logger

from book_recommender.config import load_config

# COMMAND ----------

config = load_config()
logger.info(f"Environment       : {config.catalog}")
logger.info(f"Source table      : {config.full_schema_name}.raw_documents")
logger.info(f"Target table      : {config.full_schema_name}.chunked_documents")
logger.info(f"VS endpoint       : {config.vector_search_endpoint}")
logger.info(f"VS index          : {config.full_vs_index_name}")
logger.info(f"Chunk size        : {config.chunk_size} chars")
logger.info(f"Chunk overlap     : {config.chunk_overlap} chars")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------


def split_into_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks, preferring natural paragraph breaks.

    Tries double-newline → newline → sentence → word → hard cut.
    Ensures every chunk is at most `chunk_size` characters.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            for sep in ("\n\n", "\n", ". ", " "):
                idx = text.rfind(sep, start + overlap, end)
                if idx != -1:
                    end = idx + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap if end < len(text) else len(text)

    return chunks


def build_chunk_records(
    document_id: str,
    title: str,
    reading_date: str | None,
    content: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict]:
    """Return a list of chunk records ready for insertion into chunked_documents."""
    chunks = split_into_chunks(content, chunk_size, chunk_overlap)
    now = datetime.utcnow().isoformat()
    records = []
    for idx, chunk_text in enumerate(chunks):
        chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{document_id}::{idx}"))
        records.append(
            {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "title": title,
                "reading_date": reading_date,
                "chunk_index": idx,
                "chunk_text": chunk_text,
                "chunk_char_count": len(chunk_text),
                "chunk_word_count": len(chunk_text.split()),
                "chunked_at": now,
            }
        )
    return records


# COMMAND ----------

# ---------------------------------------------------------------------------
# Incremental chunk: only process documents not yet in chunked_documents
# ---------------------------------------------------------------------------

import pandas as pd  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402, N812

spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()

source_table = f"{config.full_schema_name}.raw_documents"
target_table = f"{config.full_schema_name}.chunked_documents"

# Determine which document_ids are already chunked
try:
    chunked_ids = {
        row["document_id"]
        for row in spark.table(target_table).select("document_id").distinct().collect()
    }
    logger.info(f"{len(chunked_ids)} document(s) already chunked")
except Exception:
    chunked_ids = set()
    logger.info("chunked_documents does not exist yet — full load")

# Load only new documents
raw_df = spark.table(source_table)
new_docs = [
    row.asDict() for row in raw_df.collect() if row["document_id"] not in chunked_ids
]
logger.info(f"{len(new_docs)} new document(s) to chunk")

if not new_docs:
    logger.info("Nothing to do — chunked_documents is already up to date")
else:
    all_chunks: list[dict] = []
    for doc in new_docs:
        chunks = build_chunk_records(
            document_id=doc["document_id"],
            title=doc["title"],
            reading_date=doc.get("reading_date"),
            content=doc["content"],
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        all_chunks.extend(chunks)
        logger.info(f"  {doc['title']}: {len(chunks)} chunk(s)")

    chunks_df = spark.createDataFrame(pd.DataFrame(all_chunks))
    (
        chunks_df.write.format("delta")
        .mode("append")
        .option("delta.enableChangeDataFeed", "true")
        .saveAsTable(target_table)
    )

    # Ensure CDC is enabled (idempotent — safe to run on existing table too)
    spark.sql(
        f"ALTER TABLE {target_table} "
        f"SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')"
    )
    logger.info(f"Wrote {len(all_chunks)} chunk(s) to {target_table}")

# COMMAND ----------

total_chunks = spark.table(target_table).count()
total_docs = spark.table(target_table).select("document_id").distinct().count()
logger.info(f"chunked_documents: {total_chunks} chunks across {total_docs} document(s)")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Vector Search — endpoint setup
# ---------------------------------------------------------------------------

from databricks.sdk import WorkspaceClient  # noqa: E402
from databricks.sdk.service.vectorsearch import (  # noqa: E402
    EndpointStatusState,
    EndpointType,
)

w = WorkspaceClient()


def get_or_create_vs_endpoint(client: WorkspaceClient, endpoint_name: str) -> None:
    """Create the VS endpoint if it does not exist, then wait until ONLINE."""
    existing = {ep.name for ep in client.vector_search_endpoints.list_endpoints()}

    if endpoint_name not in existing:
        logger.info(f"Creating VS endpoint '{endpoint_name}' ...")
        client.vector_search_endpoints.create_endpoint_and_wait(
            name=endpoint_name,
            endpoint_type=EndpointType.STANDARD,
        )
        logger.info(f"VS endpoint '{endpoint_name}' created")
    else:
        logger.info(f"VS endpoint '{endpoint_name}' already exists — checking status")

    # Wait until ONLINE (may already be there)
    _wait_for_endpoint_online(client, endpoint_name)


def _wait_for_endpoint_online(
    client: WorkspaceClient,
    endpoint_name: str,
    poll_seconds: int = 15,
    timeout_seconds: int = 600,
) -> None:
    """Poll the endpoint until it reaches ONLINE state."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        ep = client.vector_search_endpoints.get_endpoint(endpoint_name)
        state = ep.endpoint_status.state if ep.endpoint_status else None
        logger.info(f"VS endpoint state: {state}")
        if state == EndpointStatusState.ONLINE:
            return
        time.sleep(poll_seconds)
    raise TimeoutError(
        f"VS endpoint '{endpoint_name}' did not become ONLINE within {timeout_seconds}s."
    )


get_or_create_vs_endpoint(w, config.vector_search_endpoint)

# COMMAND ----------

# ---------------------------------------------------------------------------
# Vector Search — index setup
# ---------------------------------------------------------------------------

from databricks.sdk.service.vectorsearch import (  # noqa: E402
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingSourceColumn,
    PipelineType,
    VectorIndexType,
)


def get_or_create_vs_index(client: WorkspaceClient, cfg: "ProjectConfig") -> None:  # noqa: F821
    """Create a Delta Sync VS index if it does not exist, then trigger a sync."""
    existing_names = {
        idx.name
        for idx in client.vector_search_indexes.list_indexes(
            endpoint_name=cfg.vector_search_endpoint
        )
    }

    if cfg.full_vs_index_name not in existing_names:
        logger.info(f"Creating VS index '{cfg.full_vs_index_name}' ...")
        client.vector_search_indexes.create_index(
            name=cfg.full_vs_index_name,
            endpoint_name=cfg.vector_search_endpoint,
            primary_key="chunk_id",
            index_type=VectorIndexType.DELTA_SYNC,
            delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
                source_table=f"{cfg.full_schema_name}.chunked_documents",
                pipeline_type=PipelineType.TRIGGERED,
                embedding_source_columns=[
                    EmbeddingSourceColumn(
                        name="chunk_text",
                        embedding_model_endpoint_name=cfg.embedding_endpoint,
                    )
                ],
            ),
        )
        logger.info(f"VS index '{cfg.full_vs_index_name}' created")
    else:
        logger.info(f"VS index '{cfg.full_vs_index_name}' exists — triggering sync")
        client.vector_search_indexes.sync_index(cfg.full_vs_index_name)
        logger.info("Sync triggered")


get_or_create_vs_index(w, config)

# COMMAND ----------

# ---------------------------------------------------------------------------
# Summary display
# ---------------------------------------------------------------------------

display(  # noqa: F821
    spark.table(target_table)
    .groupBy("document_id", "title", "reading_date")
    .agg(
        F.count("chunk_id").alias("num_chunks"),
        F.sum("chunk_char_count").alias("total_chars"),
        F.min("chunk_index").alias("first_chunk"),
        F.max("chunk_index").alias("last_chunk"),
    )
    .orderBy("reading_date")
)
