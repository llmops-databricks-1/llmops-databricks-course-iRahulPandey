# Databricks notebook source
"""
Week 1 — Book Summaries Ingestion (incremental sync)

Scans the Unity Catalog volume for all .md / .txt summary files, diffs against
what is already in `raw_documents`, and ingests only new files. Re-run the job
any time a new summary lands in the volume — existing records are never touched.

Input  : /Volumes/<catalog>/book_recommender/summaries/*.md  *.txt
Output : <catalog>.book_recommender.raw_documents (Delta, append)
"""

# COMMAND ----------

import re
import uuid
from datetime import datetime
from pathlib import Path

from loguru import logger

from book_recommender.config import load_config

# COMMAND ----------

config = load_config()
logger.info(f"Environment : {config.catalog}")
logger.info(f"Volume      : {config.full_volume_path}")
logger.info(f"Target table: {config.full_schema_name}.raw_documents")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def list_volume_files(volume_path: str) -> list[Path]:
    """Return all .md and .txt files currently in the volume."""
    root = Path(volume_path)
    files = sorted(root.glob("*.md")) + sorted(root.glob("*.txt"))
    logger.info(f"Found {len(files)} file(s) in volume")
    return files


def get_ingested_filenames(spark: "SparkSession", table: str) -> set[str]:  # noqa: F821
    """Return the set of file_name values already in raw_documents."""
    try:
        existing = spark.table(table).select("file_name").distinct()
        names = {row["file_name"] for row in existing.collect()}
        logger.info(f"{len(names)} file(s) already in {table}")
        return names
    except Exception:
        logger.info(f"{table} does not exist yet — full load")
        return set()


def parse_filename_metadata(filename: str) -> dict:
    """
    Extract title and reading date from the file stem.

    Convention: YYYY-MM-DD_kebab-title.md
      2018-04-01_godel-escher-bach.md  →  title="Godel Escher Bach",
                                          reading_date="2018-04-01"
      sapiens.md (no date)             →  title="Sapiens",            reading_date=None
    """
    stem = Path(filename).stem
    match = re.match(r"^(\d{4}-\d{2}-\d{2})_(.+)$", stem)
    if match:
        reading_date, title_slug = match.group(1), match.group(2)
    else:
        reading_date, title_slug = None, stem
    title = title_slug.replace("-", " ").replace("_", " ").title()
    return {"file_stem": stem, "title": title, "reading_date": reading_date}


def read_file_local(path: Path) -> str:
    """Read a file from the FUSE mount (Databricks cluster)."""
    return path.read_text(encoding="utf-8")


def read_file_sdk(path: str, workspace_client: "WorkspaceClient") -> str:  # noqa: F821
    """Fallback: read via Files API (local dev with databricks-connect)."""
    with workspace_client.files.download(path).contents as f:
        return f.read().decode("utf-8")


# COMMAND ----------

# ---------------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------------

import pandas as pd  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402

spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {config.full_schema_name}")

target_table = f"{config.full_schema_name}.raw_documents"

try:
    from databricks.sdk import WorkspaceClient  # noqa: PLC0415

    w = WorkspaceClient()
except Exception:
    w = None

# 1. What's in the volume?
volume_files = list_volume_files(config.full_volume_path)

# 2. What's already ingested?
ingested = get_ingested_filenames(spark, target_table)

# 3. Only process new files
new_files = [f for f in volume_files if f.name not in ingested]
logger.info(f"{len(new_files)} new file(s) to ingest")

if not new_files:
    logger.info("Nothing to do — raw_documents is already up to date")
else:
    records = []
    for file_path in new_files:
        meta = parse_filename_metadata(file_path.name)

        try:
            text = read_file_local(file_path)
            logger.debug(f"Read {file_path.name} via FUSE")
        except Exception:
            if w is None:
                raise
            text = read_file_sdk(str(file_path), w)
            logger.debug(f"Read {file_path.name} via Files API")

        records.append(
            {
                "document_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, meta["file_stem"])),
                "title": meta["title"],
                "reading_date": meta["reading_date"],
                "file_stem": meta["file_stem"],
                "file_name": file_path.name,
                "content": text,
                "char_count": len(text),
                "word_count": len(re.findall(r"\w+", text)),
                "source": "austin_tripp_summaries",
                "catalog": config.catalog,
                "schema_name": config.schema_name,
                "ingested_at": datetime.utcnow().isoformat(),
                "status": "raw",
            }
        )

    df = spark.createDataFrame(pd.DataFrame(records))
    df.write.format("delta").mode("append").saveAsTable(target_table)
    logger.info(f"Appended {len(records)} new record(s) to {target_table}")

# COMMAND ----------

total = spark.table(target_table).count()
logger.info(f"Total records in raw_documents: {total}")

from pyspark.sql import functions as F  # noqa: E402, N812

display(  # noqa: F821
    spark.table(target_table)
    .withColumn("content_preview", F.substring(F.col("content"), 1, 300))
    .select(
        "document_id",
        "title",
        "reading_date",
        "char_count",
        "word_count",
        "content_preview",
        "ingested_at",
        "status",
    )
    .orderBy("reading_date")
)
