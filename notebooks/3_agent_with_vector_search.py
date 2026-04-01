# Databricks notebook source
"""
Week 3 — Book Recommender Agent with RAG

Production-grade RAG agent that:
1. Takes user query about books/themes
2. Searches vector index for top 3 matching summaries
3. Passes retrieved context to LLM
4. Generates thoughtful book recommendations with reasons
5. Visualizes the LangGraph execution

Architecture:
- LangGraph state machine for orchestration
- Vector Search retrieval step (RETRIEVER span)
- LLM generation with retrieved context (LLM span)
- Structured JSON output with book recommendations
- Full MLflow tracing for observability
"""

# COMMAND ----------

from loguru import logger

from book_recommender.config import load_config

config = load_config()
logger.info(f"Environment: {config.catalog}")
logger.info(f"VS Index   : {config.full_vs_index_name}")
logger.info(f"LLM        : {config.llm_endpoint}")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import json
from typing import Any

import mlflow
from databricks.sdk import WorkspaceClient
from databricks_langchain import ChatDatabricks
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, END, StateGraph

# COMMAND ----------

# ---------------------------------------------------------------------------
# State Definition
# ---------------------------------------------------------------------------

from typing_extensions import TypedDict


class AgentState(TypedDict):
    """State passed through the agent workflow."""

    user_query: str
    retrieved_summaries: list[dict[str, Any]]
    llm_response: str
    recommendations: list[dict[str, Any]]


# COMMAND ----------

# ---------------------------------------------------------------------------
# Retrieval Step
# ---------------------------------------------------------------------------

w = WorkspaceClient()


@mlflow.trace(name="retrieve_summaries", span_type=mlflow.entities.SpanType.RETRIEVER)
def retrieve_summaries(query: str, num_results: int = 3) -> list[dict[str, Any]]:
    """
    Retrieve top matching summaries from Vector Search.

    Args:
        query: User's natural language query
        num_results: Number of results to retrieve

    Returns:
        List of dicts with book title, date, and excerpt
    """
    logger.info(f"Retrieving top {num_results} summaries for: {query}")

    try:
        results = w.vector_search_indexes.query_index(
            index_name=config.full_vs_index_name,
            query_text=query,
            columns=["chunk_id", "title", "reading_date", "chunk_index", "chunk_text"],
            num_results=num_results,
        )

        if not results.result.data_array:
            logger.warning("No results found")
            return []

        col_names = [c.name for c in results.manifest.columns]
        summaries = []

        for row in results.result.data_array:
            record = dict(zip(col_names, row))
            summaries.append({
                "title": record.get("title", "Unknown"),
                "reading_date": record.get("reading_date", "N/A"),
                "chunk_index": record.get("chunk_index", 0),
                "excerpt": record.get("chunk_text", ""),
            })

        logger.info(f"Retrieved {len(summaries)} summaries")
        return summaries

    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        return []


# COMMAND ----------

# ---------------------------------------------------------------------------
# Retrieval Node
# ---------------------------------------------------------------------------


@mlflow.trace(name="retrieval_node")
def retrieval_node(state: AgentState) -> AgentState:
    """
    Retrieval node: fetch relevant summaries from vector index.
    """
    summaries = retrieve_summaries(state["user_query"], num_results=3)
    state["retrieved_summaries"] = summaries
    return state


# COMMAND ----------

# ---------------------------------------------------------------------------
# Generation Node
# ---------------------------------------------------------------------------

llm = ChatDatabricks(
    endpoint=config.llm_endpoint,
    temperature=config.temperature,
    max_tokens=config.max_tokens,
)


@mlflow.trace(name="generation_node", span_type=mlflow.entities.SpanType.LLM)
def generation_node(state: AgentState) -> AgentState:
    """
    Generation node: create recommendations based on retrieved context.
    """
    user_query = state["user_query"]
    summaries = state["retrieved_summaries"]

    # Build context from retrieved summaries
    context = "\n\n".join([
        f"Book: {s['title']} (Read: {s['reading_date']})\nExcerpt: {s['excerpt']}"
        for s in summaries
    ])

    # System prompt for book recommendations
    system_prompt = """You are an expert book recommender who reads summaries and provides thoughtful recommendations.

Given the user's query and relevant book summaries, provide:
1. A concise analysis (2-3 sentences) of what the summaries reveal about the topic
2. JSON array of book recommendations with this structure:
{
  "title": "Book Title",
  "reason": "Why this book matches the user's interest based on the summary",
  "key_insight": "One key takeaway from the summary"
}

Always cite specific sections from the summaries. Be honest - if summaries don't cover a topic well, say so.
Return ONLY the JSON array, no other text."""

    # Build messages
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"User Query: {user_query}\n\nRelevant Book Summaries:\n{context}"
        ),
    ]

    # Generate response
    logger.info("Calling LLM to generate recommendations...")
    response = llm.invoke(messages)
    llm_response = response.content

    state["llm_response"] = llm_response

    # Parse recommendations from JSON
    try:
        # Extract JSON from response (handle markdown code blocks)
        json_str = llm_response
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0]
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0]

        recommendations = json.loads(json_str.strip())
        if not isinstance(recommendations, list):
            recommendations = [recommendations]

        state["recommendations"] = recommendations
        logger.info(f"Parsed {len(recommendations)} recommendations")
    except Exception as e:
        logger.warning(f"Failed to parse JSON recommendations: {e}")
        state["recommendations"] = []

    return state


# COMMAND ----------

# ---------------------------------------------------------------------------
# Build LangGraph
# ---------------------------------------------------------------------------

workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("retrieve", retrieval_node)
workflow.add_node("generate", generation_node)

# Add edges: START → retrieve → generate → END
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compile
graph = workflow.compile()

logger.info("LangGraph compiled successfully")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Visualize Graph
# ---------------------------------------------------------------------------

# Draw the graph structure
try:
    graph_image = graph.get_graph().draw_mermaid_png()
    display(graph_image)  # noqa: F821
    logger.info("Graph visualization displayed")
except Exception as e:
    logger.warning(f"Could not visualize graph: {e}")
    # Fallback: print graph structure
    print(graph.get_graph().draw_mermaid())

# COMMAND ----------

# ---------------------------------------------------------------------------
# Test the Agent
# ---------------------------------------------------------------------------


@mlflow.trace(name="run_book_recommender")
def recommend_books(user_query: str) -> dict[str, Any]:
    """
    Full RAG pipeline: retrieve summaries + generate recommendations.

    Args:
        user_query: User's question about books/themes

    Returns:
        Dict with query, summaries, and recommendations
    """
    logger.info(f"Processing query: {user_query}")

    # Run the graph
    final_state = graph.invoke({
        "user_query": user_query,
        "retrieved_summaries": [],
        "llm_response": "",
        "recommendations": [],
    })

    return {
        "query": final_state["user_query"],
        "summaries_retrieved": len(final_state["retrieved_summaries"]),
        "books_recommended": len(final_state["recommendations"]),
        "recommendations": final_state["recommendations"],
        "retrieved_summaries": final_state["retrieved_summaries"],
    }


# Test queries
test_queries = [
    "Which books discuss AI safety and its implications?",
    "I'm interested in understanding human decision-making and psychology.",
    "Recommend books that combine philosophy with practical insights.",
]

logger.info("Running test queries with full RAG pipeline...")
for i, query in enumerate(test_queries, 1):
    logger.info(f"\n[Query {i}] {query}")
    result = recommend_books(query)

    logger.info(f"Retrieved {result['summaries_retrieved']} summaries")
    logger.info(f"Generated {result['books_recommended']} recommendations")

    # Display recommendations
    display({  # noqa: F821
        "query": result["query"],
        "recommendations": result["recommendations"],
    })

# COMMAND ----------

logger.info("Agent testing complete!")

display({  # noqa: F821
    "agent_status": "ready",
    "vs_index": config.full_vs_index_name,
    "llm_endpoint": config.llm_endpoint,
    "retrieval_top_k": 3,
    "tracing": "enabled",
    "graph_compiled": True,
})
