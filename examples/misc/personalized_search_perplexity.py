"""
Run two web searches using user memories from Mem0.
Requires OPENAI_API_KEY, MEM0_API_KEY, and PERPLEXITY_API_KEY.
Importing this module updates the selected Mem0 project's custom instructions.
"""

import logging
import os
import time
from datetime import datetime

import httpx
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from perplexity import APIError, Perplexity

from mem0 import MemoryClient

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

mem0_client = MemoryClient()

# Perplexity Search API client. Reads PERPLEXITY_API_KEY and tags requests so Perplexity can attribute mem0 usage.
# Reduce the library's default HTTP timeout for this interactive demo; retain its retries.
perplexity_client = Perplexity(
    api_key=os.environ.get("PERPLEXITY_API_KEY"),
    timeout=30.0,
    default_headers={
        "X-Pplx-Integration": "mem0",
        "User-Agent": "mem0-personalized-search-example",
    },
)

# This updates the settings of the Mem0 project your MEM0_API_KEY belongs to, so use a dedicated test project.
mem0_client.project.update(
    custom_instructions="""
Extract facts the user explicitly states, including facts within questions.
Focus on location, dietary restrictions, family details, work needs, and interests.
Do not infer where a user lives from a question about the weather in a city.
"""
)

llm = ChatOpenAI(model="gpt-4.1-nano-2025-04-14", temperature=0.2)


@tool
def perplexity_search(query: str, max_results: int = 10) -> dict:
    """Search the web with the Perplexity Search API.

    Args:
        query: The search query. Include location, dietary, or family context when it is relevant.
        max_results: Number of results to return, between 1 and 20.

    On success, return {"query", "results"}; each result contains title, url, content, and date (None when omitted).
    On API failure, return {"query", "error"} with a fixed recovery message.
    """
    # The Search API accepts 1 to 20 results per request
    max_results = max(1, min(max_results, 20))
    try:
        response = perplexity_client.search.create(query=query, max_results=max_results)
    except APIError as e:
        # Fixed message per status; the response body is never forwarded to the agent or the log.
        status_code = getattr(e, "status_code", None)
        message = {
            401: "Perplexity authentication failed. Check PERPLEXITY_API_KEY.",
            422: "Perplexity rejected the request. Check query and max_results.",
            429: "Perplexity rate limit reached. Wait and retry.",
        }.get(status_code, "Perplexity Search failed. Retry the request; if it still fails, check the service status.")
        logger.error(message)
        return {"query": query, "error": message}

    results = [
        {
            "title": result.title,
            "url": result.url,
            "content": result.snippet,
            "date": result.date,
        }
        for result in response.results
    ]

    return {"query": query, "results": results}


def setup_user_history(user_id):
    """Seed five conversations for user_id and wait for their ingestion events."""
    conversations = [
        [
            {
                "role": "user",
                "content": "I live in Los Angeles. What's the weather today? I need to pick up my daughter after work.",
            },
            {"role": "assistant", "content": "I'll check the weather in Los Angeles before your pickup."},
        ],
        [
            {"role": "user", "content": "I'm looking for vegan restaurants in Santa Monica"},
            {"role": "assistant", "content": "I'll find great vegan options in Santa Monica."},
        ],
        [
            {"role": "user", "content": "My 7-year-old daughter is allergic to peanuts"},
            {
                "role": "assistant",
                "content": "I'll remember to check for peanut-free options in future recommendations.",
            },
        ],
        [
            {"role": "user", "content": "I work remotely and need coffee shops with good wifi"},
            {"role": "assistant", "content": "I'll find remote-work-friendly coffee shops."},
        ],
        [
            {"role": "user", "content": "We love hiking and outdoor activities on weekends"},
            {"role": "assistant", "content": "Great! I'll keep your outdoor activity preferences in mind."},
        ],
    ]

    logger.info(f"Setting up user history for {user_id}")
    event_ids = []
    for conversation in conversations:
        # Platform add() is asynchronous: it returns {"event_id": ..., "status": "PENDING"} and extracts memories later
        response = mem0_client.add(conversation, user_id=user_id)
        event_ids.append(response["event_id"])

    wait_for_events(event_ids)


def wait_for_events(event_ids, timeout=60):
    """Poll GET /v1/event/{event_id}/ until every add() event has SUCCEEDED.

    Raises RuntimeError if any event FAILED or if the events have not all succeeded within `timeout` seconds,
    so a search never runs against incomplete memories.
    """
    pending = set(event_ids)
    failed = []
    deadline = time.monotonic() + timeout

    def seconds_left():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            detail = f"pending: {sorted(pending)}" if pending else "the last response arrived after the deadline"
            raise RuntimeError(
                f"Mem0 ingestion did not finish within {timeout}s ({detail}). Check event status before retrying setup."
            )
        return remaining

    while pending:
        for event_id in sorted(pending):
            # Re-check the budget before every request and bound the request by what is left
            # (the Mem0 client's own default timeout is 300s)
            try:
                event = mem0_client.client.get(f"/v1/event/{event_id}/", timeout=seconds_left())
            except httpx.TimeoutException:
                raise RuntimeError(
                    f"Polling Mem0 event {event_id} timed out. Check your connection and retry setup."
                ) from None
            event.raise_for_status()
            status = event.json().get("status")
            if status == "SUCCEEDED":
                pending.discard(event_id)
            elif status == "FAILED":
                failed.append(event_id)
        if failed:
            raise RuntimeError(
                f"Mem0 ingestion failed for {sorted(failed)}. Check those events in Mem0 before retrying setup."
            )
        if pending:
            time.sleep(min(1, seconds_left()))

    # A SUCCEEDED response that arrives after the deadline does not count
    seconds_left()
    logger.info(f"All {len(event_ids)} memory add events SUCCEEDED")


def get_user_context(user_id, query):
    """Return matching memory text, or a fallback message when context is unavailable."""
    try:
        # For Platform API, user_id goes in filters
        filters = {"user_id": user_id}
        user_memories = mem0_client.search(query=query, filters=filters).get("results", [])

        if user_memories:
            context = "\n".join([f"- {memory['memory']}" for memory in user_memories])
            logger.info(f"Found {len(user_memories)} relevant memories for user {user_id}")
            return context
        else:
            logger.info(f"No relevant memories found for user {user_id}")
            return "No previous user context available."

    except Exception:
        logger.error("Mem0 lookup failed. Check MEM0_API_KEY and retry.")
        return "Memory lookup failed; no user context is available."


def create_personalized_search_agent(user_context):
    """Bind the search tool to an agent with user_context in its system prompt."""

    tools = [perplexity_search]

    system_prompt = f"""Use only the supplied memories to personalize web searches.

USER MEMORIES:
{user_context}

For each request:
1. Add relevant location, dietary, family, or work context to the query.
2. Call perplexity_search with that query.
3. Summarize the results and explain which memories informed the query.

Do not invent facts absent from the memories."""

    agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)

    return agent


def conduct_personalized_search(user_id, query):
    """Return {"agent_response": ...} or a safe {"error": ...}.

    Attempt to queue the query and answer in Mem0 after the agent finishes.
    """
    logger.info(f"Starting personalized search for user {user_id}: {query}")
    start_time = datetime.now()

    try:
        user_context = get_user_context(user_id, query)

        agent = create_personalized_search_agent(user_context)

        response = agent.invoke({"messages": [HumanMessage(content=query)]})
        messages = response["messages"]
        agent_response = messages[-1].content

        search_queries_used = [
            tool_call["args"].get("query", "")
            for message in messages
            if isinstance(message, AIMessage)
            for tool_call in message.tool_calls
            if tool_call["name"] == "perplexity_search"
        ]
        logger.info(f"Perplexity search queries used: {search_queries_used}")

        store_search_interaction(user_id, query, agent_response)

        duration = (datetime.now() - start_time).total_seconds()

        results = {"agent_response": agent_response}

        logger.info(f"Personalized search completed in {duration:.2f}s")
        return results

    except Exception:
        message = "Personalized search failed. Check provider credentials and retry."
        logger.error(message)
        return {"error": message}


def store_search_interaction(user_id, original_query, agent_response):
    """Queue the query and answer for extraction; log a safe message if the write fails."""
    try:
        interaction = [
            {"role": "user", "content": f"Searched for: {original_query}"},
            {
                "role": "assistant",
                "content": f"Provided personalized results based on user preferences: {agent_response}",
            },
        ]

        mem0_client.add(messages=interaction, user_id=user_id)

        logger.info(f"Queued search interaction for user {user_id}")

    except Exception:
        logger.error("Mem0 interaction write failed. Check MEM0_API_KEY and retry.")


def personalized_search_agent():
    """Seed memories for john, then print answers to two example queries."""

    user_id = "john"

    print("\nSetting up user history from past conversations...")
    setup_user_history(user_id)

    test_queries = [
        "good coffee shops nearby for working",
        "what can we gift our daughter for birthday? what's trending?",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n ----- {i}️⃣ PERSONALIZED SEARCH -----")
        print(f"Query: '{query}'")

        results = conduct_personalized_search(user_id, query)

        if results.get("error"):
            print(f"Error: {results['error']}")

        else:
            print(f"Agent response: {results['agent_response']}")


if __name__ == "__main__":
    personalized_search_agent()
