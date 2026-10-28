"""
Personalized Search Agent with Mem0 + Perplexity Search API
Uses LangChain agent pattern with a Perplexity Search API tool for personalized search based on user memories stored in Mem0.
"""

import logging
import os
from datetime import datetime

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from perplexity import Perplexity

from mem0 import MemoryClient

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize clients
mem0_client = MemoryClient()

# Perplexity Search API client. Reads PERPLEXITY_API_KEY and tags requests so Perplexity can attribute mem0 usage.
perplexity_client = Perplexity(
    api_key=os.environ.get("PERPLEXITY_API_KEY"),
    default_headers={
        "X-Pplx-Integration": "mem0",
        "User-Agent": "mem0-personalized-search-example",
    },
)

# Set custom instructions to infer facts and memory to understand user preferences
mem0_client.project.update(
    custom_instructions="""
INFER THE MEMORIES FROM USER QUERIES EVEN IF IT'S A QUESTION.

We are building the personalized search for which we need to understand about user's preferences and life
and extract facts and memories out of it accordingly.

BE IT TIME, LOCATION, USER'S PERSONAL LIFE, CHOICES, USER'S PREFERENCES, we need to store those for better personalized search.
"""
)

llm = ChatOpenAI(model="gpt-4.1-nano-2025-04-14", temperature=0.2)


@tool
def perplexity_search(query: str, max_results: int = 10) -> dict:
    """Search the web with the Perplexity Search API.

    Args:
        query: The search query. Include location, dietary, or family context when it is relevant.
        max_results: Number of results to return, between 1 and 20.

    Returns a dict with the query and a list of results. Each result has title, url, content (snippet), and date.
    """
    # The Search API accepts 1 to 20 results per request
    max_results = max(1, min(max_results, 20))
    response = perplexity_client.search.create(query=query, max_results=max_results)

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
    """Simulate realistic user conversation history"""
    conversations = [
        [
            {
                "role": "user",
                "content": "What will be the weather today at Los Angeles? I need to go to pick up my daughter from office.",
            },
            {
                "role": "assistant",
                "content": "I'll check the weather in LA for you, so that you can plan you daughter's pickup accordingly.",
            },
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
    for conversation in conversations:
        mem0_client.add(conversation, user_id=user_id)


def get_user_context(user_id, query):
    """Retrieve relevant user memories from Mem0"""
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

    except Exception as e:
        logger.error(f"Error retrieving user context: {e}")
        return "Error retrieving user context."


def create_personalized_search_agent(user_context):
    """Create a LangChain agent for personalized search using the Perplexity Search API"""

    tools = [perplexity_search]

    # Create personalized search prompt
    system_prompt = f"""You are a personalized search assistant. You help users find information that's relevant to their specific context and preferences.

USER CONTEXT AND PREFERENCES:
{user_context}

YOUR ROLE:
1. Analyze the user's query and their personal context/preferences above
2. Look for patterns in the context to understand their preferences, location, lifestyle, family situation, etc.
3. Create enhanced search queries that incorporate relevant personal context you discover
4. Use the perplexity_search tool everytime with enhanced queries to find personalized results


INSTRUCTIONS:
- Study the user memories carefully to understand their situation
- If any questions ask something related to nearby, close to, etc. refer to previous user context for identifying locations and enhance search query based on that.
- If memories mention specific locations, consider them for local searches
- If memories reveal dietary preferences or restrictions, factor those in for food-related queries
- If memories show family context, consider family-friendly options
- If memories indicate work style or interests, incorporate those when relevant
- Use perplexity_search tool everytime with enhanced queries (based on above context)
- Always explain which specific memories led you to personalize the search in certain ways

Do NOT assume anything not present in the user memories."""

    # Create agent
    agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)

    return agent


def conduct_personalized_search(user_id, query):
    """
    Personalized search workflow using LangChain agent + Perplexity Search API + Mem0

    Returns search results with user personalization details
    """
    logger.info(f"Starting personalized search for user {user_id}: {query}")
    start_time = datetime.now()

    try:
        # Get user context from Mem0
        user_context = get_user_context(user_id, query)

        # Create personalized search agent
        agent = create_personalized_search_agent(user_context)

        # Run the agent
        response = agent.invoke({"messages": [HumanMessage(content=query)]})
        messages = response["messages"]
        agent_response = messages[-1].content

        # Log the enhanced search queries the agent sent to Perplexity
        search_queries_used = [
            tool_call["args"].get("query", "")
            for message in messages
            if isinstance(message, AIMessage)
            for tool_call in message.tool_calls
            if tool_call["name"] == "perplexity_search"
        ]
        logger.info(f"Perplexity search queries used: {search_queries_used}")

        # Store this search interaction in Mem0 for user preferences
        store_search_interaction(user_id, query, agent_response)

        # Compile results
        duration = (datetime.now() - start_time).total_seconds()

        results = {"agent_response": agent_response}

        logger.info(f"Personalized search completed in {duration:.2f}s")
        return results

    except Exception as e:
        logger.error(f"Error in personalized search workflow: {e}")
        return {"error": str(e)}


def store_search_interaction(user_id, original_query, agent_response):
    """Store search interaction in Mem0 for future personalization"""
    try:
        interaction = [
            {"role": "user", "content": f"Searched for: {original_query}"},
            {
                "role": "assistant",
                "content": f"Provided personalized results based on user preferences: {agent_response}",
            },
        ]

        mem0_client.add(messages=interaction, user_id=user_id)

        logger.info(f"Stored search interaction for user {user_id}")

    except Exception as e:
        logger.error(f"Error storing search interaction: {e}")


def personalized_search_agent():
    """Example of the personalized search agent"""

    user_id = "john"

    # Setup user history
    print("\nSetting up user history from past conversations...")
    setup_user_history(user_id)  # This is one-time setup

    # Test personalized searches
    test_queries = [
        "good coffee shops nearby for working",
        "what can we gift our daughter for birthday? what's trending?",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n ----- {i}️⃣ PERSONALIZED SEARCH -----")
        print(f"Query: '{query}'")

        # Run personalized search
        results = conduct_personalized_search(user_id, query)

        if results.get("error"):
            print(f"Error: {results['error']}")

        else:
            print(f"Agent response: {results['agent_response']}")


if __name__ == "__main__":
    personalized_search_agent()
