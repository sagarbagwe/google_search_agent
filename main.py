import os

# LangGraph and main agent imports
import vertexai
from vertexai import agent_engines

# Imports for our custom tool
from langchain_core.tools import tool
from google import genai
from google.genai.types import GenerateContentConfig, GoogleSearch, Tool as GoogleGenAITool

# --- The Agent's Only Tool ---
# This tool directly answers a question using the `google-genai` library's
# automatic grounding feature. It's best for getting a quick, final answer.
@tool
def google_search_grounded_answer(query: str) -> str:
    """
    Provides a single, grounded, fact-based answer to a specific question by using Google Search.
    This is the only tool available to find up-to-date information about any topic,
    especially current events or specific facts.
    """
    print(f"--- Calling Custom Grounded Tool with query: '{query}' ---")
    genai_client = genai.Client()
    response = genai_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=query,
        config=GenerateContentConfig(
            tools=[
                GoogleGenAITool(google_search=GoogleSearch())
            ],
        ),
    )
    return response.text

# --- The Agent ---
# We define a powerful model and give it the list containing our single tool.
agent = agent_engines.LanggraphAgent(
    model="gemini-2.5-flash",
    tools=[google_search_grounded_answer], # The agent's entire toolkit
)

print("Agent with single grounded tool created successfully.")

# --- Query the Agent ---
# Let's ask a question that requires up-to-date information.
print("\nQuerying the agent...")
response = agent.query(
    input={"messages": [("user", "What are the most anticipated movies scheduled for release in October 2025?")]}
)


# --- Debug and Print Final Response ---
print("\n--- Debugging: Inspecting the structure of the last message ---")
last_message = response['messages'][-1]
print(last_message)


print("\nFinal Agent Response:")
# UPDATED: We access the content from the 'kwargs' dictionary inside the message
print(last_message['kwargs']['content'])