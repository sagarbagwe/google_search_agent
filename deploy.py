import os
import sys
import vertexai
from dotenv import load_dotenv

# Vertex AI Agent Engine Imports
from vertexai.preview.reasoning_engines import AdkApp
from vertexai import agent_engines

# LangGraph + Tool Imports
from langchain_core.tools import tool
from google import genai
from google.genai.types import GenerateContentConfig, GoogleSearch, Tool as GoogleGenAITool

# --- 1. Configuration ---
print("🚀 Starting LangGraph Google Search Agent deployment process...")
load_dotenv()

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# A GCS bucket is required for staging the agent artifacts during deployment.
# Ensure this bucket exists in your project.
STAGING_BUCKET = f"gs://{PROJECT_ID}-agent-engine-staging"

if not all([PROJECT_ID, LOCATION, GEMINI_MODEL]):
    print("❌ Error: Missing required environment variables. Please check your .env file.", file=sys.stderr)
    sys.exit(1)

print(f"✅ Configuration loaded for project '{PROJECT_ID}'")

# Initialize Vertex AI with the project, location, and staging bucket.
vertexai.init(project=PROJECT_ID, location=LOCATION, staging_bucket=STAGING_BUCKET)

# --- 2. Define Tool ---
print("🛠️  Defining grounded Google Search tool...")

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
        model=GEMINI_MODEL,
        contents=query,
        config=GenerateContentConfig(
            tools=[
                GoogleGenAITool(google_search=GoogleSearch())
            ],
        ),
    )
    return response.text

print("✅ Tool defined.")

# --- 3. Define the Agent ---
print("🤖 Defining the LangGraph Agent...")
agent = agent_engines.LanggraphAgent(
    model=GEMINI_MODEL,
    tools=[google_search_grounded_answer],
)
print("✅ Agent definition complete.")

# --- 4. Package and Deploy ---
print("📦 Packaging agent with AdkApp...")
app = AdkApp(
    agent=agent,
    enable_tracing=True,
)

deployment_requirements = [
    "google-cloud-aiplatform[agent_engines,adk]>=1.55.0",
    "google-adk>=0.1.0",
    "python-dotenv>=1.0.0",
    "google-auth>=2.29.0",
    "langchain-core>=0.2.0",
    "google-genai>=0.1.0",   # required for grounding
]

print("🚢 Deploying to Vertex AI Agent Engine with explicit requirements... (This may take 15-20 minutes)")

remote_app = agent_engines.create(
    app,
    requirements=deployment_requirements
)

print("\n🎉 Deployment successful!")
print("Agent Resource Name:", remote_app.resource_name)
print(remote_app)
