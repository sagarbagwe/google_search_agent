import os
import vertexai
from dotenv import load_dotenv
from vertexai.preview import agent_engines # <-- CORRECT IMPORT

# --- 1. Configuration ---
print("🚀 Loading configuration...")
load_dotenv()

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

if not all([PROJECT_ID, LOCATION]):
    raise ValueError("Missing required environment variables GOOGLE_CLOUD_PROJECT or GOOGLE_CLOUD_LOCATION")

print(f"✅ Configuration loaded for project '{PROJECT_ID}'")

# --- 2. Initialize Vertex AI ---
vertexai.init(project=PROJECT_ID, location=LOCATION)

# --- 3. Get a reference to your deployed agent ---
# Replace this with the resource name from your deployment output
AGENT_RESOURCE_NAME = "projects/113207776071/locations/us-central1/reasoningEngines/287812561733156864"

print(f"🔗 Connecting to remote agent: {AGENT_RESOURCE_NAME}")

# Use the directly imported module here
remote_agent = agent_engines.get(AGENT_RESOURCE_NAME) 

print("✅ Connected successfully!")

# --- 4. Query the agent ---
# This question will force the agent to use its Google Search tool.
prompt = "What is the latest news about the Artemis program?" 
print(f"\n💬 Sending query: '{prompt}'")

response = remote_agent.query(
    input=prompt,
)

# --- 5. Print the response ---
print("\n🤖 Agent Response:")
print(response)