from google import genai
from google.genai.types import (
    GenerateContentConfig,
    GoogleSearch,
    Tool,
)

# The client automatically uses the environment variables you set in Step 2
client = genai.Client()

# Make the API call with the Google Search tool enabled
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="When is the next total solar eclipse in the United States?",
    config=GenerateContentConfig(
        tools=[
            # This is the key part that enables grounding
            Tool(google_search=GoogleSearch())
        ],
    ),
)

# Print the model's text response
print(response.text)

# Example output might be:
# 'The next total solar eclipse visible in the United States will be on April 8, 2024.'