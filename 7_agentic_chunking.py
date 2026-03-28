from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Initialize the LLM
llm = init_chat_model(
    model="Qwen/Qwen2.5-3B-Instruct",
    model_provider="huggingface",
    temperature=0.1,
    max_tokens=512
)

# Tesla text to chunk
tesla_text = """Tesla's Q3 Results
Tesla reported record revenue of $25.2B in Q3 2024.
The company exceeded analyst expectations by 15%.
Revenue growth was driven by strong vehicle deliveries.

Model Y Performance  
The Model Y became the best-selling vehicle globally, with 350,000 units sold.
Customer satisfaction ratings reached an all-time high of 96%.
Model Y now represents 60% of Tesla's total vehicle sales.

Production Challenges
Supply chain issues caused a 12% increase in production costs.
Tesla is working to diversify its supplier base.
New manufacturing techniques are being implemented to reduce costs."""

# Create the prompt
prompt = f"""
You are a text chunking expert.

Insert the token <<<SPLIT>>> ONLY between logical sections.

Rules:
- Do NOT split every sentence
- Group sentences that belong to the same topic
- Each chunk should be under 200 characters
- Do NOT change the text
- Do NOT explain anything

Return ONLY the text with <<<SPLIT>>> markers.

Text:
{tesla_text}
"""

# Get AI response
print("🤖 Asking AI to chunk the text...")
response = llm.invoke(prompt)
marked_text = response.content

# Remove chat template
if "<|im_start|>assistant" in marked_text:
    marked_text = marked_text.split("<|im_start|>assistant")[-1]

if "<|im_end|>" in marked_text:
    marked_text = marked_text.split("<|im_end|>")[0]

# Split the text at the markers
# chunks = marked_text.split("<<<SPLIT>>>")
chunks = re.split(r"<<<?SPLIT>>>?", marked_text)

# Clean up the chunks (remove extra whitespace)
clean_chunks = []
for chunk in chunks:
    cleaned = chunk.strip()
    if cleaned:  # Only keep non-empty chunks
        clean_chunks.append(cleaned)

# Show results
print("\n🎯 AGENTIC CHUNKING RESULTS:")
print("=" * 50)

for i, chunk in enumerate(clean_chunks, 1):
    print(f"Chunk {i}: ({len(chunk)} chars)")
    print(f'"{chunk}"')
    print()