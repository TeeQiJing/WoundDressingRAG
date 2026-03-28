from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os

load_dotenv() # load environment variables from a .env file

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

persistent_directory = "db/chroma_db"

# Load embeddings and vector store
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}  
)

# Search for relevant documents
query = "What is the founder of Tesla?"

# retriever = db.as_retriever(search_kwargs={"k": 5})

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 5,
        "score_threshold": 0.3  # Only return chunks with cosine similarity ≥ 0.3
    }
)

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
# Display results
print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")


# Combine the query and the relevant document contents
combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""

# Create a ChatOpenAI model
model = init_chat_model(
    model="Qwen/Qwen2.5-3B-Instruct",
    model_provider="huggingface",
    temperature=0.2,
    max_tokens=512
)

# Define the messages for the model
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

# Invoke the model with the combined input
result = model.invoke(messages)

# Display the full result and content only
print("\n--- Generated Response ---")
# print("Full result:")
# print(result)
print("Content only:")
print(result.content)

# Synthetic Questions: 

# 1. "What was NVIDIA's first graphics accelerator called?"
# 2. "Which company did NVIDIA acquire to enter the mobile processor market?"
# 3. "What was Microsoft's first hardware product release?"
# 4. "How much did Microsoft pay to acquire GitHub?"
# 5. "In what year did Tesla begin production of the Roadster?"
# 6. "Who succeeded Ze'ev Drori as CEO in October 2008?"
# 7. "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"
# 8. "What was the original name of Microsoft before it became Microsoft?"