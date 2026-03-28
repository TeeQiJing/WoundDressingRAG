from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_chroma import Chroma
import json
import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Vector Store ──────────────────────────────────────────────────────────────

def load_vector_store(persist_directory="dbv1/chroma_db"):
    """Load existing ChromaDB vector store"""
    print("📂 Loading ChromaDB vector store...")
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print(f"✅ Vector store loaded from {persist_directory}")
    print(f"📊 Total documents: {vectorstore._collection.count()}")
    return vectorstore

db = load_vector_store("dbv1/chroma_db")
retriever = db.as_retriever(search_kwargs={"k": 3})

# ── Answer Generation (matches notebook pipeline exactly) ─────────────────────

def generate_final_answer(chunks, query):
    """Generate final answer using content from retrieved chunks"""
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # Build the text prompt
        prompt_text = f"""Based on the following documents, please answer this question: {query}

CONTENT TO ANALYZE:
"""
        for i, chunk in enumerate(chunks):
            prompt_text += f"--- Document {i+1} ---\n"

            if "original_content" in chunk.metadata:
                original_data = json.loads(chunk.metadata["original_content"])

                # Add raw text
                raw_text = original_data.get("raw_text", "")
                if raw_text:
                    prompt_text += f"TEXT:\n{raw_text}\n\n"

                # Add tables as HTML
                tables_html = original_data.get("tables_html", [])
                if tables_html:
                    prompt_text += "TABLES:\n"
                    for j, table in enumerate(tables_html):
                        prompt_text += f"Table {j+1}:\n{table}\n\n"

            prompt_text += "\n"

        prompt_text += """Please provide a clear, comprehensive answer using the text and tables above.
If the documents don't contain sufficient information to answer the question, say "I don't have enough information to answer that question based on the provided documents."

ANSWER:"""

        # Build message — text only (no image output as per requirement)
        message_content = [{"type": "text", "text": prompt_text}]

        message = HumanMessage(content=message_content)
        response = llm.invoke([message])
        return response.content

    except Exception as e:
        print(f"❌ Answer generation failed: {e}")
        return "Sorry, I encountered an error while generating the answer."

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/get_answer")
async def get_answer(question: str = Form(...)):
    try:
        # Retrieve relevant chunks
        chunks = retriever.invoke(question)

        # Generate answer using notebook pipeline
        result = generate_final_answer(chunks, question)

        return JSONResponse({"result": result})

    except Exception as e:
        return JSONResponse(
            {"result": f"Sorry, an error occurred: {str(e)}"},
            status_code=500
        )