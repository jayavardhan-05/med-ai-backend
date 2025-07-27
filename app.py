import os
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from langchain_together import Together
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- 1. DEFINE REQUEST AND RESPONSE MODELS ---
class Query(BaseModel):
    question: str

# --- 2. LOAD MODELS AND DATABASE AT STARTUP ---
print("Loading models and database... This may take a moment.")

if 'TOGETHER_API_KEY' not in os.environ:
    raise ValueError("TOGETHER_API_KEY environment variable not set.")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db_path = "my_faiss_index"
if not os.path.exists(db_path):
    raise FileNotFoundError(f"FAISS database not found at {db_path}.")
db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

llm = Together(
    model="meta-llama/Llama-3-70b-chat-hf",
    temperature=0.1,
    max_tokens=1024
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)
print("✅ Models and database loaded successfully.")

# --- 3. CREATE THE FASTAPI APP ---
app = FastAPI(
    title="Pathology Q&A API",
    description="An API to ask questions about a pathology document."
)

# --- 4. DEFINE THE API ENDPOINT ---
@app.post("/ask", summary="Ask a question")
async def ask_question(query: Query):
    question = query.question
    result = qa_chain.invoke({"query": question})
    sources = [
        {"content": doc.page_content, "metadata": doc.metadata}
        for doc in result.get('source_documents', [])
    ]
    return {"answer": result.get('result'), "sources": sources}