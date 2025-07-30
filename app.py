import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# --- 1. DEFINE REQUEST AND RESPONSE MODELS ---
class Query(BaseModel):
    question: str

# --- 2. CREATE THE FASTAPI APP ---
app = FastAPI(
    title="Pathology Q&A API",
    description="An API to ask questions about a pathology document."
)

# --- 3. LAZY LOADING SETUP ---
# The QA chain is initialized to None. It will be created on the first API call
# to prevent using too much memory on startup.
qa_chain = None

def initialize_qa_chain():
    """
    Loads all models and the database. This function is expensive and runs only once.
    """
    global qa_chain

    # This check ensures the expensive loading process runs only once.
    if qa_chain is None:
        print("Lazily loading models and database for the first time...")

        # Import necessary libraries here to keep startup memory low
        from langchain_together import Together
        from langchain.chains import RetrievalQA
        from langchain_community.vectorstores import FAISS
        # FIX: Use the recommended, updated import for HuggingFaceEmbeddings
        from langchain_huggingface import HuggingFaceEmbeddings

        # --- Check for Environment Variable ---
        if 'TOGETHER_API_KEY' not in os.environ:
            # This will stop the process if the key is missing.
            raise ValueError("TOGETHER_API_KEY environment variable not set.")

        # --- Load Embeddings Model ---
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # --- Load FAISS Database ---
        db_path = "my_faiss_index"
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"FAISS database not found at {db_path}. Cannot start service.")
        
        db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

        # --- Load LLM ---
        # MEMORY FIX: Switched from the very large 70B model to the 8B model.
        # The 70B model requires hundreds of GB of RAM and will crash on standard hosting.
        llm = Together(
            model="meta-llama/Llama-3-8b-chat-hf",
            temperature=0.1,
            max_tokens=1024
        )
        
        # --- Create and cache the QA chain in the global variable ---
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        print("✅ Models and database loaded successfully.")

# --- 4. DEFINE THE API ENDPOINT ---
@app.post("/ask", summary="Ask a question")
async def ask_question(query: Query):
    """
    Asks a question to the QA system. Initializes the models on the first request.
    """
    try:
        # This will initialize the chain on the first call and do nothing on subsequent calls.
        initialize_qa_chain()
        
        question = query.question
        result = qa_chain.invoke({"query": question})
        
        sources = [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in result.get('source_documents', [])
        ]
        
        return {"answer": result.get('result'), "sources": sources}
    except FileNotFoundError as e:
        # Handle the case where the database file is missing during initialization
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        # Handle other potential errors during initialization or invocation
        print(f"An error occurred: {e}") # Log the error for debugging
        raise HTTPException(status_code=500, detail="An internal error occurred while processing the request.")
