from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Load the document
print("Loading document...")
loader = TextLoader("document.txt", encoding="utf-8")
documents = loader.load()

# 2. Split the document into chunks
print("Splitting document into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_documents(documents)

# 3. Create embeddings
print("Creating embeddings... (This will download the model on the first run)")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Create the FAISS vector database and save it
print("Creating and saving the FAISS database...")
db = FAISS.from_documents(docs, embeddings)
db.save_local("my_faiss_index")

print("\n✅ Success! The 'my_faiss_index' folder has been created.")