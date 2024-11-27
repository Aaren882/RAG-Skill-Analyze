# from data_receiver import data
from langchain_community.embeddings import OllamaEmbeddings

# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma

# from langchain_community.vectorstores import Milvus
# from pymilvus import MilvusClient
# client = MilvusClient("local_RAG.db")

# DB (Milvus) [!! Currently Milvus local VectorDB isn't support Windows yet]
# vector_db = Milvus.from_documents(
#   documents=chunks, 
#   embedding=OllamaEmbeddings(model="nomic-embed-text",show_progress=True),
#   collection_name="local-rag"
# )

# Add to vector database (Chroma) [get Data from local Vector DataBase]
vector_db = Chroma(
  embedding_function=OllamaEmbeddings(model="nomic-embed-text",show_progress=True),
  collection_name="local-rag-Excel",
  persist_directory="Local_RAG_DB"
)
