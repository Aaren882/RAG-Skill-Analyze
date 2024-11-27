import ollama
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

client = chromadb.Client()
# collections = client.list_collections()
# print(collections)


text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
chunks = text_splitter.split_documents(data)

collection = client.create_collection(name="local-rag")

# store each document in a vector embedding database
for i, d in enumerate(chunks):
  response = ollama.embeddings(model="nomic-embed-text", prompt=d)
  embedding = response["embedding"]
  collection.add(
    ids=[str(i)],
    embeddings=[embedding],
    documents=[d]
  )

# an example prompt
prompt = """You are an AI language model assistant. Your task is to generate
  the given user question to retrieve relevant documents from
  a vector database. By generating answer base on the user question, your
  goal is to help the user overcome some of the limitations of the distance-based
  similarity search.
  Original question: {question}"""

# generate an embedding for the prompt and retrieve the most relevant doc
response = ollama.embeddings(
  prompt=prompt,
  model="llama3.1"
)
results = collection.query(
  query_embeddings=[response["embedding"]],
  n_results=1
)
data = results['documents'][0][0]