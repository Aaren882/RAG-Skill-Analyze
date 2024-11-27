from langchain_community.document_loaders import UnstructuredPDFLoader,UnstructuredWordDocumentLoader,UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
import pandas as pd

local_path = "Docs/Test.docx"
# local_path = "Docs/課業.xlsx"

if local_path:
  file_type = -1
  file_types = ["pdf","docx","xlsx"]
  for _i,_j in enumerate(file_types):
    if local_path.find("." + _j) > -1:
      file_type = _i
      break
  
  # Local PDF file uploads
  if file_type > -1:
    match file_type:
      case 0: # PDF file
        loader = UnstructuredPDFLoader(file_path=local_path)
      case 1: # Word file
        loader = UnstructuredWordDocumentLoader(file_path=local_path)
      case 2: # Excel file
        loader = UnstructuredExcelLoader(file_path=local_path)

    data = loader.load()
    
    # Split and chunk 
    text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=1000,
      chunk_overlap=300,
      length_function=len,
      is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(data)

    vector_db = Chroma.from_documents(
      documents=chunks, 
      embedding=OllamaEmbeddings(model="nomic-embed-text",show_progress=True),
      collection_name="local-rag-Excel",
      persist_directory="Local_RAG_DB"
    )