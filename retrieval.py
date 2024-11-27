from Set_DB import vector_db

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

# LLM from Ollama
# local_model = "llama3.1"
local_model = "gemma2"
llm = ChatOllama(model=local_model)
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate
    the given user question to retrieve relevant documents from
    a vector database. By generating answer base on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search.
    Original question: {question}""",
)
retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(
      search_type = "mmr"
    ), 
    llm,
    prompt=QUERY_PROMPT
)

# RAG prompt
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)