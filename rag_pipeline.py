import os
from dotenv import load_dotenv
from glob import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

# --- Load environment ---
load_dotenv()
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# --- Global embeddings ---
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- Load & Embed PDFs ---
def load_and_embed_pdfs(pdf_folder="policy_docs"):
    all_docs = []
    for file in glob(f"{pdf_folder}/*.pdf"):
        loader = PyPDFLoader(file)
        docs = loader.load()
        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("policy_vector_store")
    return vectorstore.as_retriever()

# --- Load retriever (default) ---
retriever = load_and_embed_pdfs()

# --- Prompt Template ---
prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="""
You are an expert assistant that indexes company policy documents and allows employees to ask natural language questions like “What’s our refund policy?” or “How to request design assets?”.

Query:
{question}

Relevant Clauses:
{context}

Respond in JSON with the following fields:
- Answer: Give the answer
"""
)

# --- LLM Model ---
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)

# --- Default RAG Chain ---
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- Enhanced Chain with Debug Info ---
def rag_chain_with_debug(query: str, top_k: int = 5, threshold: float = 0.65):
    db = FAISS.load_local("policy_vector_store", embeddings, allow_dangerous_deserialization=True)
    all_docs_with_scores = db.similarity_search_with_score(query, k=top_k)

    filtered_docs = [doc for doc, score in all_docs_with_scores if score >= threshold]
    context = "\n\n---\n\n".join(doc.page_content for doc in filtered_docs) or "No relevant context found."

    prompt_text = prompt.format(context=context, question=query)
    answer = llm.invoke(prompt_text)

    return {
        "response": answer,
        "chunks": context,
        "raw_prompt": prompt_text
    }
