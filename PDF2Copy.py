import os
import shutil
from pathlib import Path
from PIL import Image
import pytesseract
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import streamlit as st
from pymongo import MongoClient
import datetime
import pandas as pd

# MongoDB setup
MONGO_URI = "mongodb+srv://choprasa:Savi3650@cluster1.f2mxsnf.mongodb.net/?retryWrites=true&w=majority&appName=Cluster1"
client = MongoClient(MONGO_URI)
db = client["document_qa"]
collection = db["documents"]
query_results_collection = db["query_results"]

# LLM and Embedding setup
os.environ["GOOGLE_API_KEY"] = "AIzaSyBNWsJZFIpvaldltYnK3j26OmzsakeUgy0"
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.environ["GOOGLE_API_KEY"])

# Prompt Template
prompt_template = PromptTemplate.from_template("""
You are an intelligent assistant designed to answer questions based on the content of uploaded PDF and image documents.
Please provide accurate and helpful answers to the questions asked, using the context provided from the documents.

Context:
{context}

Question:
{question}

Helpful Answer:
""")

# Theme Synthesis Function
def perform_theme_synthesis(doc_sources, model):
    all_texts = [f"From {doc_id}: {data['answer']}" for doc_id, data in doc_sources.items()]
    joined_context = "\n".join(all_texts)

    synthesis_prompt = PromptTemplate.from_template("""
    You are a research assistant helping to synthesize insights from multiple legal or regulatory documents.

    Given the following extracted findings from different documents:

    {context}

    1. Identify common themes that appear in multiple documents.
    2. Clearly list each theme, providing a short description.
    3. For each theme, cite the relevant documents (by name or ID).

    Return the response in markdown with proper sectioning for each theme.
    """)

    synthesis_chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=None,
        return_source_documents=False,
        chain_type_kwargs={"prompt": synthesis_prompt}
    )

    result = synthesis_chain({"query": "Synthesize themes from the findings.", "context": joined_context})
    return result["result"]

# Extraction Functions
def extract_text_from_image(file):
    image = Image.open(file).convert("RGB")
    return pytesseract.image_to_string(image)

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def load_documents(files):
    documents = []
    for file in files:
        suffix = Path(file.name).suffix.lower()
        if suffix in [".jpg", ".jpeg", ".png"]:
            text = extract_text_from_image(file)
            if text.strip():
                metadata = {"file_name": file.name, "file_type": "image", "page": "N/A", "paragraph": "N/A"}
                if not collection.find_one({"file_name": file.name, "file_type": "image"}):
                    collection.insert_one({**metadata, "text": text, "timestamp": datetime.datetime.utcnow()})
                documents.append(Document(page_content=text, metadata=metadata))

        elif suffix == ".pdf":
            try:
                reader = PdfReader(file)
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        paragraphs = [p.strip() for p in page_text.split('\n\n') if p.strip()]
                        for para_num, para_text in enumerate(paragraphs, 1):
                            metadata = {"file_name": file.name, "file_type": "pdf", "page": i + 1, "paragraph": para_num}
                            if not collection.find_one({"file_name": file.name, "page": i + 1, "paragraph": para_num}):
                                collection.insert_one({**metadata, "text": para_text, "timestamp": datetime.datetime.utcnow()})
                            documents.append(Document(page_content=para_text, metadata=metadata))
            except Exception as e:
                st.error(f"Error reading PDF {file.name}: {e}")
    return documents

def split_text(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=64, separators=["\n\n", "\n", ". ", "! ", "? "])
    all_chunks = []
    for doc in documents:
        if doc.metadata.get("paragraph", "N/A") != "N/A":
            all_chunks.append(doc)
            continue
        splits = splitter.split_text(doc.page_content)
        for chunk_num, chunk_text in enumerate(splits, 1):
            all_chunks.append(Document(page_content=chunk_text, metadata={**doc.metadata, "paragraph": chunk_num}))
    return all_chunks

def setup_qa_chain(text_chunks):
    db_path = Path("./vectorstores/db_faiss")
    if db_path.exists():
        shutil.rmtree(db_path)

    vectorstore = FAISS.from_documents(text_chunks, embedding_model)
    vectorstore.save_local(str(db_path))

    memory = ConversationBufferMemory(memory_key="history", input_key="question")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    return RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template, "memory": memory},
    )

def answer_query(qa_chain, query):
    result = qa_chain({"query": query})
    return result['result'], result.get('source_documents', [])

# Streamlit UI
st.set_page_config(page_title="Multi-Document QA with Citations", layout="wide")
st.title("\U0001F4DA Multi-File Document QA (PDF + Images) with Citations")

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.header("\U0001F4C2 Upload Documents")
    uploaded_files = st.file_uploader("Upload PDFs or Images", type=["pdf", "jpg", "jpeg", "png"], accept_multiple_files=True)
    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing..."):
                docs = load_documents(uploaded_files)
                if docs:
                    chunks = split_text(docs)
                    st.session_state.qa_chain = setup_qa_chain(chunks)
                    st.success("Documents processed successfully!")
        else:
            st.warning("Please upload files first.")
    synthesize = st.button("\U0001F9E0 Synthesize Themes")
    if st.button("\U0001F4CA View Past Queries"):
        st.subheader("Query History")
        history = query_results_collection.find().sort("timestamp", -1).limit(10)
        for item in history:
            st.markdown(f"**Query:** {item['query']}  \n**Doc:** {item['document_id']}  \n**Citations:** {', '.join(item['citations'])}")

st.header("\U0001F4AC Ask Questions")
query = st.text_input("Type your question here:")
if st.button("Get Answer"):
    if not query.strip():
        st.warning("Please enter a question.")
    elif st.session_state.qa_chain is None:
        st.warning("Please process documents first.")
    else:
        with st.spinner("Generating answer..."):
            answer, source_docs = answer_query(st.session_state.qa_chain, query)
            st.session_state.chat_history.append(("User", query))

            doc_sources = {}
            for doc in source_docs:
                doc_id = Path(doc.metadata["file_name"]).stem.upper()
                if doc_id not in doc_sources:
                    doc_sources[doc_id] = {"answer": answer, "citations": set()}
                page = doc.metadata.get("page", "N/A")
                para = doc.metadata.get("paragraph", "N/A")
                doc_sources[doc_id]["citations"].add(f"Page {page}, Para {para}")
            # Store the query result in MongoDB
            query_result = {
                "query": query,
                "answer": answer,
                "sources": [
                    {
                        "doc_id": doc_id,
                        "citations": list(data["citations"])
                    } for doc_id, data in doc_sources.items()
                ],
                "timestamp": datetime.datetime.utcnow()
            }
            query_results_collection.insert_one(query_result)

            # Create table data
            table_data = [
                {
                    "Document ID": doc_id,
                    "Extracted Answer": data["answer"],
                    "Citation": "<br>".join(sorted(data["citations"]))
                }
                for doc_id, data in doc_sources.items()
            ]
        
       # Display as DataFrame
            if table_data:
                df = pd.DataFrame(table_data)
                st.dataframe(df, use_container_width=True)
            else:
                st.warning("No sources found for this query.")

            # Optionally store table_data in session_state for reuse
            st.session_state.table_data = table_data

            # Add to chat history
            st.session_state.chat_history.append(("Assistant", table_data))
