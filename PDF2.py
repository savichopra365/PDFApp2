import os
import fitz  # PyMuPDF
import shutil
import requests
from pathlib import Path
from PIL import Image
import pytesseract
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import streamlit as st
from pdf2image import convert_from_bytes


# For Mongo DB connection
from pymongo import MongoClient
import datetime

# MongoDB Atlas Connection
MONGO_URI = "mongodb+srv://choprasa:Savi3650@cluster1.f2mxsnf.mongodb.net/?retryWrites=true&w=majority&appName=Cluster1"  # <-- Replace this with your URI
client = MongoClient(MONGO_URI)
db = client["document_qa"]  # database name
collection = db["documents"]  # collection name

# for testing if documents are going in the mongodb"
'''for doc in collection.find().limit(5):
    print(doc)'''


# Setup API Key and LLM
os.environ["GOOGLE_API_KEY"] ="AIzaSyBNWsJZFIpvaldltYnK3j26OmzsakeUgy0" # Replace with your key
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.environ["GOOGLE_API_KEY"])

# Custom Prompt Template
prompt_template = PromptTemplate.from_template("""
You are an intelligent assistant designed to answer questions based on the content of uploaded PDF and image documents.
Please provide accurate and helpful answers to the questions asked, using the context provided from the documents.

Context:
{context}

Question:
{question}

Helpful Answer:
""")
prompt = PromptTemplate(template=prompt_template.template, input_variables=["history", "context", "question"])

# Streamlit UI
st.title("Multi-File Document QA (PDF + Images)")
uploaded_files = st.file_uploader("Upload PDFs or Images", type=["pdf", "jpg", "jpeg", "png"], accept_multiple_files=True)
query = st.text_input("Enter your query")

# OCR logic for images
def extract_text_from_image(file):
    image = Image.open(file).convert("RGB")
    return pytesseract.image_to_string(image)

# Text extraction for PDFs
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

# Handle all uploaded files
def load_documents(files):
    texts = []
    for file in files:
        suffix = Path(file.name).suffix.lower()
        if suffix in [".jpg", ".jpeg", ".png"]:
            text = extract_text_from_image(file)
            doc_type = "image"
        elif suffix == ".pdf":
            text = extract_text_from_pdf(file)
            doc_type = "pdf"
        else:
            continue

        if text.strip():
            # Store in MongoDB
            document_record = {
                "file_name": file.name,
                "file_type": doc_type,
                "text": text,
                "timestamp": datetime.datetime.utcnow()
            }
            collection.insert_one(document_record)
            texts.append(text)
    return texts


# Split text into chunks
def split_text(texts):
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=64)
    return splitter.create_documents(texts)

# Store vector DB and build QA chain
def setup_qa_chain(text_chunks):
    db_path = Path("./vectorstores/db_faiss")
    if db_path.exists():
        shutil.rmtree(db_path)

    vectorstore = FAISS.from_documents(text_chunks, embedding_model)
    vectorstore.save_local(str(db_path))

    memory = ConversationBufferMemory(memory_key="history", input_key="question")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    qa_chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "verbose": True,
            "prompt": prompt,
            "memory": memory,
        },
    )
    return qa_chain

# QA logic
def answer_query(qa_chain, query):
    result = qa_chain({"query": query})
    return result['result']
# Streamlit PREVIEW

if uploaded_files:
    st.subheader("📄 Uploaded File Previews")

    for file in uploaded_files:
        file_suffix = Path(file.name).suffix.lower()
        st.markdown(f"**File:** {file.name}")
        columns = st.columns(4)  # 4 previews per row
    for idx, file in enumerate(uploaded_files):
        file_suffix = Path(file.name).suffix.lower()
        col = columns[idx % 4]  # Wrap every 4 files into a new row

        with col:
            st.markdown(f"**{file.name}**")
        if file_suffix in [".pdf"]:
            # Preview for PDFs
            try:
                # Read file into memory buffer
                pdf_bytes = file.read()
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                page = doc.load_page(0)  # Load first page
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # High-res render
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                st.image(img, caption=f"Preview of {file.name} (Page 1)", use_column_width=True)
            except Exception as e:
                st.error(f"Could not preview PDF: {e}")

        elif file_suffix in [".jpg", ".jpeg", ".png"]:
            # Show image and preview OCR
            try:
                img = Image.open(file)
                st.image(img, caption=file.name, width=250)
                ocr_text = extract_text_from_image(file)
                st.text_area("Image OCR Preview", ocr_text[:500] + "...", height=150)
            except Exception as e:
                st.error(f"Could not preview image: {e}")

        st.markdown("---")



# MAIN LOGIC
if st.button("Get Answer"):
    if uploaded_files and query:
        with st.spinner("Processing files and searching for answers..."):
            raw_texts = load_documents(uploaded_files)

            if not raw_texts:
                st.error("No text could be extracted from the uploaded files.")
            else:
                text_chunks = split_text(raw_texts)
                qa_chain = setup_qa_chain(text_chunks)
                answer = answer_query(qa_chain, query)
                st.success("Answer:")
                st.write(answer)
    else:
        st.warning("Please upload at least one file and enter a query.")

