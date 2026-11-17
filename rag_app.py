"""
Multi-PDF RAG App with Sources - Social Eagle Week 2 Day 4.

Upload a folder's worth of PDFs; they are chunked, embedded and stored in ONE
FAISS vector database. Every answer lists its sources: file name + page number,
carried through from the chunk metadata.

Run:
    export OPENAI_API_KEY="your-key"
    streamlit run rag_app.py
"""

import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="Multi-PDF RAG", layout="wide")
st.title("Ask across your PDFs")

if not os.environ.get("OPENAI_API_KEY"):
    st.error("Set the OPENAI_API_KEY environment variable first.")
    st.stop()

files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if files and "store" not in st.session_state:
    chunks = []
    for f in files:
        # PyPDFLoader needs a path, so park each upload in a temp file.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.read())
            tmp_path = tmp.name
        docs = PyPDFLoader(tmp_path).load()
        for d in docs:
            d.metadata["source"] = f.name  # keep the real file name, page stays from the loader
        chunks.extend(
            RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=150
            ).split_documents(docs)
        )
    st.session_state.store = FAISS.from_documents(chunks, OpenAIEmbeddings())
    st.success(f"Indexed {len(chunks)} chunks from {len(files)} file(s)")

question = st.text_input("Your question across all uploaded PDFs")

if question and "store" in st.session_state:
    docs = st.session_state.store.as_retriever(search_kwargs={"k": 4}).invoke(question)

    context = "\n\n".join(d.page_content for d in docs)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    answer = llm.invoke(
        "Answer only from the context below. If the documents disagree, say so "
        "and show each side. If the answer is not in the context, say you cannot "
        f"find it in the uploaded documents.\n\nContext:\n{context}\n\n"
        f"Question: {question}"
    )

    st.markdown("### Answer")
    st.write(answer.content)

    st.markdown("### Sources")
    seen = set()
    for d in docs:
        src = f"{d.metadata.get('source', '?')} - page {d.metadata.get('page', '?')}"
        if src not in seen:
            seen.add(src)
            st.caption(src)
elif question:
    st.info("Upload at least one PDF first.")
