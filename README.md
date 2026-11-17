# Multi-PDF RAG App with Sources - Social Eagle Week 2 Day 4

A LangChain + Streamlit app that answers questions across several PDFs at
once. Every answer shows its sources - file name and page number - pulled
from the chunk metadata, so the reader can go and check.

## Setup and run

```bash
mkdir multi-rag && cd multi-rag
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\Activate.ps1
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"                  # Windows: $env:OPENAI_API_KEY="your-key"
streamlit run rag_app.py
```

## How it works

1. Upload multiple PDFs in one go. They are chunked (1000 chars, 150 overlap)
   and embedded into a single FAISS store kept in `st.session_state`, so it
   survives Streamlit's reruns.
2. The app reports how many chunks were indexed.
3. A question retrieves the top chunks across every document; the model answers
   only from them, and the source caption under the answer lists
   `file.pdf - page N` for each chunk used.
4. The prompt tells the model to say "these two disagree" instead of quietly
   picking one side, and to admit when the answer is not in the documents.

## Free-tier note

OpenAI embeddings are the paid piece. For a fully free run, swap
`OpenAIEmbeddings()` for
`HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")`
(`pip install langchain-huggingface sentence-transformers`) and point the chat
model at Groq's free endpoint with
`ChatOpenAI(model="llama-3.3-70b-versatile", base_url="https://api.groq.com/openai/v1", api_key=os.environ["GROQ_API_KEY"])`.
The rest of the app is unchanged.
