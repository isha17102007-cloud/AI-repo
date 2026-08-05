import streamlit as st
import chromadb
import google.generativeai as genai
import pdfplumber

st.set_page_config(page_title="AI Teacher", page_icon="📘")

# ---------- SETUP ----------
# Your Gemini API key is read from Streamlit's "Secrets" (set this up in
# Streamlit Cloud settings — never paste the key directly into this file).
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# In-memory vector store (resets each time the app restarts — fine for a
# small project; a persistent DB can be added later).
chroma_client = chromadb.Client()
if "collection" not in st.session_state:
    st.session_state.collection = chroma_client.create_collection("ncert")
    st.session_state.ingested = False


# ---------- HELPERS ----------
def chunk_text(text, size=400):
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


def ingest_pdf(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    chunks = chunk_text(text)
    st.session_state.collection.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))],
    )
    st.session_state.ingested = True
    return len(chunks)


def retrieve(query, k=3):
    results = st.session_state.collection.query(query_texts=[query], n_results=k)
    return "\n\n".join(results["documents"][0])


PROMPTS = {
    "Explain": "Explain the concept '{q}' simply, for a class {grade} student, using ONLY the material below. Keep it under 200 words.\n\nMATERIAL:\n{ctx}",
    "Practice Questions": "Using ONLY the material below, write 5 practice questions on '{q}' for a class {grade} student (2 MCQs with answers marked, 2 short-answer, 1 long-answer), plus an answer key.\n\nMATERIAL:\n{ctx}",
    "Revision Summary": "Using ONLY the material below, write a short bullet-point revision summary of '{q}' for a class {grade} student, bolding key terms, under 150 words.\n\nMATERIAL:\n{ctx}",
    "Chapter Q&A": "Answer this student question using ONLY the material below, step by step. If the material doesn't contain the answer, say so honestly.\n\nQUESTION: {q}\n\nMATERIAL:\n{ctx}",
}


def ask(feature, question, grade):
    ctx = retrieve(question)
    prompt = PROMPTS[feature].format(q=question, grade=grade, ctx=ctx)
    response = gemini_model.generate_content(prompt)
    return response.text, ctx


# ---------- UI ----------
st.title("📘 AI Teacher")
st.caption("Grounded in your own NCERT chapters, notes, and worksheets.")

with st.sidebar:
    st.header("1. Upload source material")
    uploaded = st.file_uploader("Upload a chapter PDF", type="pdf")
    if uploaded and st.button("Ingest this file"):
        with st.spinner("Reading and indexing..."):
            n = ingest_pdf(uploaded)
        st.success(f"Indexed {n} chunks from this chapter.")

    grade = st.selectbox("Class", ["6", "7", "8", "9", "10"], index=2)

st.header("2. Ask your AI teacher")
feature = st.radio(
    "What do you need?",
    ["Explain", "Practice Questions", "Revision Summary", "Chapter Q&A"],
    horizontal=True,
)
question = st.text_input(
    "Topic or question",
    placeholder="e.g. Photosynthesis" if feature != "Chapter Q&A" else "Paste the exercise question here",
)

if st.button("Ask", type="primary"):
    if not st.session_state.get("ingested"):
        st.warning("Upload and ingest a chapter PDF first (left sidebar).")
    elif not question.strip():
        st.warning("Type a topic or question first.")
    else:
        with st.spinner("Thinking..."):
            answer, source = ask(feature, question, grade)
        st.markdown(answer)
        with st.expander("Source material used"):
            st.write(source) 