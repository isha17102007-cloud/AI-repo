import streamlit as st
import chromadb
import google.generativeai as genai
import pdfplumber

st.set_page_config(page_title="AI Teacher", page_icon="📘")

# ---------- SETUP ----------
# Your Gemini API key is read from Streamlit's "Secrets" (set this up in
# Streamlit Cloud settings — never paste the key directly into this file).
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# In-memory vector store (resets each time the app restarts — fine for a
# small project; a persistent DB can be added later).
chroma_client = chromadb.Client()
if "collection" not in st.session_state:
    st.session_state.collection = chroma_client.create_collection("ncert")
    st.session_state.ingested_files = []  # tracks what's been added so far


# ---------- HELPERS ----------
def chunk_text(text, size=400):
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


def ingest_pdf(uploaded_file, grade, subject, chapter_name):
    with pdfplumber.open(uploaded_file) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    chunks = chunk_text(text)

    # Unique ID prefix per file so chunks from different chapters never collide
    file_key = f"{grade}_{subject}_{chapter_name}_{uploaded_file.name}".replace(" ", "_")
    ids = [f"{file_key}_{i}" for i in range(len(chunks))]
    metadatas = [
        {"grade": grade, "subject": subject, "chapter": chapter_name}
        for _ in chunks
    ]

    st.session_state.collection.add(
        documents=chunks,
        ids=ids,
        metadatas=metadatas,
    )
    st.session_state.ingested_files.append(
        {"file": uploaded_file.name, "grade": grade, "subject": subject, "chapter": chapter_name, "chunks": len(chunks)}
    )
    return len(chunks)


def retrieve(query, grade, subject, k=3):
    results = st.session_state.collection.query(
        query_texts=[query],
        n_results=k,
        where={"$and": [{"grade": grade}, {"subject": subject}]},
    )
    docs = results["documents"][0]
    if not docs:
        return None
    return "\n\n".join(docs)


PROMPTS = {
    "Explain": "Explain the concept '{q}' simply, for a class {grade} student, using ONLY the material below. Keep it under 200 words.\n\nMATERIAL:\n{ctx}",
    "Practice Questions": "Using ONLY the material below, write 5 practice questions on '{q}' for a class {grade} student (2 MCQs with answers marked, 2 short-answer, 1 long-answer), plus an answer key.\n\nMATERIAL:\n{ctx}",
    "Revision Summary": "Using ONLY the material below, write a short bullet-point revision summary of '{q}' for a class {grade} student, bolding key terms, under 150 words.\n\nMATERIAL:\n{ctx}",
    "Chapter Q&A": "Answer this student question using ONLY the material below, step by step. If the material doesn't contain the answer, say so honestly.\n\nQUESTION: {q}\n\nMATERIAL:\n{ctx}",
}


def ask(feature, question, grade, subject):
    ctx = retrieve(question, grade, subject)
    if ctx is None:
        return None, None
    prompt = PROMPTS[feature].format(q=question, grade=grade, ctx=ctx)
    response = gemini_model.generate_content(prompt)
    return response.text, ctx


# ---------- UI ----------
st.title("📘 AI Teacher")
st.caption("Grounded in your own NCERT chapters, notes, and worksheets.")

SUBJECTS = ["Science", "Social Science", "Mathematics", "English", "Hindi"]
GRADES = ["6", "7", "8", "9", "10"]

with st.sidebar:
    st.header("1. Upload chapters")
    st.caption("Upload as many chapters as you like — tag each one so retrieval stays accurate.")

    up_grade = st.selectbox("This chapter's class", GRADES, index=2, key="up_grade")
    up_subject = st.selectbox("This chapter's subject", SUBJECTS, key="up_subject")
    up_chapter_name = st.text_input("Chapter name/title", placeholder="e.g. The Invisible Living World")
    uploaded_files = st.file_uploader(
        "Upload PDF(s) for this class + subject", type="pdf", accept_multiple_files=True
    )

    if uploaded_files and st.button("Ingest uploaded file(s)"):
        if not up_chapter_name.strip():
            st.warning("Give the chapter a name first.")
        else:
            total = 0
            with st.spinner("Reading and indexing..."):
                for f in uploaded_files:
                    total += ingest_pdf(f, up_grade, up_subject, up_chapter_name.strip())
            st.success(f"Indexed {total} chunks across {len(uploaded_files)} file(s).")

    if st.session_state.ingested_files:
        st.divider()
        st.caption("Chapters ingested so far:")
        for item in st.session_state.ingested_files:
            st.write(f"• Class {item['grade']} · {item['subject']} · {item['chapter']} ({item['chunks']} chunks)")

st.header("2. Ask your AI teacher")

col1, col2 = st.columns(2)
with col1:
    ask_grade = st.selectbox("Class", GRADES, index=2, key="ask_grade")
with col2:
    ask_subject = st.selectbox("Subject", SUBJECTS, key="ask_subject")

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
    if not st.session_state.ingested_files:
        st.warning("Upload and ingest at least one chapter first (left sidebar).")
    elif not question.strip():
        st.warning("Type a topic or question first.")
    else:
        with st.spinner("Thinking..."):
            answer, source = ask(feature, question, ask_grade, ask_subject)
        if answer is None:
            st.warning(f"No ingested chapter matches Class {ask_grade} · {ask_subject}. Check your selection or upload that chapter.")
        else:
            st.markdown(answer)
            with st.expander("Source material used"):
                st.write(source)

