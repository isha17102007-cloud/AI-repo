import streamlit as st
import chromadb
import google.generativeai as genai
import pdfplumber

st.set_page_config(page_title="AI Teacher", page_icon="📘")

# ---------- SETUP ----------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# PERSISTENT vector store — saved to disk, so chapters stay ingested even
# after you close the app or someone else opens it. You only ingest each
# chapter ONCE, ever (unless the app is redeployed/reset by Streamlit Cloud).
chroma_client = chromadb.PersistentClient(path="chroma_store")
collection = chroma_client.get_or_create_collection("ncert")


def list_ingested():
    """Rebuild the list of ingested chapters from what's actually stored,
    so it survives page refreshes and new visitors — not just this session."""
    data = collection.get(include=["metadatas"])
    seen = {}
    for m in data["metadatas"]:
        key = (m["grade"], m["subject"], m["chapter"])
        seen[key] = seen.get(key, 0) + 1
    return [
        {"grade": g, "subject": s, "chapter": c, "chunks": n}
        for (g, s, c), n in sorted(seen.items())
    ]


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

    file_key = f"{grade}_{subject}_{chapter_name}_{uploaded_file.name}".replace(" ", "_")
    ids = [f"{file_key}_{i}" for i in range(len(chunks))]
    metadatas = [{"grade": grade, "subject": subject, "chapter": chapter_name} for _ in chunks]

    collection.add(documents=chunks, ids=ids, metadatas=metadatas)
    return len(chunks)


def retrieve(query, grade, subject, k=6):
    results = collection.query(
        query_texts=[query],
        n_results=k,
        where={"$and": [{"grade": grade}, {"subject": subject}]},
    )
    docs = results["documents"][0]
    if not docs:
        return None
    return "\n\n---\n\n".join(docs)


PROMPTS = {
    "Explain": "Explain the concept '{q}' simply, for a class {grade} student, using ONLY the material below. Cover it thoroughly but clearly — don't skip relevant details found in the material. \n\nMATERIAL:\n{ctx}",
    "Practice Questions": "Using ONLY the material below, write 5 practice questions on '{q}' for a class {grade} student (2 MCQs with answers marked, 2 short-answer, 1 long-answer), plus an answer key.\n\nMATERIAL:\n{ctx}",
    "Revision Summary": "Using ONLY the material below, write a clear, complete bullet-point revision summary of '{q}' for a class {grade} student, bolding key terms. Include every important point found in the material.\n\nMATERIAL:\n{ctx}",
    "Chapter Q&A": "Answer this student question using ONLY the material below, step by step and in full detail. If the material doesn't contain the answer, say so honestly.\n\nQUESTION: {q}\n\nMATERIAL:\n{ctx}",
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
    st.caption("Each chapter only needs to be ingested once — it's saved permanently.")

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
            st.success(f"Indexed {total} chunks across {len(uploaded_files)} file(s). Saved permanently.")

    ingested = list_ingested()
    if ingested:
        st.divider()
        st.caption(f"Already ingested ({len(ingested)} chapters — permanent, no need to re-upload):")
        for item in ingested:
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
    if not list_ingested():
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
            with st.expander("Full source material used (all retrieved chunks)"):
                st.write(source)
