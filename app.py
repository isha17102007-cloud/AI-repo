import streamlit as st
import chromadb
import google.generativeai as genai
import pdfplumber

st.set_page_config(page_title="AI Teacher", page_icon="📘")

# setup gemini api
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# setup database to store the chapters
# this is saved to disk (not memory) so once a chapter is ingested
# it stays saved even after closing the app or restarting the phone
chroma_client = chromadb.PersistentClient(path="chroma_store")
collection = chroma_client.get_or_create_collection("ncert")


def list_ingested():
    # this goes through everything stored and builds a list of
    # which chapters have already been uploaded
    data = collection.get(include=["metadatas"])
    seen = {}
    for m in data["metadatas"]:
        key = (m["grade"], m["subject"], m["chapter"])
        if key in seen:
            seen[key] = seen[key] + 1
        else:
            seen[key] = 1

    result = []
    for key in seen:
        grade = key[0]
        subject = key[1]
        chapter = key[2]
        count = seen[key]
        result.append({"grade": grade, "subject": subject, "chapter": chapter, "chunks": count})

    result.sort(key=lambda x: (x["grade"], x["subject"], x["chapter"]))
    return result


def chunk_text(text, size=400):
    # breaks the text into smaller pieces of about 400 words each
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        piece = words[i:i + size]
        piece_text = " ".join(piece)
        chunks.append(piece_text)
        i = i + size
    return chunks


def read_pdf_text(uploaded_file):
    # opens the pdf and pulls out all the text page by page
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text = text + page_text + "\n"
    return text


def ingest_files(uploaded_files, grade, subject, chapter_name):
    # reads every uploaded file first and builds one big list of
    # chunks/ids/metadata, then saves everything in a single call
    # so the whole batch is either fully saved or not saved at all
    all_documents = []
    all_ids = []
    all_metadatas = []

    for uploaded_file in uploaded_files:
        text = read_pdf_text(uploaded_file)
        chunks = chunk_text(text)

        file_key = grade + "_" + subject + "_" + chapter_name + "_" + uploaded_file.name
        file_key = file_key.replace(" ", "_")

        for i in range(len(chunks)):
            all_documents.append(chunks[i])
            all_ids.append(file_key + "_" + str(i))
            all_metadatas.append({"grade": grade, "subject": subject, "chapter": chapter_name})

    collection.add(documents=all_documents, ids=all_ids, metadatas=all_metadatas)
    return len(all_documents)


def retrieve(query, grade, subject, k=6):
    # searches the database for the most relevant chunks matching
    # the class and subject that was selected
    results = collection.query(
        query_texts=[query],
        n_results=k,
        where={"$and": [{"grade": grade}, {"subject": subject}]},
    )
    docs = results["documents"][0]
    if len(docs) == 0:
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

    prompt_template = PROMPTS[feature]
    prompt = prompt_template.format(q=question, grade=grade, ctx=ctx)

    response = gemini_model.generate_content(prompt)
    return response.text, ctx


# ---- main app starts here ----

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
        if up_chapter_name.strip() == "":
            st.warning("Give the chapter a name first.")
        else:
            with st.spinner("Reading and indexing..."):
                total = ingest_files(uploaded_files, up_grade, up_subject, up_chapter_name.strip())
            st.success("Indexed " + str(total) + " chunks across " + str(len(uploaded_files)) + " file(s), saved all at once.")

    ingested = list_ingested()
    if len(ingested) > 0:
        st.divider()
        st.caption("Already ingested (" + str(len(ingested)) + " chapters — permanent, no need to re-upload):")
        for item in ingested:
            line = "• Class " + item["grade"] + " · " + item["subject"] + " · " + item["chapter"] + " (" + str(item["chunks"]) + " chunks)"
            st.write(line)

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

if feature == "Chapter Q&A":
    question = st.text_input("Topic or question", placeholder="Paste the exercise question here")
else:
    question = st.text_input("Topic or question", placeholder="e.g. Photosynthesis")

if st.button("Ask", type="primary"):
    if len(list_ingested()) == 0:
        st.warning("Upload and ingest at least one chapter first (left sidebar).")
    elif question.strip() == "":
        st.warning("Type a topic or question first.")
    else:
        with st.spinner("Thinking..."):
            answer, source = ask(feature, question, ask_grade, ask_subject)
        if answer is None:
            st.warning("No ingested chapter matches Class " + ask_grade + " · " + ask_subject + ". Check your selection or upload that chapter.")
        else:
            st.markdown(answer)
            with st.expander("Full source material used (all retrieved chunks)"):
                st.write(source)
