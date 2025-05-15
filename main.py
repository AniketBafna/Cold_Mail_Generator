# ------------------ Page Config & Custom Styling ------------------ #
import streamlit as st

st.set_page_config(
    page_title="ColdMail Generator",
    page_icon="📬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to right, #667eea, #764ba2);
    color: #ffffff;
}
[data-testid="stHeader"], [data-testid="stToolbar"] {
    background: rgba(0,0,0,0);
    visibility: hidden;
}
h1, h2, h3 {
    color: #ffffff;
    text-align: center;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.stButton>button {
    background-color: #6a11cb;
    color: white;
    border-radius: 10px;
    font-size: 18px;
    padding: 10px 24px;
    font-weight: bold;
    border: none;
}
.stButton>button:hover {
    background-color: #8e2de2;
}
textarea, input {
    border-radius: 8px !important;
}
hr {
    border-top: 1px solid #e0e0e0;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# main.py
import streamlit as st
from streamlit_lottie import st_lottie
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import requests
import os
import re
import json
#import chromadb
from helpers import (
    extract_resume_text,
    extract_job_description,
    extract_keywords,
    ats_score,
    load_lottie_url,
    semantic_similarity_score,
    load_embedder  # For embeddings; already cached
)

# ------------------ Environment & LLM Setup ------------------ #
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    temperature=0.4,
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile"
)

# ------------------ Lottie Animation ------------------ #
lottie_loading = load_lottie_url("https://assets9.lottiefiles.com/packages/lf20_usmfx6bp.json")

# ------------------ Session State ------------------ #
if "email_history" not in st.session_state:
    st.session_state["email_history"] = []

# ------------------ Sidebar Settings ------------------ #
with st.sidebar:
    st.header("⚙️ Generation Settings")
    tone = st.selectbox("Tone", ["Professional", "Friendly", "Persuasive", "Formal"], index=0)
    word_limit = st.slider("Word Limit", 100, 400, 200, step=50)
    st.session_state["tone"] = tone
    st.session_state["word_limit"] = word_limit

# ------------------ App Title & Description ------------------ #
st.markdown("<h1 style='text-align: center;'>📬 ColdMail Generator</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Generate Cold Email or Cover Letter from Resume + Job Link</h3>", unsafe_allow_html=True)
st.markdown("---")
st.header("How It Works")
st.markdown("""
    - 📄 Upload your Resume (PDF)
    - 🔗 Paste a Job Posting URL
    - ✍️ Add optional details
    - 🚀 Get a professional cold email or cover letter
""")
st.header("🎯 Start Here")

# ------------------ Form UI ------------------ #
with st.form(key="form"):
    uploaded_resume = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    job_link = st.text_input("Paste Job Posting URL")
    col1, col2 = st.columns(2)
    with col1:
        recipient_name = st.text_input("Recipient's Name")
    with col2:
        recipient_position = st.text_input("Recipient's Position")
    company_name = st.text_input("Company Name")
    custom_message = st.text_area("Custom Message (Optional)")
    st.subheader("🔗 Your Links")
    linkedin_link = st.text_input("LinkedIn")
    github_link = st.text_input("GitHub")
    portfolio_link = st.text_input("Portfolio")
    generate_cover_letter = st.toggle("📄 Generate Cover Letter Instead of Cold Email")
    submit_button = st.form_submit_button("✨ Generate ✨")

# ------------------ Processing on Form Submit ------------------ #
if submit_button:
    if uploaded_resume is None or not job_link:
        st.error("⚠️ Resume and Job Link are required.")
    else:
        with st.spinner("Analyzing and generating..."):
            if lottie_loading:
                st_lottie(lottie_loading, height=200)

            # Extract text and job details
            resume_text = extract_resume_text(uploaded_resume)
            job_description = extract_job_description(job_link)
            resume_keywords = extract_keywords(resume_text)
            job_keywords = extract_keywords(job_description)
            resume_words = set([word for word, _ in resume_keywords])
            job_words = set([word for word, _ in job_keywords])
            matching_keywords = resume_words.intersection(job_words)

            # ATS & Semantic Similarity Scores
            col1, col2, col3 = st.columns(3)
            with col1:
                ats = ats_score(resume_text, job_description)
                st.metric("📊 ATS Match Score", f"{ats}%")
            with col2:
                semantic_score = semantic_similarity_score(resume_text, job_description)
                st.metric("🧠 Semantic Match Score", f"{semantic_score}%")
            with col3:
                final_score = round((ats * 0.4) + (semantic_score * 0.6), 2)
                st.metric("🎯 Combined Match Score", f"{final_score}%")

            prompt_type = "cover letter" if generate_cover_letter else "cold email"
            system_prompt = f"""
You are a professional job application writer. Write a {prompt_type} using:
- Resume:\n{resume_text}
- Job Description:\n{job_description}
- Match Keywords: {', '.join(matching_keywords)}
- Custom Message: {custom_message or 'None'}
- LinkedIn: {linkedin_link}, GitHub: {github_link}, Portfolio: {portfolio_link}
- Recipient: {recipient_name} ({recipient_position}) at {company_name}
Limit to {st.session_state['word_limit']} words. Tone: {st.session_state['tone'].lower()} and compelling.
"""
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", f"Generate the {prompt_type} now.")
            ])
            chain = prompt | llm
            response = chain.invoke({}).content

            # Save generation to session state history
            st.session_state["email_history"].append({
                "type": prompt_type,
                "output": response,
                "ats_score": ats
            })

            st.text_area("📝 Your Generated Message", value=response, height=350)
            st.download_button("📥 Download", data=response, file_name=f"{prompt_type}.txt", mime="text/plain")

# ------------------ Display History ------------------ #
if st.session_state["email_history"]:
    st.markdown("## 🧠 History")
    st.markdown("<hr style='margin-top: -10px; margin-bottom: 10px;'>", unsafe_allow_html=True)
    for idx, entry in enumerate(reversed(st.session_state["email_history"])):
        with st.expander(f"📝 {entry['type'].title()} - ATS Score: {entry['ats_score']}%", expanded=False):
            st.code(entry["output"])
    st.markdown("---")
    clear_col1, clear_col2, clear_col3 = st.columns([1, 2, 1])
    with clear_col2:
        if st.button("🧹 Clear History", use_container_width=True):
            st.session_state["email_history"] = []
            st.success("🧼 Email history cleared successfully!")

st.markdown("---")
st.markdown(
    """
    <center><small>
    ✨ Built with ❤️ by <a href="https://www.linkedin.com/in/aniket-bafna/" target="_blank">
    <span style='color:#ffd700; font-weight:bold;'>Aniket Bafna</span></a>
    </small></center>
    """, unsafe_allow_html=True)
