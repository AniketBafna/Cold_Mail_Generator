import re
import requests
import pdfplumber
import json
import os
from collections import Counter
from bs4 import BeautifulSoup
import streamlit as st
from sentence_transformers import SentenceTransformer, util

# ------------------ Resume Parsing ------------------ #
def extract_resume_text(uploaded_file):
    if uploaded_file is not None:
        try:
            with pdfplumber.open(uploaded_file) as pdf:
                pages = [page.extract_text() or '' for page in pdf.pages]
            text = "\n".join(pages)
            text = re.sub(r'\n+', '\n', text)
            return text.strip()
        except Exception:
            st.error("❌ Unable to parse your resume. Make sure it's a readable PDF.")
    return ""

# ------------------ Job Description Extraction ------------------ #
def extract_job_description(url):
    try:
        if not url.startswith("http"):
            raise ValueError("URL must start with http/https")
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        content = " ".join([tag.get_text() for tag in soup.find_all(['p', 'div'])])
        return content.strip()[:3000] if content else "No text found."
    except Exception as e:
        st.error(f"⚠️ Error fetching job posting: {e}")
        return ""

# ------------------ Keyword Extraction ------------------ #
def extract_keywords(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.lower().split()
    stopwords = set([
        'the', 'and', 'for', 'with', 'you', 'your', 'are', 'our', 'have',
        'has', 'will', 'this', 'that', 'from', 'but', 'they', 'their',
        'them', 'about', 'into', 'who', 'what', 'when', 'where', 'which',
        'how', 'can', 'also', 'etc', 'should', 'must', 'may', 'could', 'would'
    ])
    words = [w for w in words if w not in stopwords and len(w) > 2]
    return Counter(words).most_common(50)

# ------------------ ATS Score Calculation ------------------ #
def ats_score(resume_text, job_description):
    resume_words = set(resume_text.lower().split())
    job_words = set(job_description.lower().split())
    common = resume_words.intersection(job_words)
    return round(len(common) / len(job_words) * 100, 2) if job_words else 0

# ------------------ Lottie Loader ------------------ #
def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to load Lottie animation: {e}")
        return None

# ------------------ Embedding and Semantic Similarity ------------------ #
@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedder = load_embedder()

def semantic_similarity_score(resume_text, job_description):
    resume_embedding = embedder.encode(resume_text, convert_to_tensor=True)
    job_embedding = embedder.encode(job_description, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(resume_embedding, job_embedding).item()
    return round(similarity * 100, 2)
