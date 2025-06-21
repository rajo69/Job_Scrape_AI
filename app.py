# app.py

import streamlit as st
import requests
from bs4 import BeautifulSoup
import time
import random
import pandas as pd
import re
import spacy
from rake_nltk import Rake
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import os
import nltk

# --- App Configuration ---
st.set_page_config(page_title="LinkedIn AI Job Analyzer", layout="wide")

# --- Self-Contained NLTK Data Setup ---
def ensure_nltk_data():
    if 'nltk_data_checked' in st.session_state: return
    st.info("First-time setup: Checking for required NLP data...")
    try:
        nltk.data.find('tokenizers/punkt_tab')
        st.info("✅ 'punkt_tab' dataset found.")
    except LookupError:
        st.info("Downloading 'punkt_tab' dataset for sentence tokenization...")
        nltk.download('punkt_tab')
    try:
        nltk.data.find('corpora/stopwords')
        st.info("✅ 'stopwords' dataset found.")
    except LookupError:
        st.info("Downloading 'stopwords' dataset...")
        nltk.download('stopwords')
    st.success("NLP data is ready!")
    time.sleep(2)
    st.session_state['nltk_data_checked'] = True


# --- Caching for NLP Models and Functions ---
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

@st.cache_data
def parse_time_posted(time_text):
    if not time_text: return None
    num_match = re.search(r'\d+', time_text)
    if not num_match: return None
    num = int(num_match.group(0))
    time_text = time_text.lower()
    if 'second' in time_text or 'minute' in time_text or 'hour' in time_text: return 0
    elif 'day' in time_text: return num * 24
    elif 'week' in time_text: return num * 7 * 24
    elif 'month' in time_text: return num * 30 * 24
    else: return None

@st.cache_data
def parse_num_applicants(applicant_text):
    if not applicant_text: return None
    num_match = re.search(r'\d+', applicant_text)
    if num_match: return int(num_match.group(0))
    else: return None

@st.cache_data
def extract_keywords(_text):
    if not isinstance(_text, str) or not _text.strip(): return []
    r = Rake()
    r.extract_keywords_from_text(_text)
    return r.get_ranked_phrases()[:5]

@st.cache_data
def extract_skills(_text, _nlp_model):
    if not isinstance(_text, str) or not _text.strip(): return []
    doc = _nlp_model(_text)
    return list(set([ent.text for ent in doc.ents if ent.label_ in ('ORG', 'PRODUCT')]))

@st.cache_data
def analyze_sentiment(_text):
    if not isinstance(_text, str) or not _text.strip(): return 0.0
    return TextBlob(_text).sentiment.polarity

@st.cache_data
def calculate_match(job_desc, resume_text):
    if not all(isinstance(t, str) and t.strip() for t in [job_desc, resume_text]): return 0.0
    vectorizer = TfidfVectorizer()
    try:
        vectors = vectorizer.fit_transform([job_desc, resume_text])
        return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    except ValueError: return 0.0

def read_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages: text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return ""

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')


# --- Core Scraping and Processing Functions ---
def run_linkedin_scraper(title, location, num_pages):
    # ... (Code to get job IDs remains the same) ...
    id_list, start, count = [], 0, 0
    status_text = st.empty()
    progress_bar = st.progress(0, text="Scraping progress")
    status_text.text("Part 1/2: Collecting Job IDs...")
    while count < num_pages:
        list_url = f"https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search?keywords={title}&location={location}&start={start}"
        response = requests.get(list_url)
        if response.status_code != 200: break
        list_soup = BeautifulSoup(response.text, "html.parser")
        page_jobs = list_soup.find_all("li")
        if not page_jobs: break
        for job in page_jobs:
            job_id = job.find("div", {"class": "base-card"}).get("data-entity-urn", "").split(":")[-1]
            if job_id and job_id not in id_list: id_list.append(job_id)
        count += 1
        start += 25
        status_text.text(f"Part 1/2: Collecting Job IDs... Found {len(id_list)} IDs across {count} page(s).")
        progress_bar.progress(count / num_pages)
        time.sleep(random.uniform(1, 2))
    if not id_list: return None
    
    # --- Scraping loop ---
    job_list, total_ids = [], len(id_list)
    status_text.text(f"Part 2/2: Scraping details for {total_ids} jobs...")
    progress_bar.progress(0)
    for i, job_id in enumerate(id_list):
        job_url = f"https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}"
        job_response = requests.get(job_url)
        if job_response.status_code == 200:
            job_soup = BeautifulSoup(job_response.text, "html.parser")
            
            time_posted_raw = None
            num_applicants_raw = None
            salary_raw = None # <<< ADDED: Initialize salary variable
            
            try:
                time_posted_raw = job_soup.find("span", {"class": "posted-time-ago__text"}).text.strip()
            except: pass
            try:
                num_applicants_raw = job_soup.find("span", {"class": "num-applicants__caption"}).text.strip()
            except: pass
            
            # <<< ADDED: Robust salary extraction logic >>>
            try:
                # Salary is often in a list item with other "insights"
                insights = job_soup.find_all("li", {"class": "job-details-jobs-unified-top-card__job-insight"})
                for insight in insights:
                    # The salary insight is reliably identified by the money bag emoji
                    if "💰" in insight.get_text():
                        salary_raw = insight.find("span").text.strip()
                        break # Found it, no need to continue the loop
            except:
                # If any part of this fails, salary_raw will remain None
                pass
            
            job_post = {
                'job_link': f"https://www.linkedin.com/jobs/view/{job_id}",
                'job_title': job_soup.find("h2", {"class":"top-card-layout__title"}).text.strip() if job_soup.find("h2", {"class":"top-card-layout__title"}) else "N/A",
                'company_name': job_soup.find("a", {"class": "topcard__org-name-link"}).text.strip() if job_soup.find("a", {"class": "topcard__org-name-link"}) else "N/A",
                'salary': salary_raw, # <<< ADDED: Add salary to the dictionary
                'job_desc': job_soup.find("div", {"class": "show-more-less-html__markup"}).get_text(separator="\n").strip() if job_soup.find("div", {"class": "show-more-less-html__markup"}) else "",
                'hours_posted': parse_time_posted(time_posted_raw),
                'applicants_count': parse_num_applicants(num_applicants_raw)
            }
            job_list.append(job_post)
            
        status_text.text(f"Part 2/2: Scraping details... Job {i+1}/{total_ids}")
        progress_bar.progress((i + 1) / total_ids)
        time.sleep(random.uniform(0.5, 1.5))
        
    status_text.empty()
    progress_bar.empty()
    return pd.DataFrame(job_list) if job_list else None

# ... (process_nlp_features function remains exactly the same) ...
def process_nlp_features(df, resume_text=None):
    if df is None or df.empty: return df
    nlp = load_spacy_model()
    status_text, progress_bar = st.empty(), st.progress(0, text="Analyzing jobs...")
    total_steps = 4 if resume_text else 3
    status_text.text("Step 1/{} : Extracting Keywords...".format(total_steps))
    df['top_keywords'] = df['job_desc'].apply(extract_keywords)
    progress_bar.progress(1/total_steps)
    status_text.text("Step 2/{} : Extracting Skills...".format(total_steps))
    df['extracted_skills'] = df['job_desc'].apply(lambda x: extract_skills(x, nlp))
    progress_bar.progress(2/total_steps)
    status_text.text("Step 3/{} : Analyzing Sentiment...".format(total_steps))
    df['sentiment_score'] = df['job_desc'].apply(analyze_sentiment)
    progress_bar.progress(3/total_steps)
    if resume_text:
        status_text.text("Step 4/4: Calculating Resume Match Score...")
        df['resume_match'] = df['job_desc'].apply(lambda x: f"{calculate_match(x, resume_text) * 100:.2f}%")
        progress_bar.progress(4/total_steps)
    status_text.empty()
    progress_bar.empty()
    return df


# --- Streamlit App UI ---
st.title("🤖 LinkedIn AI Job Analyzer")
st.write("Scrape job postings and analyze them with NLP to find the best fit.")

ensure_nltk_data()

# ... (Sidebar and Form UI remain exactly the same) ...
with st.sidebar:
    st.header("📝 Resume Analysis")
    st.write("Provide your resume to calculate a match score for each job.")
    resume_option = st.radio("Resume Input Method:", ("Paste Text", "Upload File"), key="resume_option")
    if resume_option == "Paste Text":
        resume_text_input = st.text_area("Paste your resume content here:", height=300, key="resume_paste")
        if resume_text_input: st.session_state['resume_text'] = resume_text_input
    else:
        uploaded_file = st.file_uploader("Upload Resume", type=['txt', 'pdf'], key="resume_upload")
        if uploaded_file:
            st.session_state['resume_text'] = read_pdf(uploaded_file) if uploaded_file.type == "application/pdf" else uploaded_file.getvalue().decode("utf-8")
    if 'resume_text' in st.session_state and st.session_state['resume_text']:
        st.success("Resume loaded successfully!")

with st.form("scrape_form"):
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        title_input = st.text_input("Job Title", value="Data Scientist")
    with col2:
        location_input = st.text_input("Location", value="United Kingdom")
    with col3:
        pages_to_scrape = st.number_input("# Pages", min_value=1, max_value=40, value=1, help="Each page has ~25 jobs.")
    submitted = st.form_submit_button("Analyze Jobs")


if submitted:
    raw_df = run_linkedin_scraper(title_input, location_input, pages_to_scrape)
    if raw_df is not None and not raw_df.empty:
        resume_content = st.session_state.get('resume_text', None)
        enhanced_df = process_nlp_features(raw_df, resume_content)
        st.success(f"Analysis complete! Processed {len(enhanced_df)} jobs.")
        
        # <<< MODIFIED: Added 'salary' to the final display list >>>
        final_cols = [
            'job_title', 'company_name', 'salary', 'hours_posted', 'applicants_count', 
            'resume_match', 'top_keywords', 'extracted_skills', 'sentiment_score', 
            'job_link', 'job_desc'
        ]
        
        if 'resume_match' not in enhanced_df.columns: final_cols.remove('resume_match')
        
        st.dataframe(enhanced_df[final_cols])
        csv_data = convert_df_to_csv(enhanced_df[final_cols])
        st.download_button(
            label="📥 Download Enhanced Data as CSV",
            data=csv_data,
            file_name=f"{title_input.replace(' ', '_')}_jobs_analyzed.csv",
            mime='text/csv',
        )
    else:
        st.error("Scraping did not return any data. Please try different keywords.")