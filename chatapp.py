# app.py
# AI Job Buddy – Ghana
# Streamlit deployment with TF-IDF baseline and DistilBERT+OpenVINO option
import os
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import re, json, warnings
from typing import List, Dict
warnings.filterwarnings('ignore')

# ============ OpenVINO Matcher ============
try:
    from openvino.runtime import Core
    from transformers import AutoTokenizer

    class OpenVINOJobMatcher:
        def __init__(self, model_xml="openvino_model/model.xml",
                     tokenizer_dir="distilbert_finetuned",
                     label_map="distilbert_finetuned/label_map.json"):
            self.core = Core()
            self.model = self.core.read_model(model=model_xml)
            self.compiled = self.core.compile_model(self.model, device_name="CPU")
            self.input_names = [inp.get_any_name() for inp in self.compiled.inputs]
            self.output_name = self.compiled.outputs[0].get_any_name()

            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
            with open(label_map, "r") as f:
                self.label_map = json.load(f)
            self.labels = [self.label_map[str(i)] if str(i) in self.label_map else self.label_map[i] for i in range(len(self.label_map))]

        def predict_category(self, text, max_length=128):
            enc = self.tokenizer(text, return_tensors="np",
                                 padding="max_length",
                                 truncation=True,
                                 max_length=max_length)
            inputs = {
                self.input_names[0]: enc["input_ids"].astype("int32"),
                self.input_names[1]: enc["attention_mask"].astype("int32")
            }
            result = self.compiled(inputs)[self.output_name]
            probs = np.exp(result) / np.exp(result).sum(axis=-1, keepdims=True)
            pred = int(np.argmax(probs, axis=-1)[0])
            return self.labels[pred], float(probs[0][pred])

except Exception as e:
    OpenVINOJobMatcher = None
    print("⚠️ OpenVINO not available:", e)

# ============ Job Matching Engine ============
class JobMatchingEngine:
    def __init__(self, use_openvino=False, ov_matcher=None):
        self.jobs_df = None
        self.use_openvino = use_openvino
        self.ov_matcher = ov_matcher
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.job_vectors = None
        self.skills_encoder = LabelEncoder()

    def load_jobs(self, csv_path: str):
        self.jobs_df = pd.read_csv(csv_path)
        self.preprocess_jobs()
        # Ensure category column exists (if not, try to infer from job_function/Industry)
        if 'category' not in self.jobs_df.columns:
            # try to create a simple category from job_function or Industry if present
            def derive_cat(r):
                jf = r.get('job_function')
                ind = r.get('Industry')
                if isinstance(jf, str) and jf.strip(): return jf
                if isinstance(ind, str) and ind.strip(): return ind
                return "Other/Unknown"
            self.jobs_df['category'] = self.jobs_df.apply(derive_cat, axis=1)
        if not self.use_openvino:
            self.create_job_vectors()

    def preprocess_jobs(self):
        self.jobs_df = self.jobs_df.fillna('')
        self.jobs_df['clean_title'] = self.jobs_df['name'].astype(str).str.lower()
        self.jobs_df['clean_description'] = self.jobs_df['info'].astype(str).str.lower()
        self.jobs_df['extracted_skills'] = self.jobs_df['clean_description'].apply(self.extract_skills_from_text)
        self.jobs_df['clean_location'] = self.jobs_df['Location'].apply(self.standardize_location)
        self.jobs_df['experience_level'] = self.jobs_df['details'].apply(self.extract_experience_level)

    def extract_skills_from_text(self, text: str) -> List[str]:
        common_skills = [
            'python','java','javascript','html','css','sql','excel',
            'communication','teamwork','leadership','project management',
            'marketing','sales','customer service','accounting','finance',
            'data analysis','microsoft office','social media','writing',
            'research','problem solving','time management','organization'
        ]
        return [skill for skill in common_skills if skill in str(text).lower()]

    def standardize_location(self, location: str) -> str:
        if pd.isna(location) or location == '':
            return 'Unknown'
        location_lower = str(location).lower()
        if 'accra' in location_lower: return 'Accra'
        elif 'kumasi' in location_lower: return 'Kumasi'
        elif 'tamale' in location_lower: return 'Tamale'
        elif 'cape coast' in location_lower: return 'Cape Coast'
        elif 'takoradi' in location_lower: return 'Takoradi'
        else: return str(location).title()

    def extract_experience_level(self, details: str) -> str:
        if pd.isna(details) or details == '': return 'Mid level'
        d = str(details).lower()
        if 'entry level' in d or '0-2 years' in d: return 'Entry level'
        elif 'executive' in d or 'senior' in d: return 'Executive level'
        else: return 'Mid level'

    def create_job_vectors(self):
        job_texts = [
            f"{row['clean_title']} {row['clean_description']} {' '.join(row['extracted_skills'])}"
            for _, row in self.jobs_df.iterrows()
        ]
        self.job_vectors = self.tfidf_vectorizer.fit_transform(job_texts)

    def find_matching_jobs(self, user_query: str, user_location: str = '', num_results: int = 5) -> List[Dict]:
        results = []
        if self.use_openvino and self.ov_matcher:
            # Use OpenVINO model to predict a category, then sample top jobs from that category
            predicted_cat, confidence = self.ov_matcher.predict_category(user_query)
            subset = self.jobs_df[self.jobs_df['category'].astype(str) == str(predicted_cat)]
            if subset.empty:
                # fallback to TF-IDF if category not found
                if self.job_vectors is None:
                    self.create_job_vectors()
            else:
                top_jobs = subset.sample(min(len(subset), num_results)).to_dict(orient='records')
                for job in top_jobs:
                    results.append({
                        'title': job.get('name',''),
                        'company': job.get('hiring_firm',''),
                        'location': job.get('clean_location',''),
                        'salary': job.get('Salary',''),
                        'job_type': job.get('Job type',''),
                        'experience_level': job.get('experience_level',''),
                        'similarity_score': confidence,
                        'job_url': job.get('job_url',''),
                        'skills': job.get('extracted_skills',[])
                    })
                return results

        # Default TF-IDF retrieval fallback (or regular baseline mode)
        if self.job_vectors is None:
            self.create_job_vectors()
        query_vector = self.tfidf_vectorizer.transform([str(user_query).lower()])
        similarity_scores = cosine_similarity(query_vector, self.job_vectors)[0]
        job_indices = np.argsort(similarity_scores)[::-1]
        for idx in job_indices[:num_results * 3]:
            job = self.jobs_df.iloc[idx]
            if user_location and user_location.lower() not in job['clean_location'].lower():
                continue
            if similarity_scores[idx] < 0.01:
                continue
            results.append({
                'title': job['name'],
                'company': job['hiring_firm'],
                'location': job['clean_location'],
                'salary': job['Salary'],
                'job_type': job['Job type'],
                'experience_level': job['experience_level'],
                'similarity_score': similarity_scores[idx],
                'job_url': job['job_url'],
                'skills': job['extracted_skills']
            })
            if len(results) >= num_results:
                break
        return results

# ============ Conversational Interface ============
class ConversationalInterface:
    def __init__(self, job_engine: JobMatchingEngine):
        self.job_engine = job_engine
        self.conversation_state = {}

    def process_user_input(self, user_input: str, user_id: str = 'default') -> str:
        if user_id not in self.conversation_state:
            self.conversation_state[user_id] = {'context': 'greeting', 'preferences': {}}
        # quick rule-based intent detection
        if any(word in user_input.lower() for word in ['job','work','position','career','openings','vacancy','vacancies']):
            return self.handle_job_search(user_input, user_id)
        if any(word in user_input.lower() for word in ['location','where','city']):
            return self.handle_location_preference(user_input, user_id)
        if any(word in user_input.lower() for word in ['salary','pay','money']):
            return self.handle_salary_inquiry(user_input, user_id)
        # default
        return self.handle_job_search(user_input, user_id)

    def handle_job_search(self, query: str, user_id: str) -> str:
        # try extract a location word
        locations = ['accra', 'kumasi', 'tamale', 'cape coast', 'takoradi']
        user_location = ''
        for location in locations:
            if location in query.lower():
                user_location = location.title()
                break
        jobs = self.job_engine.find_matching_jobs(query, user_location)
        if not jobs:
            return f"😔 No jobs found for '{query}'. Try different keywords or broaden the location."
        response = f"🎯 Found {len(jobs)} job matches:\n\n"
        for i, job in enumerate(jobs, 1):
            salary_info = f"💰 {job['salary']}" if job['salary'] and str(job['salary']).lower() != 'nan' else "💰 Not specified"
            response += f"""**{i}. {job['title']}**
🏢 {job['company']}
📍 {job['location']}
{salary_info}
⭐ Match Score: {job['similarity_score']:.2f}
🔗 {job['job_url']}

---
"""
        return response

    def handle_location_preference(self, query: str, user_id: str) -> str:
        return """📍 I can help you find jobs in major Ghanaian cities (Accra, Kumasi, Tamale, Cape Coast, Takoradi)."""

    def handle_salary_inquiry(self, query: str, user_id: str) -> str:
        return """💰 Salary estimates:
- Entry Level: GHS 800 - 2,000/month
- Mid Level: GHS 2,000 - 5,000/month
- Senior Level: GHS 5,000+/month"""

# ============ Streamlit App ============
def main():
    st.set_page_config(page_title="AI Job Buddy Ghana", page_icon="🤖", layout="wide")
    st.title("🤖 AI Job Buddy - Ghana")
    st.subheader("Your AI-powered job search assistant")

    if 'job_engine' not in st.session_state:
        st.session_state.job_engine = None
        st.session_state.chat_interface = None
        st.session_state.jobs_loaded = False
        st.session_state.messages = []
        st.session_state.uploaded_path = None
        st.session_state.selected_engine = "TF-IDF Baseline"

    st.sidebar.header("📂 Load Job Data")
    uploaded_file = st.sidebar.file_uploader("Upload Ghana jobs CSV file", type=['csv'])

    model_choice = st.sidebar.radio("Select matching engine:", ["TF-IDF Baseline", "DistilBERT + OpenVINO"])
    st.sidebar.markdown("---")
    st.sidebar.write("Tip: pick the engine **before** uploading**. If you change engine after upload, click **Apply engine change** below.")

    # If user has uploaded now, save temp file
    if uploaded_file is not None:
        # save a temporary file so reloads keep it
        tmp_path = "temp_jobs.csv"
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.uploaded_path = tmp_path

    # If user already uploaded earlier and wants to apply a new engine selection
    if st.session_state.jobs_loaded and model_choice != st.session_state.selected_engine:
        if st.sidebar.button("Apply engine change"):
            # reinitialize engine with the new choice using the existing temp file
            try:
                if model_choice == "DistilBERT + OpenVINO" and OpenVINOJobMatcher:
                    ov_matcher = OpenVINOJobMatcher()
                    st.session_state.job_engine = JobMatchingEngine(use_openvino=True, ov_matcher=ov_matcher)
                else:
                    st.session_state.job_engine = JobMatchingEngine()
                # reload jobs from temp path
                st.session_state.job_engine.load_jobs(st.session_state.uploaded_path)
                st.session_state.chat_interface = ConversationalInterface(st.session_state.job_engine)
                st.session_state.selected_engine = model_choice
                st.success(f"Engine switched to: {model_choice}")
            except Exception as e:
                st.error(f"Failed to switch engine: {e}")

    # If there's an uploaded file and jobs not yet loaded, load them
    if st.session_state.uploaded_path and not st.session_state.jobs_loaded:
        try:
            if model_choice == "DistilBERT + OpenVINO" and OpenVINOJobMatcher:
                ov_matcher = OpenVINOJobMatcher()
                st.session_state.job_engine = JobMatchingEngine(use_openvino=True, ov_matcher=ov_matcher)
            else:
                st.session_state.job_engine = JobMatchingEngine()
            st.session_state.job_engine.load_jobs(st.session_state.uploaded_path)
            st.session_state.chat_interface = ConversationalInterface(st.session_state.job_engine)
            st.session_state.jobs_loaded = True
            st.session_state.selected_engine = model_choice
            st.sidebar.success(f"✅ Loaded {len(st.session_state.job_engine.jobs_df)} jobs!")
            # dataset overview
            df = st.session_state.job_engine.jobs_df
            st.sidebar.write("**Dataset overview:**")
            st.sidebar.write(f"• Total jobs: {len(df)}")
            st.sidebar.write(f"• Companies: {df['hiring_firm'].nunique()}")
            st.sidebar.write(f"• Locations: {df['clean_location'].nunique()}")
            if 'category' in df.columns:
                st.sidebar.write(f"• Categories: {df['category'].nunique()}")
        except Exception as e:
            st.sidebar.error(f"Error loading data: {e}")

    # Main chat area
    if st.session_state.jobs_loaded:
        st.header("💬 Chat with Your Job Buddy")
        # display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # chat input
        if prompt := st.chat_input("Ask me about jobs in Ghana..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            with st.chat_message("assistant"):
                response = st.session_state.chat_interface.process_user_input(prompt)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

        # Quick actions (restored)
        st.subheader("🚀 Quick Actions")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🏢 Find Marketing Jobs"):
                prompt = "Find marketing jobs in Ghana"
                st.session_state.messages.append({"role": "user", "content": prompt})
                response = st.session_state.chat_interface.process_user_input(prompt)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

        with col2:
            if st.button("💻 Tech Opportunities"):
                prompt = "Show me IT and technology jobs"
                st.session_state.messages.append({"role": "user", "content": prompt})
                response = st.session_state.chat_interface.process_user_input(prompt)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

        with col3:
            if st.button("🎓 Entry Level Jobs"):
                prompt = "Find entry level positions for fresh graduates"
                st.session_state.messages.append({"role": "user", "content": prompt})
                response = st.session_state.chat_interface.process_user_input(prompt)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

        # utility buttons (clear chat, download results)
        c1, c2 = st.columns([1,1])
        with c1:
            if st.button("🧹 Clear chat"):
                st.session_state.messages = []
                st.rerun()
        with c2:
            if st.button("📥 Download current jobs (CSV)"):
                csv_bytes = pd.DataFrame(st.session_state.job_engine.jobs_df).to_csv(index=False).encode('utf-8')
                st.download_button("Download CSV", csv_bytes, "ghana_jobs_export.csv", "text/csv")

    else:
        st.info("👆 Upload your Ghana jobs CSV file to get started!")
        st.header("🎯 Features")
        col1, col2 = st.columns(2)
        with col1:
            st.write("""
            **🤖 AI-Powered Matching**
            - Intelligent job recommendations
            - Natural language search
            - Skills-based matching
            """)
        with col2:
            st.write("""
            **🇬🇭 Ghana-Focused**
            - Local job market data
            - Ghana-specific locations
            - Cedis salary information
            """)

if __name__ == "__main__":
    main()
