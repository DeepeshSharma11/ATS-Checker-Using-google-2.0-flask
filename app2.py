import base64
import streamlit as st
import os
import io
import re
import time
from PIL import Image, ImageDraw, ImageFont
import pdf2image
from docx import Document
# The new SDK import
from google import genai
from google.genai import types
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from dotenv import load_dotenv

# ==========================================
# CONFIGURATION & API SETUP
# ==========================================
load_dotenv()

# Model selection: gemini-2.0-flash is the current state-of-the-art flash model
MODEL_ID = "gemini-2.0-flash"

# Fetch API Key from environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize the new Google GenAI Client
if GOOGLE_API_KEY:
    client = genai.Client(api_key=GOOGLE_API_KEY)
else:
    st.error("Missing GOOGLE_API_KEY. Please ensure it is set in your .env file or environment variables.")
    st.stop()

# Initialize session state for history and analysis
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}

# ==========================================
# CORE AI LOGIC (NEW SDK)
# ==========================================
def get_gemini_response(input_text, file_content_b64, prompt, temperature=0.7):
    """
    Generates a response using the latest Gemini model via the new SDK.
    """
    try:
        # Construct content using new SDK types
        content = [
            f"TARGET JOB DESCRIPTION:\n{input_text}",
            types.Part.from_bytes(
                data=base64.b64decode(file_content_b64),
                mime_type="image/jpeg"
            ),
            f"INSTRUCTION:\n{prompt}"
        ]
        
        # Generation configuration
        config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=0.9,
            candidate_count=1,
            system_instruction="You are a professional ATS (Applicant Tracking System) and Senior Technical Recruiter. Provide objective, data-driven analysis."
        )
        
        # API call
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=content,
            config=config
        )
        
        return response.text
    except Exception as e:
        raise ValueError(f"Gemini API Error: {str(e)}")

# ==========================================
# MULTIMODAL FILE PROCESSING
# ==========================================
def input_pdf_setup(uploaded_file):
    """Converts PDF to high-quality image for Vision-based analysis."""
    try:
        uploaded_file.seek(0)
        images = pdf2image.convert_from_bytes(uploaded_file.read())
        # Modern ATS analysis often focuses on the first page/summary
        # but we use high quality JPEG for the AI to "read" the text
        first_page = images[0]
        
        img_byte_arr = io.BytesIO()
        first_page.save(img_byte_arr, format='JPEG', quality=95)
        return base64.b64encode(img_byte_arr.getvalue()).decode()
    except Exception as e:
        raise ValueError(f"PDF Error: {e}. (Ensure 'poppler' is installed on your OS)")

def input_docx_setup(uploaded_file):
    """Converts DOCX to high-resolution image to maintain multimodal consistency."""
    try:
        uploaded_file.seek(0)
        doc = Document(uploaded_file)
        full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        
        # Create a visual representation for the Vision model
        img_w, img_h = 1000, 1400
        img = Image.new('RGB', (img_w, img_h), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        try:
            # Try to load a clean font, fallback to default
            font = ImageFont.load_default()
        except:
            font = None

        y_pos = 50
        for line in full_text.split('\n')[:70]: # First 70 lines
            draw.text((50, y_pos), line[:110], fill=(0, 0, 0), font=font)
            y_pos += 18

        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        return base64.b64encode(img_byte_arr.getvalue()).decode()
    except Exception as e:
        raise ValueError(f"DOCX Processing Error: {e}")

def process_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return input_pdf_setup(uploaded_file)
    elif name.endswith(".docx"):
        return input_docx_setup(uploaded_file)
    else:
        raise ValueError("Unsupported format. Use PDF or DOCX.")

# ==========================================
# ANALYTICS & VISUALIZATION
# ==========================================
def parse_analysis_data(text):
    """Extracts match percentage and insights using Regex."""
    data = {'match_percentage': 0}
    match = re.search(r"(\d+)%", text)
    if match:
        data['match_percentage'] = int(match.group(1))
    return data

def create_gauge_chart(score):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        title = {'text': "ATS Compatibility Score", 'font': {'size': 20, 'color': '#1f77b4'}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "#1f77b4"},
            'bgcolor': "white",
            'steps': [
                {'range': [0, 50], 'color': "#ffebee"},
                {'range': [50, 80], 'color': "#fff9c4"},
                {'range': [80, 100], 'color': "#e8f5e9"}
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}
        }
    ))
    fig.update_layout(height=350, margin=dict(l=30, r=30, t=50, b=20))
    return fig

# ==========================================
# ENHANCED PROMPTS
# ==========================================
PROMPTS = {
    "submit1": """
        Evaluate this resume against the JD as a Technical HR Manager.
        Structure:
        🎯 **Overall Assessment**: [Alignment Summary]
        ✅ **Key Strengths**: [List 3-5 matches]
        ⚠️ **Gaps**: [List 3-5 missing areas]
        💡 **Actionable Recommendations**: [Next steps]
    """,
    "submit3": """
        Act as an ATS Scanner. Output a 'MATCH SCORE: [X]%' followed by a detailed keyword analysis. 
        List matching keywords and critical missing keywords from the JD.
    """,
    "submit4": """
        Act as a Career Development Coach. Create a 30-day skill improvement plan based on the resume gaps. 
        Recommend specific certifications, project ideas, and resources.
    """,
    "submit5": """
        Identify CRITICAL missing keywords (Red) and IMPORTANT missing keywords (Yellow). 
        Suggest exact phrasing to add to the resume for better indexing.
    """,
    "submit6": """
        Provide a detailed technical audit. Analyze specific technical tools, frameworks, and experience years. 
        Rate formatting (0-10) and readability (0-10).
    """,
    "interview": "Generate 5 technical and 3 behavioral interview questions based on this resume's specific experience and this JD.",
    "optimization": "Analyze the visual layout and keyword density. Suggest specific formatting changes for ATS readability."
}

# ==========================================
# STREAMLIT UI
# ==========================================
st.set_page_config(page_title="ATS Pro AI", page_icon="🎯", layout="wide")

# Custom Styling
st.markdown("""
<style>
    .main-header { font-size: 2.8rem; color: #1f77b4; text-align: center; font-weight: 800; margin-bottom: 0px;}
    .sub-header { text-align: center; color: #666; margin-bottom: 2rem; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3.5em; font-weight: bold; background-color: #1f77b4; color: white; }
    .stButton>button:hover { background-color: #155a8a; border: 1px solid white; }
    .report-container { background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🚀 Advanced ATS Resume Expert</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Gemini 2.0 Flash Edition • Intelligent Multi-Format Analysis</div>', unsafe_allow_html=True)

# Main Grid
col_inputs, col_results = st.columns([1, 1], gap="large")

with col_inputs:
    st.subheader("📋 Job & Resume Data")
    jd_input = st.text_area("Paste Job Description", height=250, placeholder="Requirements, responsibilities, skills...")
    
    upload_file = st.file_uploader("Upload Resume", type=["pdf", "docx"], help="PDF and DOCX supported")
    if upload_file:
        st.success(f"File '{upload_file.name}' loaded.")

    st.divider()
    st.subheader("🛠️ Analysis Tools")
    
    r1_c1, r1_c2 = st.columns(2)
    with r1_c1:
        eval_btn = st.button("📋 Comprehensive Review")
        match_btn = st.button("📊 Get Match Score")
    with r1_c2:
        gap_btn = st.button("🔍 Keyword Gaps")
        roadmap_btn = st.button("🚀 Career Roadmap")
    
    r2_c1, r2_c2 = st.columns(2)
    with r2_c1:
        interview_btn = st.button("🎤 Interview Questions")
    with r2_c2:
        opt_btn = st.button("🔧 ATS Optimization")

# Action Logic
triggered_key = None
if eval_btn: triggered_key = "submit1"
elif match_btn: triggered_key = "submit3"
elif roadmap_btn: triggered_key = "submit4"
elif gap_btn: triggered_key = "submit5"
elif interview_btn: triggered_key = "interview"
elif opt_btn: triggered_key = "optimization"

with col_results:
    if triggered_key:
        if not upload_file or not jd_input:
            st.warning("⚠️ Please provide both the Resume file and the Job Description.")
        else:
            try:
                with st.spinner("🤖 AI is analyzing your documents..."):
                    # Step 1: Process File to B64
                    b64_img = process_file(upload_file)
                    
                    # Step 2: Get AI Response
                    response = get_gemini_response(jd_input, b64_img, PROMPTS[triggered_key])
                    
                    # Step 3: Parse Data for Visuals
                    analysis_data = parse_metrics = parse_analysis_data(response)
                    
                    # Step 4: UI Presentation
                    st.subheader("✨ Analysis Output")
                    
                    if analysis_data['match_percentage'] > 0 or triggered_key == "submit3":
                        st.plotly_chart(create_gauge_chart(analysis_data['match_percentage']), use_container_width=True)
                    
                    st.markdown(f'<div class="report-container">{response}</div>', unsafe_allow_html=True)
                    
                    # Step 5: Save to history
                    st.session_state.chat_history.append({
                        "type": triggered_key,
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "score": analysis_data['match_percentage']
                    })
                    
                    st.balloons()
                    
                    # Step 6: Download option
                    st.download_button(
                        label="📥 Download Report",
                        data=response,
                        file_name=f"ATS_Report_{datetime.now().strftime('%Y%m%d')}.md",
                        mime="text/markdown"
                    )
            except Exception as e:
                st.error(f"❌ Application Error: {str(e)}")
    else:
        st.info("Upload your resume and select an analysis tool on the left to begin.")
        
        # Display history if exists
        if st.session_state.chat_history:
            st.subheader("🕒 Recent Activity")
            for item in st.session_state.chat_history[-3:]:
                st.text(f"[{item['timestamp']}] {item['type']} - Score: {item['score']}%")

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray; font-size: 0.8em;'>Powered by Google Gemini 2.0 Flash • Multi-modal Visual Extraction Pipeline</div>", unsafe_allow_html=True)