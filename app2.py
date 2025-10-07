from dotenv import load_dotenv
load_dotenv()

import base64
import streamlit as st
import os
import io
import json
import time
from PIL import Image, ImageDraw, ImageFont
import pdf2image
from docx import Document
import google.generativeai as genai
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Configure the Generative AI Model
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Google API Key not found. Please set the GOOGLE_API_KEY environment variable.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# Initialize session state for chat history and analysis results
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'resume_text' not in st.session_state:
    st.session_state.resume_text = ""

def get_gemini_response(input_text, file_content, prompt, temperature=0.7):
    """
    Generates a response using Google Generative AI model
    """
    try:
        # Create a more structured request with proper formatting
        request_parts = [
            f"Job Description: {input_text}",
            *file_content,
            f"Instruction: {prompt}"
        ]
        
        # Use newer model with safety settings
        generation_config = {
            "temperature": temperature,
            "top_p": 0.8,
            "top_k": 40,
        }
        
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        
        model = genai.GenerativeModel(
            'gemini-1.5-flash',
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        response = model.generate_content(request_parts)
        return response.text
    except Exception as e:
        raise ValueError(f"Error generating response from Gemini model: {e}")

def extract_text_from_pdf(uploaded_file):
    """Extract text from PDF for better analysis"""
    try:
        images = pdf2image.convert_from_bytes(uploaded_file.read())
        text = ""
        # For now, we'll use the image approach, but you can integrate OCR here
        return text
    except Exception as e:
        st.warning(f"Text extraction from PDF failed: {e}")
        return ""

def input_pdf_setup(uploaded_file):
    """
    Processes uploaded PDF file and converts to base64 encoded image
    """
    try:
        # Reset file pointer
        uploaded_file.seek(0)
        images = pdf2image.convert_from_bytes(uploaded_file.read())
        first_page = images[0]
        
        # Create higher quality image
        img_byte_arr = io.BytesIO()
        first_page.save(img_byte_arr, format='JPEG', quality=85)
        img_byte_arr = img_byte_arr.getvalue()
        
        file_parts = [{
            "mime_type": "image/jpeg",
            "data": base64.b64encode(img_byte_arr).decode()
        }]
        return file_parts
    except Exception as e:
        raise ValueError(f"Error processing PDF: {e}")

def input_docx_setup(uploaded_file):
    """
    Processes uploaded DOCX file and converts to image with better formatting
    """
    try:
        uploaded_file.seek(0)
        document = Document(uploaded_file)
        text = "\n".join([paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()])
        
        # Store text in session state for later use
        st.session_state.resume_text = text
        
        # Improved text wrapping
        lines = []
        max_chars = 100
        font_size = 12
        
        for paragraph in text.split('\n'):
            words = paragraph.split()
            current_line = ""
            
            for word in words:
                if len(current_line) + len(word) + 1 <= max_chars:
                    current_line += f"{word} "
                else:
                    if current_line:
                        lines.append(current_line.strip())
                    current_line = f"{word} "
            
            if current_line:
                lines.append(current_line.strip())
            lines.append("")  # Add empty line between paragraphs
        
        # Calculate image dimensions
        line_height = 20
        img_width = 900
        img_height = max(400, len(lines) * line_height + 40)
        
        # Create image with better visual appearance
        img = Image.new('RGB', (img_width, img_height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Draw text with better formatting
        y_position = 20
        for line in lines:
            if line.strip():  # Only draw non-empty lines
                draw.text((20, y_position), line, fill=(0, 0, 0), font=font)
            y_position += line_height
        
        # Convert to base64
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=90)
        img_byte_arr = img_byte_arr.getvalue()
        
        file_parts = [{
            "mime_type": "image/jpeg",
            "data": base64.b64encode(img_byte_arr).decode()
        }]
        return file_parts
    except Exception as e:
        raise ValueError(f"Error processing DOCX: {e}")

def process_uploaded_file(uploaded_file):
    """
    Processes uploaded file based on type with enhanced error handling
    """
    try:
        if uploaded_file.name.endswith(".pdf"):
            return input_pdf_setup(uploaded_file)
        elif uploaded_file.name.endswith(".docx"):
            return input_docx_setup(uploaded_file)
        else:
            raise ValueError("Unsupported file format. Please upload PDF or DOCX.")
    except Exception as e:
        st.error(f"File processing error: {str(e)}")
        raise

def create_visualization(metrics_data):
    """Create visualizations for analysis results"""
    try:
        # Match percentage gauge
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = metrics_data.get('match_percentage', 0),
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "ATS Match Score"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightcoral"},
                    {'range': [50, 80], 'color': "lightyellow"},
                    {'range': [80, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        
        return fig_gauge
    except Exception as e:
        st.error(f"Visualization error: {e}")
        return None

def analyze_resume_structure(response_text):
    """Analyze response and extract structured data"""
    try:
        # This is a simplified extraction - you can make it more sophisticated
        analysis = {
            'match_percentage': 0,
            'strengths': [],
            'weaknesses': [],
            'missing_skills': [],
            'suggestions': []
        }
        
        # Simple keyword extraction (you can enhance this with regex)
        if '%' in response_text:
            import re
            percentages = re.findall(r'(\d+)%', response_text)
            if percentages:
                analysis['match_percentage'] = int(percentages[0])
        
        return analysis
    except Exception as e:
        return {'match_percentage': 0, 'strengths': [], 'weaknesses': [], 'missing_skills': [], 'suggestions': []}

# Enhanced Prompts with better structure
input_prompts = {
    "submit1": """
        As an experienced Technical HR Manager, provide a comprehensive evaluation of the resume against the job description.
        
        Structure your response as:
        🎯 **Overall Assessment**
        [Brief summary of alignment]
        
        ✅ **Key Strengths:**
        - [Strength 1 with specific examples]
        - [Strength 2 with specific examples]
        
        ⚠️ **Areas for Improvement:**
        - [Weakness 1 with suggestions]
        - [Weakness 2 with suggestions]
        
        💡 **Recommendations:**
        - [Actionable recommendation 1]
        - [Actionable recommendation 2]
    """,
    
    "submit3": """
        As an ATS expert, provide a detailed match analysis in this exact format:
        
        📊 **MATCH SCORE: [X]%**
        
        🔍 **KEYWORDS ANALYSIS:**
        ✅ **Present Keywords:** [List 5-8 matching keywords]
        ❌ **Missing Keywords:** [List 5-8 missing critical keywords]
        
        📈 **BREAKDOWN:**
        - Technical Skills Match: [X]%
        - Experience Match: [X]%
        - Education Match: [X]%
        
        💭 **FINAL ASSESSMENT:**
        [Detailed analysis of why this score was given and what can be improved]
    """,
    
    "submit4": """
        As a career development specialist, provide actionable skill improvement plan:
        
        🎯 **SKILL GAP ANALYSIS**
        
        🚀 **HIGH PRIORITY SKILLS:**
        - [Skill 1]: [Why it's important and how to acquire]
        - [Skill 2]: [Why it's important and how to acquire]
        
        📚 **LEARNING RESOURCES:**
        - Online Courses: [Specific platform/course recommendations]
        - Practical Projects: [Project ideas to build skills]
        - Certifications: [Relevant certifications]
        
        ⏱️ **30-DAY ACTION PLAN:**
        - Week 1: [Specific actions]
        - Week 2: [Specific actions]
        - Week 3-4: [Specific actions]
    """,
    
    "submit5": """
        As an ATS scanner, identify missing critical elements:
        
        🔴 **CRITICAL MISSING KEYWORDS:**
        [List exact keywords and phrases missing from resume]
        
        🟡 **IMPORTANT BUT MISSING:**
        [List important skills/terms not found]
        
        🟢 **PRESENT BUT CAN BE ENHANCED:**
        [Skills that are present but need better highlighting]
        
        💡 **OPTIMIZATION TIPS:**
        - Keyword placement suggestions
        - Section improvements
        - Formatting recommendations
    """,
    
    "submit6": """
        Provide a comprehensive resume analysis:
        
        📋 **EXECUTIVE SUMMARY**
        [Overall suitability assessment]
        
        🔬 **DETAILED ANALYSIS:**
        
        **Experience Alignment:**
        ✅ [What matches well]
        ❌ [Where there are gaps]
        
        **Skill Mapping:**
        ✅ [Technical skills that align]
        ❌ [Technical skills missing]
        
        **Achievement Impact:**
        ✅ [Strong achievements]
        💡 [How to better quantify achievements]
        
        **ATS Optimization Score:**
        - Formatting: [X]/10
        - Keyword Density: [X]/10
        - Readability: [X]/10
        
        🛠️ **SPECIFIC IMPROVEMENT RECOMMENDATIONS:**
        [Bulleted list of specific changes needed]
    """
}

# Streamlit App Configuration
st.set_page_config(
    page_title="Advanced ATS Resume Expert",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
</style>
""", unsafe_allow_html=True)

# Main App Interface
st.markdown('<h1 class="main-header">🚀 Advanced ATS Resume Expert</h1>', unsafe_allow_html=True)

# Sidebar for additional features
with st.sidebar:
    st.header("⚙️ Settings")
    analysis_mode = st.selectbox(
        "Analysis Mode",
        ["Comprehensive", "Quick Scan", "Technical Focus", "Executive Level"]
    )
    
    temperature = st.slider("AI Creativity Level", 0.1, 1.0, 0.7, 0.1)
    
    st.header("📈 History")
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
    
    for i, chat in enumerate(st.session_state.chat_history[-5:]):
        st.text(f"{i+1}. {chat['type']} - {chat['timestamp']}")

# Main layout columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Job Description")
    input_text = st.text_area(
        "Paste the job description here:",
        height=200,
        placeholder="Copy and paste the complete job description here...",
        key="input"
    )
    
    st.subheader("📄 Upload Resume")
    uploaded_file = st.file_uploader(
        "Choose your resume file",
        type=["pdf", "docx"],
        help="Supported formats: PDF, DOCX"
    )
    
    if uploaded_file is not None:
        # File info card
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024:.2f} KB",
            "File type": uploaded_file.type
        }
        
        st.markdown('<div class="success-message">✅ File Uploaded Successfully</div>', unsafe_allow_html=True)
        st.json(file_details)
    else:
        st.info("👆 Please upload your resume to get started")

with col2:
    st.subheader("🎯 Analysis Tools")
    
    # Analysis buttons in a grid
    col1_btn, col2_btn = st.columns(2)
    
    with col1_btn:
        submit1 = st.button("📋 Resume Evaluation", use_container_width=True)
        submit3 = st.button("📊 Match Percentage", use_container_width=True)
        submit4 = st.button("🚀 Skill Improvement", use_container_width=True)
    
    with col2_btn:
        submit5 = st.button("🔍 Missing Keywords", use_container_width=True)
        submit6 = st.button("📈 Detailed Analysis", use_container_width=True)
        compare_btn = st.button("⚡ Quick Compare", use_container_width=True)

# Additional features section
st.markdown("---")
st.subheader("✨ Additional Features")

# Feature columns
feat_col1, feat_col2, feat_col3 = st.columns(3)

with feat_col1:
    if st.button("💬 Interview Questions"):
        if uploaded_file and input_text:
            with st.spinner("Generating interview questions..."):
                try:
                    file_content = process_uploaded_file(uploaded_file)
                    prompt = "Generate 5 technical and 3 behavioral interview questions based on the resume and job description."
                    response = get_gemini_response(input_text, file_content, prompt, temperature)
                    st.subheader("🎤 Suggested Interview Questions")
                    st.write(response)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        'type': 'Interview Questions',
                        'timestamp': datetime.now().strftime("%H:%M:%S"),
                        'response': response
                    })
                except Exception as e:
                    st.error(f"Error: {e}")

with feat_col2:
    if st.button("📝 Cover Letter Tips"):
        if uploaded_file and input_text:
            with st.spinner("Generating cover letter suggestions..."):
                try:
                    file_content = process_uploaded_file(uploaded_file)
                    prompt = "Provide cover letter writing tips and key points to highlight based on the resume and job description."
                    response = get_gemini_response(input_text, file_content, prompt, temperature)
                    st.subheader("✍️ Cover Letter Suggestions")
                    st.write(response)
                except Exception as e:
                    st.error(f"Error: {e}")

with feat_col3:
    if st.button("🔄 ATS Optimization"):
        if uploaded_file:
            with st.spinner("Analyzing ATS optimization..."):
                try:
                    file_content = process_uploaded_file(uploaded_file)
                    prompt = "Provide specific ATS optimization tips for this resume including keyword placement, formatting suggestions, and content structure improvements."
                    response = get_gemini_response(input_text, file_content, prompt, temperature)
                    st.subheader("🔧 ATS Optimization Tips")
                    st.write(response)
                except Exception as e:
                    st.error(f"Error: {e}")

# Main action handlers
action_mapping = [
    (submit1, "submit1"),
    (submit3, "submit3"), 
    (submit4, "submit4"),
    (submit5, "submit5"),
    (submit6, "submit6"),
]

for submit, prompt_key in action_mapping:
    if submit:
        if not uploaded_file:
            st.error("📁 Please upload your resume first!")
            st.stop()
        
        if not input_text.strip():
            st.error("📝 Please enter the job description!")
            st.stop()
        
        try:
            with st.spinner("🔍 Analyzing your resume..."):
                # Show progress
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # Process file and get response
                file_content = process_uploaded_file(uploaded_file)
                prompt = input_prompts[prompt_key]
                response = get_gemini_response(input_text, file_content, prompt, temperature)
                
                # Store analysis results
                analysis_data = analyze_resume_structure(response)
                st.session_state.analysis_results = analysis_data
                
                # Display results
                st.subheader("🎯 Analysis Results")
                
                # Create columns for results and visualization
                result_col, viz_col = st.columns([2, 1])
                
                with result_col:
                    st.markdown(response)
                
                with viz_col:
                    if analysis_data['match_percentage'] > 0:
                        fig = create_visualization(analysis_data)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Show key metrics
                        st.metric("Match Score", f"{analysis_data['match_percentage']}%")
                        st.metric("Strengths Identified", len(analysis_data['strengths']))
                        st.metric("Improvement Areas", len(analysis_data['weaknesses']))
                
                # Add to chat history
                st.session_state.chat_history.append({
                    'type': prompt_key.replace('submit', 'Analysis '),
                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                    'response': response[:200] + "..." if len(response) > 200 else response
                })
                
                # Success message
                st.balloons()
                
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>Built with ❤️ using Streamlit and Google Gemini AI</p>
        <p>🚀 Get more job interviews with optimized resumes!</p>
    </div>
    """,
    unsafe_allow_html=True
)
