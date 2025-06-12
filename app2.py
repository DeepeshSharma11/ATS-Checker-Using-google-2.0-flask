from dotenv import load_dotenv
load_dotenv()

import base64
import streamlit as st
import os
import io
from PIL import Image, ImageDraw, ImageFont
import pdf2image
from docx import Document
import google.generativeai as genai

# Configure the Generative AI Model
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Google API Key not found. Please set the GOOGLE_API_KEY environment variable.")
    st.stop()
genai.configure(api_key=GOOGLE_API_KEY)

def get_gemini_response(input_text, file_content, prompt):
    """
    Generates a response using the Google Generative AI model based on the input text, file content, and prompt.
    """
    try:
        # The Gemini API expects each file/image as a dict in the prompt list
        # So we concatenate: input_text, file_content (a list), prompt
        # Example: [input_text, {file_part_dict}, prompt]
        request_parts = [input_text] + file_content + [prompt]
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(request_parts)
        return response.text
    except Exception as e:
        raise ValueError(f"Error generating response from Gemini model: {e}")

def input_pdf_setup(uploaded_file):
    """
    Processes the uploaded PDF file and converts the first page into a base64 encoded image.
    """
    try:
        images = pdf2image.convert_from_bytes(uploaded_file.read())
        first_page = images[0]
        img_byte_arr = io.BytesIO()
        first_page.save(img_byte_arr, format='JPEG')
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
    Processes the uploaded DOCX file, extracts text, and converts it into an image.
    """
    try:
        document = Document(uploaded_file)
        text = "\n".join([paragraph.text for paragraph in document.paragraphs])
        # Split text into lines that fit into the image
        lines = []
        max_chars = 90
        for paragraph in text.split('\n'):
            while len(paragraph) > max_chars:
                lines.append(paragraph[:max_chars])
                paragraph = paragraph[max_chars:]
            lines.append(paragraph)

        img_height = 20 + 20 * len(lines)
        img = Image.new('RGB', (800, max(img_height, 200)), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        y_text = 10
        for line in lines:
            draw.text((10, y_text), line, fill=(0, 0, 0), font=font)
            y_text += 20

        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
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
    Processes the uploaded file based on its type (.pdf or .docx).
    """
    if uploaded_file.name.endswith(".pdf"):
        return input_pdf_setup(uploaded_file)
    elif uploaded_file.name.endswith(".docx"):
        return input_docx_setup(uploaded_file)
    else:
        raise ValueError("Unsupported file format. Please upload a PDF or DOCX file.")

# Streamlit App
st.set_page_config(page_title="ATS Resume Expert")
st.header("ATS Tracking System")

# User Input
input_text = st.text_area("Job Description: ", key="input")
uploaded_file = st.file_uploader("Upload your resume (PDF or DOCX)...", type=["pdf", "docx"])

if uploaded_file is not None:
    st.write("File Uploaded Successfully")
else:
    st.info("Please upload a resume.")

# Buttons for Actions
submit1 = st.button("Tell Me About the Resume")
submit3 = st.button("Percentage Match")
submit4 = st.button("Skill Improvement Suggestions")
submit5 = st.button("Highlight Missing Skills")
submit6 = st.button("Detailed Analysis")

# Prompts
input_prompts = {
    "submit1": """
        You are an experienced Technical Human Resource Manager. Your task is to review the provided resume against the job description. 
        Please share your professional evaluation on whether the candidate's profile aligns with the role. 
        Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements.
    """,
    "submit3": """
        You are a skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality. 
        Your task is to evaluate the resume against the provided job description. Give me the percentage of match if the resume matches
        the job description. First, the output should come as a percentage and then keywords missing and last, final thoughts.
    """,
    "submit4": """
        As a career coach, identify key areas where the candidate's skills can be improved to better align with the job description.
        Provide actionable suggestions for skill development and enhancement.
    """,
    "submit5": """
        You are an ATS system. Highlight the keywords and skills missing in the provided resume that are critical for the job description.
        Focus on technical skills, soft skills, and language proficiency.
    """,
    "submit6": """
        Perform a detailed analysis of the resume against the job description. Provide insights into the candidate's suitability, 
        focusing on experience, skills, and achievements. Suggest specific ways to improve the resume for a better match.
    """
}

# Action Handlers
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
            st.error("Please upload the resume.")
            st.stop()
        if not input_text.strip():
            st.error("Please enter the job description.")
            st.stop()
        try:
            file_content = process_uploaded_file(uploaded_file)
            prompt = input_prompts[prompt_key]
            response = get_gemini_response(input_text, file_content, prompt)
            st.subheader("The Response is:")
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
