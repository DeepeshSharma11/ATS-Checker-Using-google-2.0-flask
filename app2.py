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
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(input_text, file_content, prompt):
    """
    Generates a response using the Google Generative AI model based on the input text, file content, and prompt.
    """
    try:
        # Updated to use the gemini-2.0-flash model
        model = genai.GenerativeModel('gemini-2.0-flash')  
        response = model.generate_content([input_text, file_content[0], prompt])
        return response.text
    except Exception as e:
        raise ValueError(f"Error generating response from Gemini model: {e}")

def input_pdf_setup(uploaded_file):
    """
    Processes the uploaded PDF file and converts the first page into a base64 encoded image.
    """
    try:
        # Convert the PDF to images
        images = pdf2image.convert_from_bytes(uploaded_file.read())

        first_page = images[0]

        # Convert to bytes
        img_byte_arr = io.BytesIO()
        first_page.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        file_parts = [
            {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(img_byte_arr).decode()  # encode to base64
            }
        ]
        return file_parts
    except Exception as e:
        raise ValueError(f"Error processing PDF: {e}")

def input_docx_setup(uploaded_file):
    """
    Processes the uploaded DOCX file, extracts text, and converts it into an image.
    """
    try:
        # Load the DOCX file
        document = Document(uploaded_file)
        text = "\n".join([paragraph.text for paragraph in document.paragraphs])

        # Create an image from the text
        img = Image.new('RGB', (800, 1000), color=(255, 255, 255))  # White background
        draw = ImageDraw.Draw(img)

        # Use a default system font
        font = ImageFont.load_default()
        draw.text((10, 10), text, fill=(0, 0, 0), font=font)

        # Convert to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        file_parts = [
            {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(img_byte_arr).decode()  # encode to base64
            }
        ]
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
    st.write("Please upload a resume.")

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
for submit, prompt_key in zip([submit1, submit3, submit4, submit5, submit6], input_prompts.keys()):
    if submit:
        try:
            if uploaded_file is not None:
                file_content = process_uploaded_file(uploaded_file)
                prompt = input_prompts[prompt_key]
                response = get_gemini_response(input_text, file_content, prompt)
                st.subheader("The Response is:")
                st.write(response)
            else:
                st.write("Please upload the resume.")
        except Exception as e:
            st.error(f"An error occurred: {e}")