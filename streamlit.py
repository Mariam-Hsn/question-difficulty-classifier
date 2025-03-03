import streamlit as st
import pickle
from docx import Document
import PyPDF2
import re

# Function to clean and preprocess text
def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

# Function to extract text from files
def extract_text_from_file(uploaded_file):
    if uploaded_file.name.endswith('.txt'):
        return uploaded_file.read().decode('utf-8')
    elif uploaded_file.name.endswith('.docx'):
        doc = Document(uploaded_file)
        return "\n".join([p.text for p in doc.paragraphs])
    elif uploaded_file.name.endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        return "\n".join([page.extract_text() for page in pdf_reader.pages])
    else:
        st.error("Unsupported file type. Please upload a .txt, .docx, or .pdf file.")
        return ""

# Load the trained model
MODEL_PATH = "question_classifier.pkl"
try:
    with open(MODEL_PATH, 'rb') as file:
        model_pipeline = pickle.load(file)
except FileNotFoundError:
    st.error(f"Model file '{MODEL_PATH}' not found. Please train the model first.")

# Streamlit app UI
st.title("Question Difficulty Classifier")
st.write("Upload a file with questions (txt, docx, pdf) to classify their difficulty level.")

# Upload a file
uploaded_file = st.file_uploader("Upload a file", type=["txt", "docx", "pdf"])
if uploaded_file:
    # Extract text from the uploaded file
    extracted_text = extract_text_from_file(uploaded_file)
    if extracted_text:
        # Split the text into questions based on the question mark ('?')
        questions = [clean_text(q) for q in re.split(r'\?+\s*', extracted_text) if q.strip()]
        
        # Display the detected questions
        st.write("Detected Questions:")
        st.write(questions)

        # Predict difficulty levels
        if 'model_pipeline' in locals():
            predictions = model_pipeline.predict(questions)
            difficulty_map = {0: "Easy", 1: "Medium", 2: "Hard"}
            results = [{"Question": q, "Predicted Difficulty": difficulty_map[p]} for q, p in zip(questions, predictions)]
            st.write("Classification Results:")
            st.table(results)
        else:
            st.error("Model is not loaded. Please check your model file.")