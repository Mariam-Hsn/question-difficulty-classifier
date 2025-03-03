import streamlit as st
import pickle
from docx import Document
import PyPDF2
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import pandas as pd

# Cleaning and preprocessing for text
def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

# Extracting text from files
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

# Load trained model
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
    # Extracting text from uploaded file
    extracted_text = extract_text_from_file(uploaded_file)
    if extracted_text:
        # Split text into questions based on the question mark ('?')
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

            # --- Visualizations ---
            difficulties = [difficulty_map[p] for p in predictions]
            difficulty_counts = pd.Series(difficulties).value_counts()

            # Pie Chart: Difficulty Distribution
            st.subheader("Question Difficulty Distribution (Pie Chart)")
            fig1, ax1 = plt.subplots()
            ax1.pie(difficulty_counts, labels=difficulty_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set2"))
            ax1.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
            st.pyplot(fig1)

            # Word Cloud for Most Common Terms
            st.subheader("Word Cloud for Most Common Terms")
            all_questions = ' '.join(questions)
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_questions)

            fig2, ax2 = plt.subplots(figsize=(10, 8))
            ax2.imshow(wordcloud, interpolation='bilinear')
            ax2.axis('off')
            st.pyplot(fig2)

            # Histogram: Question Lengths
            question_lengths = [len(q.split()) for q in questions]
            st.subheader("Histogram of Question Lengths")
            fig4, ax4 = plt.subplots(figsize=(8, 6))
            sns.histplot(question_lengths, kde=True, bins=30, color='blue', ax=ax4)
            ax4.set_title('Histogram of Question Lengths')
            ax4.set_xlabel('Length of Questions (Number of Words)')
            ax4.set_ylabel('Frequency')
            st.pyplot(fig4)

        else:
            st.error("Model is not loaded. Please check your model file.")
