# Question Difficulty Classifier

## Project Overview
This project is a **Question Difficulty Classifier** that utilizes **Logistic Regression** to predict the difficulty level of a given question. The model is trained using a Kaggle dataset containing labeled questions categorized into three difficulty levels: **Easy, Medium, and Hard**. A **Streamlit web application** is also provided to allow users to upload text documents and classify question difficulties interactively.

## Features
- **Data Preprocessing**: Cleans and structures the dataset for model training.
- **Model Training**: Implements a **TF-IDF vectorizer** and **Logistic Regression** pipeline.
- **Hyperparameter Tuning**: Uses **GridSearchCV** to optimize model parameters.
- **Visualization**: Generates data distributions and confusion matrices.
- **Web Application**: Implements a **Streamlit-based UI** for real-time classification.
- **Support for Multiple File Formats**: Supports `.txt`, `.docx`, and `.pdf` file uploads.

## Dataset
The dataset used in this project is sourced from Kaggle:
[Question Answer Dataset](https://www.kaggle.com/datasets/rtatman/questionanswer-dataset)

The dataset consists of **questions labeled by difficulty**. The final refined dataset achieved **90.25% accuracy** with **Logistic Regression**.

### Dataset Challenges:
- Overfitting and generalization issues in previous models (Random Forest: **47%**, LSTM: **34%**).
- The refined dataset improved accuracy significantly.

## Installation
### Prerequisites
Ensure you have **Python 3.7+** installed along with the required libraries.

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Required Libraries
- `streamlit`
- `scikit-learn`
- `seaborn`
- `matplotlib`
- `numpy`
- `pandas`
- `pickle`
- `docx`
- `PyPDF2`
- `wordcloud`

## Training the Model
To train the model and generate a classifier:
```bash
final_visualization.ipynb
```
This will:
1. Preprocess the dataset.
2. Train the Logistic Regression model.
3. Save the trained model as `question_classifier.pkl`.

## Running the Streamlit App
Start the **web application** using:
```bash
streamlit run streamlit_final.py
```
Then upload a **.txt, .docx, or .pdf** file to classify questions.

## Model Evaluation
The final Logistic Regression model achieved **90.25% accuracy** on the Kaggle dataset. The confusion matrix visualizes classification performance.

## Future Improvements
- **Expand the dataset** for better generalization.
- **Experiment with deep learning models** like transformers.
- **Enhance UI** for a more interactive experience.
