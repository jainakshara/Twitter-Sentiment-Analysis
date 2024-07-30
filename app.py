# app.py

import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

# Download stopwords if needed
nltk.download('stopwords')

# Load the sentiment analysis model and vectorizer
with open('D:\\Twitter Sentiment Analysis\\trained_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('D:\\Twitter Sentiment Analysis\\vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Initialize PorterStemmer
port_stem = PorterStemmer()

# Define preprocessing function
def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [port_stem.stem(word) for word in text if not word in stopwords.words('english')]
    text = ' '.join(text)
    return text

# Define a function to predict sentiment
def predict_sentiment(text):
    processed_text = preprocess(text)
    transformed_text = vectorizer.transform([processed_text])
    prediction = model.predict(transformed_text)
    return prediction[0]

# Streamlit app layout
st.set_page_config(page_title="Sentiment Analysis", page_icon=":speech_balloon:", layout="centered")

# App Title with Styling
st.markdown("""
    <style>
    .title {
        font-size: 40px;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 20px;
    }
    .description {
        font-size: 20px;
        color: #34495e;
        text-align: center;
        margin-bottom: 30px;
    }
    .text-input {
        font-size: 18px;
        padding: 10px;
        border-radius: 5px;
        border: 2px solid #3498db;
        margin-bottom: 20px;
    }
    .button {
        background-color: #3498db;
        color: white;
        font-size: 18px;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        cursor: pointer;
    }
    .button:hover {
        background-color: #2980b9;
    }
    .result {
        font-size: 22px;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
    }
    </style>
    <div class="title">Sentiment Analysis</div>
    <div class="description">Enter a sentence to analyze its sentiment.</div>
    """, unsafe_allow_html=True)

# User input and analysis
user_input = st.text_input('', '', placeholder='Type your text here...', key='text_input', help='Enter text to analyze sentiment', label_visibility='collapsed', )

if st.button('Analyze', key='analyze_button'):
    if user_input:
        sentiment = predict_sentiment(user_input)
        sentiment_label = 'Positive' if sentiment == 1 else 'Negative'
        st.markdown(f'<div class="result">Sentiment: {sentiment_label}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result">Please enter some text for analysis.</div>', unsafe_allow_html=True)
