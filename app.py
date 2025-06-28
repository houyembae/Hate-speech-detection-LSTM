import streamlit as st
import keras
import pickle
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


model = keras.models.load_model("best_model.keras")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

vocab_size = 10000
max_len = 20


def clean_text(text):
    text = text.lower()
    text = re.sub(r'@[\w]+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Preprocessing function
def preprocess(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    return padded

# Streamlit UI
st.set_page_config(page_title="Hate Speech Classifier", page_icon="🧠")
st.title("Hate Speech Classifier")
st.markdown("Enter a message to detect whether it contains **hate speech**, **offensive language**, or is **neutral**.")

input_text = st.text_area("✏️ Enter your text here:", height=150)

if st.button("Classify"):
    if input_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        x_input = preprocess(input_text)
        prediction = model.predict(x_input)
        label_map = {0: "Hate Speech", 1: "Offensive Language", 2: "Neutral"}
        label = np.argmax(prediction, axis=1)[0]
        st.success(f"🔍 Predicted: **{label_map[label]}**")
