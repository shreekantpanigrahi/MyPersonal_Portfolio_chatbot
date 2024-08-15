import os
import random
import json
import pickle
import time
import numpy as np
import streamlit as st
import pdfplumber
from PIL import Image
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import gensim.downloader as api

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Load the model
model = load_model('chatbot_model.keras')

# Load Word2Vec model
if not os.path.exists('word2vec_model.pkl'):
    word2vec_model = api.load("glove-wiki-gigaword-300")
    pickle.dump(word2vec_model, open('word2vec_model.pkl', 'wb'))
else:
    word2vec_model = pickle.load(open('word2vec_model.pkl', 'rb'))

embedding_dim = 300  # GloVe embedding size
vector_cache = {}

def vectorize_sentence(sentence):
    sentence_key = tuple(sentence)
    if sentence_key in vector_cache:
        return vector_cache[sentence_key]
    
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    sentence_vec = np.zeros((embedding_dim,))
    
    for word in sentence_words:
        if word in word2vec_model.key_to_index:
            sentence_vec += word2vec_model[word]
        else:
            sentence_vec += np.random.normal(size=(embedding_dim,))
    
    vector_cache[sentence_key] = sentence_vec / len(sentence_words)
    return vector_cache[sentence_key]

def predict_classes(sentence):
    vec = vectorize_sentence(sentence)
    res = model.predict(np.array([vec]), verbose=0)[0]
    ERROR_THRESHOLD = 0.07
    results = [(i, r) for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]

def get_response(intents_list, intents_json):
    if intents_list:
        tag = intents_list[0]['intent']
        for i in intents_json['intents']:
            if i['tag'] == tag:
                return random.choice(i['responses'])
    return "Sorry, I didn't understand that. Can you please rephrase?"

# Initialize the conversation in session state if it doesn't exist
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Streamlit App Layout
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Chatbot", "Resume", "Internships", "Certifications", "Contact"])

with tab1:
    st.title("My Personal Portfolio Chatbot")
    st.write("Ask me anything about my skills, education, and experience!")

    user_input = st.text_input("You: ")

    if st.button("Send"):
        if user_input:
            with st.spinner("Chatbot is typing..."):
                time.sleep(1)  # Reduced delay
                ints = predict_classes(user_input)
                res = get_response(ints, intents)
                st.session_state.conversation.append({"user": user_input, "bot": res})
                st.write(f"Chatbot: {res}")
    
    st.write("### Chat History")
    for chat in st.session_state.conversation:
        st.write(f"**You:** {chat['user']}")
        st.write(f"**Chatbot:** {chat['bot']}")
        st.write("---")

with tab2:
    st.title("Resume")
    st.subheader("Download my resume from below")
    with open("Images/Resume.pdf", "rb") as file:
        st.download_button(
            label="Download Resume as PDF",
            data=file,
            file_name="Shreekant_Panigrahi_Resume.pdf",
            mime="application/pdf"
        )

    st.write("### Resume Preview")
    with pdfplumber.open("Images/Resume.pdf") as pdf:
        page = pdf.pages[0]
        image = page.to_image(resolution=250)  # Reduced resolution for faster load
        st.image(image.original)

with tab3:
    st.title("Internships")
    st.subheader("Visuals of Internship certificates are given below")
    st.write("### Preview")
    # Render only the first page of each PDF for preview
    for pdf_name in ["Images/DA.pdf", "Images/DS.pdf", "Images/Gen_AI.pdf"]:
        with pdfplumber.open(pdf_name) as pdf:
            page = pdf.pages[0]
            image = page.to_image(resolution=200)
            st.image(image.original)

with tab4:
    st.title("Certification")
    st.subheader("Visuals of certificates are given below")
    st.write("### Preview")
    with pdfplumber.open("Images/Big_Data.pdf") as pdf:
        page = pdf.pages[0]
        image = page.to_image(resolution=200)
        st.image(image.original)
    st.write("---")
    st.write("Badge of Big Data Level 1")
    st.image("Images/Badge.png")

with tab5:
    st.title("Contact Developer")

    st.write("Feel free to reach out to me through the following platforms:")
    st.write("---")
    st.subheader("Email")
    st.write("[Email](mailto:your.light@gmail.com)")
    st.write("---")
    st.subheader("GitHub")
    st.write("[GitHub Profile](https://github.com/shreekantpanigrahi)")
    st.write("---")
    st.subheader("LinkedIn")
    st.write("[LinkedIn Profile](https://www.linkedin.com/in/shreekant-panigrahi-b570b2232)")
    st.write("---")
    st.subheader("X (formerly Twitter)")
    st.write("[X Profile](https://twitter.com/Shreekant357?t=QC4V_vFvLrO0_b-KKAV1Yg&s=09)")
    st.write("---")
    st.subheader("Linktree")
    st.write("[Linktree Profile](https://linktr.ee/Shreekant_02)")

    st.write("---")
    st.write("Thanks for visiting here!")
