import streamlit as st
import tensorflow as tf
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from tensorflow.keras.utils import pad_sequences
import pickle

# Download NLTK data
@st.cache_resource
def load_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')

load_nltk()


@st.cache_resource
def load_model_and_tokenizer():
    # Attempt to load without the complex function definitions first
    try:
        # Note the change to .keras extension
        model = tf.keras.models.load_model('summarizer_model.keras', compile=False)
        
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        
        return model, tokenizer
    except Exception as e:
        # If it still asks for NotEqual, we use this ultra-simple mapping
        st.info("Applying custom layer mapping...")
        custom_objects = {
            "NotEqual": tf.math.not_equal,
            "Any": tf.math.reduce_any
        }
        model = tf.keras.models.load_model(
            'summarizer_model.keras', 
            compile=False, 
            custom_objects=custom_objects
        )
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        return model, tokenizer


MAX_SENTS = 20
MAX_WORDS = 20
EMBEDDING_DIM = 100

# --- PREPROCESSING FUNCTIONS ---
def split_article(text, max_sents=30):
    text = text.replace('\n', ' ').strip()
    sentences = sent_tokenize(text)
    return sentences[:max_sents]


def create_3D_X_matrix(splitted_articles, tokenizer):
  X_data = np.zeros((len(splitted_articles), MAX_SENTS, MAX_WORDS), dtype='int32')
  for i, article_sentences in enumerate(splitted_articles):
      for j, sent in enumerate(article_sentences):
          if j < MAX_SENTS:
            word_tokens = tokenizer.texts_to_sequences([sent])[0]
            padded_tokens = pad_sequences([word_tokens], maxlen=MAX_WORDS, padding='post', truncating='post')[0]
            X_data[i, j, :] = padded_tokens
  return X_data


#---------------------------------------------------------APP-----------------------------------------------------------------


st.sidebar.header("Summary Settings")
num_sentences = st.sidebar.slider("Number of sentences in summary:", 3, 10, 5)


st.set_page_config(page_title="AI Summarizer")

st.title("Intelligent Summarizer")
st.markdown("""
This tool uses a **Hierarchical Bi-LSTM** with **GloVe Embeddings** to extract the most important sentences from any article.
""")

article_input = st.text_area("Paste your article here:", height=300, placeholder="Once upon a time in Vienna...")

if st.button("Generate Summary"):
    if article_input:
        with st.spinner('Analyzing text structure...'):
            try:
                model, tokenizer = load_model_and_tokenizer()
                
                # 1. Preprocess
                original_sentences = split_article(article_input)
                input_data = create_3D_X_matrix([article_input], tokenizer)
                
                # 2. Predict
                predictions = model.predict(input_data)[0]
                
                num_actual_sents = len(original_sentences)
                top_k = min(num_sentences, num_actual_sents)
                sent_scores = predictions[:num_actual_sents].flatten()
                
                # Get indices of top scores
                top_indices = np.argsort(sent_scores)[-top_k:]
                top_indices.sort()
                
                # 4. Display
                st.subheader("Summary")
                summary_text = ""
                for idx in top_indices:
                    summary_text += f"- {original_sentences[idx]}\n\n"
                
                st.success("Summary Generated!")
                st.markdown(summary_text)
                
            except Exception as e:
                st.error(f"Error: {e}. Make sure 'summarizer_model.h5' and 'tokenizer.pkl' are in the same folder.")
    else:
        st.warning("Please paste an article first.")
