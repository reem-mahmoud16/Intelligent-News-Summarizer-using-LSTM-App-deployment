#-------------------------------------------------Imports----------------------------------------------------------
# Text Processing
import nltk
from nltk.tokenize import sent_tokenize

# Data Handling
import numpy as np
import pandas as pd

# Deep Learning
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models, Model
from tensorflow.keras.regularizers import l2
import pickle

#-------------------------------------------------Downloading----------------------------------------------------------

nltk.download('punkt')
nltk.download('punkt_tab')

#-------------------------------------------------Loading data----------------------------------------------------------

train_df = pd.read_csv("train.csv", nrows=22000)
train_df = train_df.dropna(subset=['article', 'highlights'])

#--------------------------------------------Preprocessing functions----------------------------------------------------

def split_article(text, max_sents=30):
    text = text.replace('\n', ' ').strip()
    sentences = sent_tokenize(text)
    return sentences[:max_sents]

# 1. Initialize the Tokenizer
tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")

# "Fit" it on the original (unsplit) articles
tokenizer.fit_on_texts(train_df["article"])

MAX_SENTS = 20     # How many sentences per article
MAX_WORDS = 20     # How many words per sentence

def create_3D_X_matrix(splitted_articles):
  # 1. Initialize a 3D zero matrix
  X_data = np.zeros((len(splitted_articles), MAX_SENTS, MAX_WORDS), dtype='int32')

  # 2. Fill the matrix
  for i, article_sentences in enumerate(splitted_articles):
      for j, sent in enumerate(article_sentences):
          if j < MAX_SENTS:
              # Convert text sentence to list of integers
              word_tokens = tokenizer.texts_to_sequences([sent])[0]
              # Pad/Truncate the sentence to exactly MAX_WORDS (20)
              padded_tokens = pad_sequences([word_tokens], maxlen=MAX_WORDS, padding='post', truncating='post')[0]
              # Place the 1D list of words into the 3D grid
              X_data[i, j, :] = padded_tokens
  return X_data

def create_labels(article_sentences, highlight_text, threshold=0.5):
    # Convert highlights to lowercase words for comparison
    highlight_words = set(highlight_text.lower().split())
    labels = []

    for sent in article_sentences:
        sent_words = set(sent.lower().split())
        # Check how many words overlap
        overlap = len(sent_words.intersection(highlight_words)) / max(len(sent_words), 1)

        if overlap >= threshold:
            labels.append(1) # Important sentence!
        else:
            labels.append(0) # Not important

    return labels

def create_3D_Y_matrics(splitted_articles, highlights):
  # 1. Initialize Y matrix: (Total Articles, Max Sents, 1 Label per sent)
  Y_data = np.zeros((len(splitted_articles), MAX_SENTS, 1), dtype='float32')

  # 2. Loop through and fill the labels
  for i in range(len(splitted_articles)):
      # Get the split sentences of the current article
      article_sents = splitted_articles.iloc[i]
      # Get the highlight text for this article
      highlight_text = highlights.iloc[i]

      # Generate 1s and 0s for this specific article
      labels_list = create_labels(article_sents, highlight_text)

      # Inject into our matrix
      for j, label in enumerate(labels_list):
          if j < MAX_SENTS:
              Y_data[i, j, 0] = label
  return Y_data

#-------------------------------------------------Embedding with glove----------------------------------------------------------

glove_path = 'glove.6B.100d.txt'

embeddings_index = {}
with open(glove_path, encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs


EMBEDDING_DIM = 100 # Must match the GloVe file version you downloaded
vocab_size = len(tokenizer.word_index) + 1 # +1 for padding index 0

# Initialize matrix with zeros
embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))

for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in glove will remain all-zeros
        embedding_matrix[i] = embedding_vector

#-------------------------------------------------Train Data Preprocessing-------------------------------------------------

all_articles_split = train_df["article"].apply(lambda x: split_article(x, max_sents=MAX_SENTS))
X_data = create_3D_X_matrix(all_articles_split)
Y_data = create_3D_Y_matrics(all_articles_split, train_df['highlights'])

#-------------------------------------------------Model Building----------------------------------------------------------

# --- STEP 1: THE SENTENCE ENCODER (Word Level) ---
word_input = layers.Input(shape=(MAX_WORDS,), name="Word_Input")

word_embeddings = layers.Embedding(
    input_dim=vocab_size,
    output_dim=EMBEDDING_DIM,
    weights=[embedding_matrix],
    trainable=False,
    mask_zero=True,
    name="GloVe_Embedding"
)(word_input)

sent_vector = layers.LSTM(64, name="Word_LSTM")(word_embeddings)

sentence_encoder = Model(word_input, sent_vector, name="Sentence_Encoder")


# --- STEP 2: THE DOCUMENT ENCODER (Sentence Level) ---
doc_input = layers.Input(shape=(MAX_SENTS, MAX_WORDS), name="Doc_Input")

masking_layer = layers.Masking(mask_value=0.0)(doc_input)

# TimeDistributed applies the 'sentence_encoder' to every one of the 10 sentences
sent_embeddings = layers.TimeDistributed(sentence_encoder, name="Sentence_Sequencer")(masking_layer)

# Bidirectional allows the model to look forward and backward at sentences
contextualized_sent_vectors = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.3), name="Context_LSTM")(sent_embeddings)

# --- STEP 3: THE CLASSIFIER (Decision Level) ---
# For every sentence, we want a probability (0 to 1)
x = layers.TimeDistributed(layers.Dense(64, activation='relu', kernel_regularizer=l2(0.01)))(contextualized_sent_vectors)
x = layers.TimeDistributed(layers.Dropout(0.3))(x)
output_layer = layers.TimeDistributed(layers.Dense(1, activation='sigmoid'), name="Final_Classifier")(x)

# --- FINAL MODEL ASSEMBLY ---
model = Model(doc_input, output_layer)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#-------------------------------------------------Model Training----------------------------------------------------------

# 1. Create an array of weights (default everything to 1.0)
sample_weights = np.ones(shape=(Y_data.shape[0], Y_data.shape[1]))

# 2. Assign higher weight (e.g., 5.0) where the label is 1
sample_weights[Y_data.reshape(Y_data.shape[0], Y_data.shape[1]) == 1] = 5.0

history = model.fit(
    X_data,
    Y_data,
    epochs=5,
    batch_size=32,
    sample_weight=sample_weights, 
    validation_split=0.2
)

#-------------------------------------------------Loading Test Data----------------------------------------------------------

test_df = pd.read_csv("test.csv", nrows=4000)
test_df = test_df.dropna(subset=['article', 'highlights'])

#-------------------------------------------------Test Data Preprocessing-------------------------------------------------

all_test_articles_split = test_df["article"].apply(lambda x: split_article(x, max_sents=MAX_SENTS))
X_test = create_3D_X_matrix(all_test_articles_split)
Y_test = create_3D_Y_matrics(all_test_articles_split, test_df['highlights'])

#-------------------------------------------------Model Evaluating-------------------------------------------------

results = model.evaluate(X_test, Y_test)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

#---------------------------------------------------Model Saving----------------------------------------------------

model.save('summarizer_model.keras')

with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)