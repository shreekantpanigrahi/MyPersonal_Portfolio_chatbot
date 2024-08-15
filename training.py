import os
import random
import json
import pickle
import numpy as np
import gensim.downloader as api

import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# Initialize lemmatizer and load intents
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', ',', '.']

# Tokenize and lemmatize each word in patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

# Save words and classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Set embedding dimension
embedding_dim = 300  # GloVe model dimension size

# Check if the embedding matrix already exists
if os.path.exists('embedding_matrix.pkl'):
    embedding_matrix = pickle.load(open('embedding_matrix.pkl', 'rb'))
else:
    # Load the pre-trained Word2Vec model
    word2vec_model = api.load("glove-wiki-gigaword-300")
    embedding_matrix = np.zeros((len(words), embedding_dim))
    
    for i, word in enumerate(words):
        if word in word2vec_model.key_to_index:
            embedding_matrix[i] = word2vec_model[word]
        else:
            embedding_matrix[i] = np.random.normal(size=(embedding_dim,))
    
    # Save the embedding matrix for future use
    pickle.dump(embedding_matrix, open('embedding_matrix.pkl', 'wb'))

# Vectorize the sentences in the documents
vector_cache = {}

def vectorize_sentence(sentence):
    sentence_key = tuple(sentence)
    if sentence_key in vector_cache:
        return vector_cache[sentence_key]
    
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence]
    sentence_vec = np.zeros((embedding_dim,))
    for word in sentence_words:
        if word in word2vec_model.key_to_index:
            sentence_vec += word2vec_model[word]
        else:
            sentence_vec += np.random.normal(size=(embedding_dim,))
    
    vector_cache[sentence_key] = sentence_vec / len(sentence_words)
    return vector_cache[sentence_key]

# Prepare the training data
training = []
output_empty = [0] * len(classes)

for document in documents:
    vector = vectorize_sentence(document[0])
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([vector, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Build the neural network model
model = Sequential()
model.add(Dense(128, input_shape=(embedding_dim,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='softmax'))

sgd = SGD(learning_rate=0.01, weight_decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
model.fit(train_x, train_y, epochs=100, batch_size=16, verbose=1)

# Save the trained model
model.save('chatbot_model.keras', include_optimizer=False)
print('Done')


