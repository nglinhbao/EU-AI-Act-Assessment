import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow_hub as hub
from scipy.sparse import hstack, csr_matrix
from gensim.models import KeyedVectors
from tqdm import tqdm
import joblib
import pandas as pd
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

def load_models(model_path='./models/Model-LR_TUW.pkl', word2vec_path='./models/GoogleNews-vectors-negative300.bin'):
    """Load all required models."""
    print("Loading Word2Vec model...")
    word2vec_model = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    word2vec_words = list(word2vec_model.key_to_index)
    
    print("Loading classifier model...")
    classifier_model = joblib.load(model_path)
    
    print("Loading Universal Sentence Encoder...")
    use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    
    return word2vec_model, word2vec_words, classifier_model, use_model

def transform_tfidf_char(texts):
    """Transform texts using character-level TF-IDF."""
    vectorizer = joblib.load('models/vectorizer.pkl')
    return vectorizer.transform(texts)

def transform_use(texts, use_model):
    """Transform texts using Universal Sentence Encoder."""
    embeddings = use_model(texts)
    return csr_matrix(embeddings.numpy())

def transform_word2vec(texts, word2vec_model, word2vec_words):
    """Transform texts using Word2Vec averaging."""
    sent_vectors = []
    
    for text in tqdm(texts, desc="Processing Word2Vec"):
        words = text.split()
        sent_vec = np.zeros(300)
        count_words = 0
        
        for word in words:
            if word in word2vec_words:
                word_vectors = word2vec_model[word]
                sent_vec += word_vectors
                count_words += 1
        
        if count_words != 0:
            sent_vec /= count_words
        sent_vectors.append(sent_vec)
    
    return csr_matrix(np.array(sent_vectors))

def combine_features(texts, word2vec_model, word2vec_words, use_model):
    """Combine all feature transformations."""
    print("Starting feature transformation...")
    
    tfidf_features = transform_tfidf_char(texts)
    print("TF-IDF features completed")
    
    use_features = transform_use(texts, use_model)
    print("USE features completed")
    
    w2v_features = transform_word2vec(texts, word2vec_model, word2vec_words)
    print("Word2Vec features completed")
    
    return hstack((tfidf_features, use_features, w2v_features))

def evaluate(df):
    """Predict fairness scores and calculate accuracy."""
    # Load models
    word2vec_model, word2vec_words, classifier_model, use_model = load_models()
    
    # Process features
    x_processed = combine_features(
        df['Reason'].tolist(),
        word2vec_model,
        word2vec_words,
        use_model
    )
    
    # Get probability predictions
    fairness_scores = classifier_model.predict_proba(x_processed)[:, 1]
    
    # Calculate accuracy
    if 'Predicted System Type' in df.columns and 'System Type' in df.columns:
        accuracy = (df['Predicted System Type'] == df['System Type']).mean()
        print(f"Accuracy: {accuracy:.4f}")
    else:
        accuracy = None
    
    # Add Fairness score and accuracy columns to df
    df['Fairness Score'] = fairness_scores
    df['Accuracy'] = accuracy

    df.to_csv('datasets/results.csv', index=False)
    
    return df

evaluate(pd.read_csv('datasets/results.csv'))