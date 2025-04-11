import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow_hub as hub
from scipy.sparse import hstack, csr_matrix
from gensim.models import KeyedVectors
from tqdm import tqdm
import joblib
import pandas as pd
import tensorflow as tf

def load_models(model_path='/content/drive/MyDrive/EU-AI-Act/models/Model-LR_TUW.pkl', word2vec_path='/content/drive/MyDrive/EU-AI-Act/models/GoogleNews-vectors-negative300.bin'):
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
    vectorizer = joblib.load('/content/drive/MyDrive/EU-AI-Act/models/vectorizer.pkl')
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

def evaluate(file_path):
    tf.config.set_visible_devices([], 'GPU')
    """Predict fairness scores and calculate accuracy."""
    
    # Load CSV
    df = pd.read_csv(file_path)
    
    # Drop or fill NaNs in 'content'
    df['content'] = df['content'].fillna('')  # or use df.dropna(subset=['content']) if you prefer dropping
    
    # Load models
    word2vec_model, word2vec_words, classifier_model, use_model = load_models()
    
    # Process features
    x_processed = combine_features(
        df['content'].tolist(),
        word2vec_model,
        word2vec_words,
        use_model
    )
    
    # Get probability predictions
    fairness_scores = classifier_model.predict_proba(x_processed)[:, 1]
    
    # Add Fairness score to df
    df['Fairness Score'] = fairness_scores

    # Save to the same file
    df.to_csv(file_path, index=False)
    
    return df

def predict_fairness_single_text(text, 
                                  model_path='/content/drive/MyDrive/EU-AI-Act/models/Model-LR_TUW.pkl',
                                  word2vec_path='/content/drive/MyDrive/EU-AI-Act/models/GoogleNews-vectors-negative300.bin',
                                  vectorizer_path='/content/drive/MyDrive/EU-AI-Act/models/vectorizer.pkl'):
    """Predict fairness score for a single text input."""
    # Ensure TF doesn't allocate GPU (optional if you're using CPU only)
    tf.config.set_visible_devices([], 'GPU')

    # Load models
    print("Loading models for single text prediction...")
    word2vec_model = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    word2vec_words = list(word2vec_model.key_to_index)
    classifier_model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    # Transform input text
    print("Transforming features for the input text...")
    tfidf_feature = vectorizer.transform([text])
    use_feature = transform_use([text], use_model)
    w2v_feature = transform_word2vec([text], word2vec_model, word2vec_words)

    # Combine features
    combined_features = hstack((tfidf_feature, use_feature, w2v_feature))

    # Predict fairness score
    fairness_score = classifier_model.predict_proba(combined_features)[:, 1][0]

    return fairness_score

def evaluate_app_fairness(app_name, word2vec_model, word2vec_words, classifier_model, 
    use_model, file_path="/content/drive/MyDrive/EU-AI-Act/datasets/ai_app_reviews.csv"):
    """
    Calculate fairness scores for the given app and return the average score.
    
    Parameters:
    - file_path: Path to the CSV file containing app reviews.
    - app_name: Name of the app to filter reviews.
    
    Returns:
    - average_fairness_score: Average fairness score for the specified app.
    """
    # Ensure TensorFlow does not allocate GPU (optional)
    # tf.config.set_visible_devices([], 'GPU')
    
    # Load CSV
    df = pd.read_csv(file_path)
    
    # Filter reviews for the specified app
    app_reviews = df[df['app_name'] == app_name].copy()
    
    if app_reviews.empty:
        print(f"No reviews found for app: {app_name}")
        return None
    
    # Fill NaNs in 'content'
    app_reviews['content'] = app_reviews['content'].fillna('')
    
    # Process features
    x_processed = combine_features(
        app_reviews['content'].tolist(),
        word2vec_model,
        word2vec_words,
        use_model
    )
    
    # Get probability predictions
    fairness_scores = classifier_model.predict_proba(x_processed)[:, 1]
    
    # Add Fairness score to app_reviews DataFrame
    app_reviews['Fairness Score'] = fairness_scores
    
    # Calculate average fairness score
    average_fairness_score = app_reviews['Fairness Score'].mean()
    
    print(f"Average Fairness Score for '{app_name}': {average_fairness_score:.4f}")
    
    return average_fairness_score