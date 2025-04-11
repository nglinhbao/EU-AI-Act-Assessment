import os
import re
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class ARMiner:
    def __init__(self):
        self.vectorizer = CountVectorizer(max_features=5000)
        self.classifier = MultinomialNB()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
    
    def load_training_data(self, app_name, data_dir='datasets'):
        """Load labeled and unlabeled training data for an app"""
        labeled_data = []
        labels = []
        
        # Load informative reviews (positive class)
        info_path = os.path.join(data_dir, app_name, 'trainL', 'info.txt')
        if os.path.exists(info_path):
            try:
                # Try with errors='ignore' to skip problematic characters
                with open(info_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        # Parse: [Number of tokens] [Rating] [Review text]
                        match = re.match(r'len\w+ rating\w+ (.*)', line.strip())
                        if match:
                            review_text = match.group(1)
                            labeled_data.append(review_text)
                            labels.append(1)  # 1 for informative
            except UnicodeDecodeError:
                # Fall back to latin-1 encoding if UTF-8 fails
                with open(info_path, 'r', encoding='latin-1') as f:
                    for line in f:
                        match = re.match(r'len\w+ rating\w+ (.*)', line.strip())
                        if match:
                            review_text = match.group(1)
                            labeled_data.append(review_text)
                            labels.append(1)  # 1 for informative
        
        # Load non-informative reviews (negative class)
        non_info_path = os.path.join(data_dir, app_name, 'trainL', 'non-info.txt')
        if os.path.exists(non_info_path):
            try:
                with open(non_info_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        # Parse: [Number of tokens] [Rating] [Review text]
                        match = re.match(r'len\w+ rating\w+ (.*)', line.strip())
                        if match:
                            review_text = match.group(1)
                            labeled_data.append(review_text)
                            labels.append(0)  # 0 for non-informative
            except UnicodeDecodeError:
                with open(non_info_path, 'r', encoding='latin-1') as f:
                    for line in f:
                        match = re.match(r'len\w+ rating\w+ (.*)', line.strip())
                        if match:
                            review_text = match.group(1)
                            labeled_data.append(review_text)
                            labels.append(0)  # 0 for non-informative
        
        # Load unlabeled data
        unlabeled_data = []
        unlabeled_path = os.path.join(data_dir, app_name, 'trainU')
        if os.path.exists(unlabeled_path):
            for filename in os.listdir(unlabeled_path):
                file_path = os.path.join(unlabeled_path, filename)
                if os.path.isfile(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            for line in f:
                                match = re.match(r'len\w+ rating\w+ (.*)', line.strip())
                                if match:
                                    review_text = match.group(1)
                                    unlabeled_data.append(review_text)
                    except UnicodeDecodeError:
                        with open(file_path, 'r', encoding='latin-1') as f:
                            for line in f:
                                match = re.match(r'len\w+ rating\w+ (.*)', line.strip())
                                if match:
                                    review_text = match.group(1)
                                    unlabeled_data.append(review_text)
        
        return labeled_data, labels, unlabeled_data
        
    def train(self, app_names, data_dir='datasets', max_iterations=10):
        """Train the classifier using labeled and unlabeled data"""
        all_labeled_data = []
        all_labels = []
        all_unlabeled_data = []
        
        print("[INFO] Starting training process...")
        
        # Load data from all specified apps
        for app_name in app_names:
            print(f"[INFO] Loading data for app: {app_name}")
            labeled_data, labels, unlabeled_data = self.load_training_data(app_name, data_dir)
            print(f"[DEBUG] Labeled samples: {len(labeled_data)}, Unlabeled samples: {len(unlabeled_data)}")
            all_labeled_data.extend(labeled_data)
            all_labels.extend(labels)
            all_unlabeled_data.extend(unlabeled_data)
        
        # Preprocess all data
        print("[INFO] Preprocessing labeled data...")
        processed_labeled = [self.preprocess_text(text) for text in all_labeled_data]
        print("[INFO] Preprocessing unlabeled data...")
        processed_unlabeled = [self.preprocess_text(text) for text in all_unlabeled_data]
        
        # Fit the vectorizer on all data (labeled + unlabeled)
        print("[INFO] Fitting vectorizer on combined data...")
        self.vectorizer.fit(processed_labeled + processed_unlabeled)
        
        # Transform the labeled data
        print("[INFO] Vectorizing labeled data...")
        X_labeled = self.vectorizer.transform(processed_labeled)
        y_labeled = np.array(all_labels)
        
        # Create initial classifier using labeled data
        print("[INFO] Training initial classifier on labeled data...")
        self.classifier.fit(X_labeled, y_labeled)
        
        # If we have unlabeled data, use EM algorithm
        if processed_unlabeled:
            print("[INFO] Starting Expectation-Maximization iterations...")
            X_unlabeled = self.vectorizer.transform(processed_unlabeled)
            
            for iteration in range(1):
                print(f"[INFO] EM Iteration {iteration + 1}/{max_iterations}")
                
                # E-step: Estimate labels for unlabeled data
                unlabeled_proba = self.classifier.predict_proba(X_unlabeled)
                unlabeled_labels = np.argmax(unlabeled_proba, axis=1)
                print(f"[DEBUG] Sample predicted labels (first 10): {unlabeled_labels[:10]}")
                
                # M-step: Refit the model with all data
                X_all = np.vstack((X_labeled.toarray(), X_unlabeled.toarray()))
                y_all = np.concatenate((y_labeled, unlabeled_labels))
                print(f"[DEBUG] Combined data size: {X_all.shape}, Labels size: {len(y_all)}")
                
                self.classifier.fit(X_all, y_all)
        
        # Calculate training accuracy
        print("[INFO] Evaluating training accuracy...")
        y_pred = self.classifier.predict(X_labeled)
        accuracy = np.mean(y_pred == y_labeled)
        print(f"[RESULT] Training accuracy: {accuracy:.4f}")
        print(classification_report(y_labeled, y_pred))
        
        print("[INFO] Training process completed.")
        return self

    
    def save_model(self, filepath):
        """Save the trained model to a file"""
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """Load a trained model from a file"""
        model = cls()
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        model.vectorizer = model_data['vectorizer']
        model.classifier = model_data['classifier']
        return model

    def preprocess_text(self, text):
        """Preprocess the text as described in AR-Miner paper"""
        # Handle non-string inputs (NaN, None, etc.)
        if not isinstance(text, str):
            if pd.isna(text):  # Check if NaN or None
                return ""
            else:
                # Convert other non-string values to string
                text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove non-alphanumeric symbols (except spaces)
        text = re.sub(r'[^\w\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stop words and apply stemming
        processed_tokens = [
            self.stemmer.stem(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 1
        ]
        
        # Return processed text
        return ' '.join(processed_tokens)

    def predict(self, reviews):
        """Predict if reviews are informative or not"""
        processed_reviews = [self.preprocess_text(review) for review in reviews]
        X = self.vectorizer.transform(processed_reviews)
        predictions = self.classifier.predict(X)
        probabilities = self.classifier.predict_proba(X)[:, 1]  # Probability of being informative
        
        return predictions, probabilities

    def filter_reviews(self, content_series):
        """
        Filter reviews to only include informative ones
        
        Parameters:
        content_series - pandas Series containing review text
        
        Returns:
        tuple - (is_informative, probabilities) where:
          - is_informative: boolean Series indicating if each review is informative
          - probabilities: Series with probability scores for each review
        """
        # Handle NaN values and convert to strings
        content_series = content_series.fillna("").astype(str)
        
        # Get predictions and probabilities
        predictions, probabilities = self.predict(content_series.tolist())
        
        # Return as Series with same index as input
        return pd.Series(predictions, index=content_series.index).astype(bool), pd.Series(probabilities, index=content_series.index)
    