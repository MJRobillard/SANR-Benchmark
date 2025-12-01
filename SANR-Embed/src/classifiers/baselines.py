import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score
import joblib
import os

class BaselineClassifier:
    """
    Implements Tier 1 Baseline: Logistic Regression with TF-IDF.
    """
    def __init__(self, max_features=5000):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))),
            ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, solver='lbfgs', multi_class='multinomial'))
        ])
        self.model_path = None
        
    def train(self, train_df, text_col='text_original', label_col='label_primary'):
        print(f"Training Logistic Regression on {len(train_df)} samples using column '{text_col}'...")
        
        # Handle NaN
        X = train_df[text_col].fillna("")
        y = train_df[label_col]
        
        # Filter out classes with very few samples if necessary, but LogisticRegression handles it okay usually
        # (though validation split might fail if not stratified properly before)
        
        self.pipeline.fit(X, y)
        print("Training complete.")
        
    def predict(self, test_df, text_col='text_original'):
        X = test_df[text_col].fillna("")
        return self.pipeline.predict(X)
        
    def evaluate(self, test_df, text_col='text_original', label_col='label_primary'):
        print(f"Evaluating on {len(test_df)} samples using column '{text_col}'...")
        preds = self.predict(test_df, text_col)
        y_true = test_df[label_col]
        
        # Filter out rows where label is NaN
        mask = y_true.notna()
        y_true = y_true[mask]
        preds = preds[mask]
        
        if len(y_true) == 0:
            return {'f1_macro': 0.0, 'report': {}}

        report = classification_report(y_true, preds, output_dict=True, zero_division=0)
        f1 = f1_score(y_true, preds, average='macro', zero_division=0)
        
        return {
            'f1_macro': f1,
            'report': report,
            'predictions': preds
        }
    
    def save(self, path):
        joblib.dump(self.pipeline, path)
        self.model_path = path
        print(f"Model saved to {path}")
        
    def load(self, path):
        self.pipeline = joblib.load(path)
        self.model_path = path
        print(f"Model loaded from {path}")

class RandomClassifier:
    """
    Baseline random classifier based on class distribution
    """
    def train(self, train_df, text_col='text_original', label_col='label_primary'):
        self.classes_ = train_df[label_col].value_counts(normalize=True)
        
    def predict(self, test_df, text_col='text_original'):
        return np.random.choice(self.classes_.index, size=len(test_df), p=self.classes_.values)
        
    def evaluate(self, test_df, text_col='text_original', label_col='label_primary'):
        preds = self.predict(test_df, text_col)
        y_true = test_df[label_col]
        f1 = f1_score(y_true, preds, average='macro', zero_division=0)
        return {'f1_macro': f1}


