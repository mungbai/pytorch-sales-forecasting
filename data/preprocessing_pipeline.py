import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import re
from datetime import datetime
import pickle
import os

class EcommercePreprocessingPipeline:
    def __init__(self):
        self.label_encoders = {}
        self.tfidf_vectorizer = None
        self.scaler = None
        
    def load_data(self, file_path):
        """Load data from CSV file"""
        return pd.read_csv(file_path)
    
    def handle_missing_data(self, df):
        """Handle missing values in the dataset"""
        numeric_columns = ['price_ori', 'price_actual', 'item_rating', 
                         'total_rating', 'total_sold', 'favorite']
        categorical_columns = ['delivery', 'item_category_detail', 
                             'seller_name', 'sitename']
        
        # Fill numeric columns with median
        for col in numeric_columns:
            df[col] = df[col].fillna(df[col].median())
            
        # Fill categorical columns with mode
        for col in categorical_columns:
            df[col] = df[col].fillna(df[col].mode()[0])
            
        return df
    
    def extract_timestamp_features(self, df):
        """Extract temporal features from timestamp"""
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['year'] = df['timestamp'].dt.year
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        return df
    
    def encode_categorical_features(self, df):
        """Encode categorical variables"""
        categorical_columns = ['delivery', 'item_category_detail', 
                             'seller_name', 'sitename']
        
        for col in categorical_columns:
            self.label_encoders[col] = LabelEncoder()
            df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])
            
        return df
    
    def transform_text_features(self, df):
        """Transform text features using TF-IDF"""
        # Combine title and description
        df['text_combined'] = df['title'].fillna('') + ' ' + df['desc'].fillna('')
        
        # Initialize and fit TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english'
        )
        text_features = self.tfidf_vectorizer.fit_transform(df['text_combined'])
        
        # Convert to DataFrame
        text_feature_names = [f'text_feature_{i}' for i in range(100)]
        text_features_df = pd.DataFrame(
            text_features.toarray(),
            columns=text_feature_names
        )
        
        return pd.concat([df, text_features_df], axis=1)
    
    def normalize_numerical_features(self, df):
        """Normalize numerical features"""
        numerical_columns = ['price_ori', 'price_actual', 'item_rating',
                           'total_rating', 'total_sold', 'favorite']
        
        self.scaler = StandardScaler()
        df[numerical_columns] = self.scaler.fit_transform(df[numerical_columns])
        
        return df
    
    def convert_to_tensor(self, df):
        """Convert processed features to PyTorch tensor"""
        # Select features for the final tensor
        feature_columns = (
            ['price_ori', 'price_actual', 'item_rating', 'total_rating', 
             'total_sold', 'favorite', 'hour', 'day', 'month', 'year', 
             'day_of_week'] +
            [col for col in df.columns if col.endswith('_encoded')] +
            [col for col in df.columns if col.startswith('text_feature_')]
        )
        
        # Convert to tensor
        X = torch.FloatTensor(df[feature_columns].values)
        
        return X, feature_columns
    
    def process_data(self, file_path):
        """Main pipeline function"""
        # Load data
        df = self.load_data(file_path)
        
        # Apply preprocessing steps
        df = self.handle_missing_data(df)
        df = self.extract_timestamp_features(df)
        df = self.encode_categorical_features(df)
        df = self.transform_text_features(df)
        df = self.normalize_numerical_features(df)
        
        # Convert to tensor
        X, feature_names = self.convert_to_tensor(df)
        
        # Create preprocessors dictionary
        preprocessors = {
            'label_encoders': self.label_encoders,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'scaler': self.scaler
        }
        
        # Create processed directory if it doesn't exist
        os.makedirs('data/processed', exist_ok=True)
        
        # Save outputs to processed directory
        torch.save(X, 'data/processed/preprocessed_data.pt')
        torch.save(feature_names, 'data/processed/feature_names.pt')
        torch.save(preprocessors, 'data/processed/preprocessors.pt')
        
        return X, feature_names, preprocessors

if __name__ == "__main__":
    # Initialize pipeline
    pipeline = EcommercePreprocessingPipeline()
    
    # Process data
    file_path = 'data/raw/20240121_shopee_data.csv'
    X, feature_names, preprocessors = pipeline.process_data(file_path)
    
    print(f"Processing complete. Output shape: {X.shape}")
    print(f"Number of features: {len(feature_names)}") 