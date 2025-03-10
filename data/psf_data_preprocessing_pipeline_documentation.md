# E-commerce Data Preprocessing Pipeline Documentation

## Overview
This python file implements a comprehensive data preprocessing pipeline for e-commerce product data. The pipeline transforms raw CSV data into PyTorch tensors suitable for machine learning models.

## Data Structure
The input data should be a CSV file containing the following fields:
- `price_ori`: Original price
- `delivery`: Delivery information
- `item_category_detail`: Product category
- `specification`: Product specifications
- `title`: Product title
- `w_date`: Date information
- `link_ori`: Original product link
- `item_rating`: Product rating
- `seller_name`: Seller information
- `idElastic`: Elastic search ID
- `price_actual`: Actual price
- `sitename`: E-commerce platform name
- `idHash`: Hash ID
- `total_rating`: Total number of ratings
- `id`: Product ID
- `total_sold`: Total units sold
- `pict_link`: Product image link
- `favorite`: Number of favorites
- `timestamp`: Timestamp
- `desc`: Product description

## Pipeline Components

### 1. Data Loading and Import Statements

### 2. Missing Data Handling

- Handles missing values in numeric columns by filling with median values
- Handles missing values in categorical columns by filling with mode values
- Returns: DataFrame with no missing values

### 3. Timestamp Feature Extraction

Extracts temporal features from the timestamp:
- Hour of day (0-23)
- Day of month (1-31)
- Month (1-12)
- Year
- Day of week (0-6)

### 4. Categorical Encoding

- Transforms categorical variables into numerical format
- Uses LabelEncoder for:
  - delivery
  - item_category_detail
  - seller_name
  - sitename
- Creates new columns with '_encoded' suffix
- Returns: DataFrame and dictionary of label encoders

### 5. Text Feature Transformation

- Combines 'title' and 'desc' fields
- Applies TF-IDF vectorization
- Parameters:
  - max_features: 100
  - stop_words: English
- Returns: DataFrame with text features and TF-IDF vectorizer

### 6. Numerical Feature Normalization

- Standardizes numerical features using StandardScaler
- Processes: price_ori, price_actual, item_rating, total_rating, total_sold, favorite
- Returns: Normalized DataFrame and scaler object

### 7. PyTorch Tensor Conversion

- Combines all processed features:
  - Numerical features
  - Encoded categorical features
  - TF-IDF text features
- Converts to PyTorch FloatTensor
- Returns: Tensor and feature names

### 8. Main Pipeline Function

Orchestrates all preprocessing steps in sequence and returns:
- Processed data tensor (X)
- Feature names list
- Dictionary of preprocessors

## Output Files
The pipeline saves three files:
1. `preprocessed_data.pt`: PyTorch tensor containing processed features
2. `feature_names.pt`: List of feature names
3. `preprocessors.pt`: Dictionary containing:
   - Label encoders
   - TF-IDF vectorizer
   - Standard scaler

## Usage Instructions
1. CSV file is in pytorch-sales-forecasting/data/raw/20240121_shopee_data.csv

## Requirements
- Python 3.x
- pandas
- numpy
- scikit-learn
- PyTorch
- re (regular expressions)

## Notes
- The pipeline is modular and can be modified for different feature sets
- TF-IDF max_features can be adjusted based on needs
- Additional preprocessing steps can be added to the main pipeline function
- Preprocessors are saved for consistent transformation of new data
