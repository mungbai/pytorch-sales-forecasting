"""
Configuration file defining features used in the preprocessing pipeline.
"""

NUMERICAL_FEATURES = [
    'price_ori',      # Original price
    'price_actual',   # Actual price
    'item_rating',    # Product rating
    'total_rating',   # Total number of ratings
    'total_sold',     # Total units sold
    'favorite'        # Number of favorites
]

CATEGORICAL_FEATURES = [
    'delivery',              # Delivery information
    'item_category_detail',  # Product category
    'seller_name',          # Seller information
    'sitename'              # E-commerce platform name
]

TEXT_FEATURES = [
    'title',  # Product title
    'desc'    # Product description
]

TIMESTAMP_FEATURES = [
    'hour',         # Hour of day (0-23)
    'day',          # Day of month (1-31)
    'month',        # Month (1-12)
    'year',         # Year
    'day_of_week'   # Day of week (0-6)
]

# TF-IDF configuration
TFIDF_CONFIG = {
    'max_features': 100,
    'stop_words': 'english'
}

# Feature processing configuration
FEATURE_CONFIG = {
    'numerical': {
        'features': NUMERICAL_FEATURES,
        'strategy': 'standard_scaler',
        'missing_value_strategy': 'median'
    },
    'categorical': {
        'features': CATEGORICAL_FEATURES,
        'strategy': 'label_encoder',
        'missing_value_strategy': 'mode'
    },
    'text': {
        'features': TEXT_FEATURES,
        'strategy': 'tfidf',
        'config': TFIDF_CONFIG
    },
    'timestamp': {
        'source_column': 'timestamp',
        'derived_features': TIMESTAMP_FEATURES
    }
}

# Output feature ordering (for consistency in tensor creation)
FINAL_FEATURE_ORDER = (
    NUMERICAL_FEATURES +
    TIMESTAMP_FEATURES +
    [f'{col}_encoded' for col in CATEGORICAL_FEATURES] +
    [f'text_feature_{i}' for i in range(TFIDF_CONFIG['max_features'])]
) 