data/features/README.md
# Features Directory

## Overview
This directory contains feature definitions and configurations used in the e-commerce data preprocessing pipeline. It defines how raw data fields are processed and transformed into features suitable for machine learning models.

## Files
- `feature_config.py`: Main configuration file defining feature groups and their processing strategies
- `__init__.py`: Package initialization file

## Feature Groups

### Numerical Features
- Price-related features (original and actual prices)
- Rating-related features (item rating, total ratings)
- Engagement metrics (total sold, favorites)
- Processing: StandardScaler normalization

### Categorical Features
- Delivery information
- Product category
- Seller information
- Platform name
- Processing: Label encoding

### Text Features
- Product title
- Product description
- Processing: TF-IDF vectorization (100 features)

### Timestamp Features
Derived from timestamp field:
- Hour of day
- Day of month
- Month
- Year
- Day of week

## Usage
Import feature configurations in preprocessing scripts:
```python
from features.feature_config import FEATURE_CONFIG, FINAL_FEATURE_ORDER
```

## Notes
- Feature configurations can be modified in `feature_config.py`
- All feature transformations maintain consistent ordering as defined in `FINAL_FEATURE_ORDER`
- Missing value strategies are defined per feature group
