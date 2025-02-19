# Vehicle Auction Analytics Project 🚗 

## Overview
A comprehensive machine learning solution for analyzing and predicting outcomes in vehicle auctions. This project implements both classification (bid status prediction) and regression (price prediction) models using Random Forests, with a focus on practical business applications in the automotive auction industry.

## 🎯 Key Features
- Bid status prediction with multi-class classification
- Vehicle price prediction using regression analysis
- Automated feature extraction from vehicle descriptions
- Comprehensive data cleaning and preprocessing pipeline
- Advanced exploratory data analysis (EDA)
- Model evaluation and visualization
- AWS deployment-ready configuration

## 🛠️ Technologies Used
- **Python** - Primary programming language
- **pandas & numpy** - Data manipulation and numerical operations
- **scikit-learn** - Machine learning implementation
- **matplotlib & seaborn** - Data visualization
- **boto3** - AWS integration
- **joblib** - Model serialization


## 🔍 Features Engineered
- Vehicle Type Classification
- Manufacturer Extraction
- Year Detection
- Capacity Analysis (Seats/Tonnage)
- Temporal Bid Patterns
- Price Normalization

## 📈 Model Performance
### Classification Model (Bid Status Prediction)
- Implements balanced Random Forest Classifier
- Handles class imbalance through weights
- Includes comprehensive evaluation metrics

### Regression Model (Price Prediction)
- Random Forest Regressor for price estimation
- Feature importance analysis
- Actual vs Predicted visualization

## 🚀 Deployment
The project includes configuration for AWS deployment:
- Model serialization using joblib
- S3 bucket integration
- Production-ready pipeline structure

## 🎓 Skills Demonstrated
- Data Cleaning & Preprocessing
- Feature Engineering
- Machine Learning Pipeline Development
- Model Evaluation & Optimization
- Business Logic Implementation
- Cloud Integration
- Data Visualization
- Code Organization & Documentation

## 🔧 Setup & Installation
1. Clone the repository
```bash
git clone https://github.com/yourusername/vehicle-auction-analytics.git
```

2. Install the required packages
```bash
pip install -r requirements.txt
```

3. Run the main script
```bash
python src/model_training.py
```

## 📝 Future Improvements
- Implementation of API endpoints for real-time predictions
- Integration of deep learning models for image-based vehicle analysis
- Development of an interactive dashboard for visualization
- Addition of more sophisticated feature engineering techniques
- Implementation of model monitoring and retraining pipeline

## 📄 License
This project is licensed under the MIT License - see the LICENSE.md file for details
