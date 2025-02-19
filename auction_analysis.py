# ---------------------------
# STEP 1: INITIAL SETUP
# ---------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import re
import boto3
import joblib

# ---------------------------
# STEP 2: DATA LOADING & CLEANING
# ---------------------------
# Load data
df = pd.read_csv("BID Data.csv", encoding="utf-8")

# Clean data
def clean_data(df):
    # Clean amount column
    df['Amount'] = df['Amount'].str.replace(',', '').astype(float)
    
    # Clean datetime - remove timezone and fix year
    df['Placed at'] = (
        df['Placed at']
        .str.replace(' SAST', '')
        .str.replace('2025', '2023')  # Fix year typo
    )
    df['Placed at'] = pd.to_datetime(
        df['Placed at'], 
        format='%Y-%m-%d %I:%M:%S%p', 
        errors='coerce'
    )
    return df

df = clean_data(df)

# ---------------------------
# STEP 3: EXPLORATORY DATA ANALYSIS
# ---------------------------
def plot_analysis(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Status', data=df, hue='Bid type')
    plt.title('Bid Status Distribution by Type')
    plt.show()

    df['Bid Hour'] = df['Placed at'].dt.hour
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Bid Hour'], bins=24, kde=True)
    plt.title('Bid Activity by Hour of Day')
    plt.show()

plot_analysis(df)

# ---------------------------
# STEP 4: FEATURE ENGINEERING (IMPROVED)
# ---------------------------
def extract_vehicle_features(lot_name):
    features = {
        'Vehicle Type': 'Other',
        'Manufacturer': 'Other',
        'Year': np.nan,
        'Capacity': np.nan
    }
    
    try:
        # Extract vehicle type
        if 'BUS' in lot_name:
            features['Vehicle Type'] = 'Bus'
        elif 'LOADER' in lot_name:
            features['Vehicle Type'] = 'Loader'
        elif 'TRUCK' in lot_name:
            features['Vehicle Type'] = 'Truck'
        
        # Extract manufacturer
        if 'TATA' in lot_name:
            features['Manufacturer'] = 'TATA'
        elif 'BOBCAT' in lot_name:
            features['Manufacturer'] = 'BOBCAT'
        
        # Extract year
        year_match = re.search(r'\((\d{4})\)', lot_name)
        if year_match:
            features['Year'] = int(year_match.group(1))
        
        # Extract capacity
        capacity_match = re.search(r'(\d+)-SEAT', lot_name) or re.search(r'(\d+)-TON', lot_name)
        if capacity_match:
            features['Capacity'] = int(capacity_match.group(1))
            
    except Exception as e:
        print(f"Error processing: {lot_name}")
        print(str(e))
    
    return pd.Series(features)

df[['Vehicle Type', 'Manufacturer', 'Year', 'Capacity']] = df['Lot name'].apply(extract_vehicle_features)

# Handle remaining missing values
df['Capacity'] = df['Capacity'].fillna(df['Capacity'].median())
df['Year'] = df['Year'].fillna(df['Year'].median())

# ---------------------------
# STEP 5: BID STATUS PREDICTION 
# ---------------------------
# Prepare data
X_clf = df[['Amount', 'Bid type', 'Vehicle Type', 'Manufacturer', 'Bid Hour']]
y_clf = df['Status']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Bid type', 'Vehicle Type', 'Manufacturer'])
    ],
    remainder='passthrough'
)

# Build pipeline with class weighting
clf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        random_state=42,
        class_weight='balanced',
        n_estimators=200
    ))
])

# Split and train
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, 
    test_size=0.2, 
    random_state=42,
    stratify=y_clf  # Maintain class balance
)

clf_pipeline.fit(X_train_clf, y_train_clf)

# ---------------------------
# STEP 6: PRICE PREDICTION (WITH IMPROVEMENTS)
# ---------------------------
# Prepare data - use only successful sales
sold_lots = df[df['Status'] == 'sold'].copy()

# Get final sale prices using groupby
final_prices = (
    sold_lots
    .sort_values('Amount', ascending=False)
    .groupby('Lot identifier')
    .first()
    .reset_index()
)

# Prepare regression data
X_reg = final_prices[['Vehicle Type', 'Manufacturer', 'Year', 'Capacity']]
y_reg = final_prices['Amount']

# Enhanced regression pipeline
reg_pipeline = Pipeline([
    ('preprocessor', OneHotEncoder(handle_unknown='ignore')),
    ('regressor', RandomForestRegressor(
        random_state=42,
        n_estimators=200,
        max_depth=5
    ))
])

# Time-based split (assuming data is chronological)
X_train_reg = X_reg.iloc[:int(0.8*len(X_reg))]
X_test_reg = X_reg.iloc[int(0.8*len(X_reg)):]
y_train_reg = y_reg.iloc[:int(0.8*len(y_reg))]
y_test_reg = y_reg.iloc[int(0.8*len(y_reg)):]

reg_pipeline.fit(X_train_reg, y_train_reg)

# ---------------------------
# STEP 7: VISUALIZATION & EVALUATION
# ---------------------------

# Classification report with zero_division
print("\nBid Status Classification Report:")
print(classification_report(
    y_test_clf, 
    clf_pipeline.predict(X_test_clf),
    zero_division=1  # Set undefined precision to 1.0
))

# Check class distribution
print("\nTraining Class Distribution:")
print(y_train_clf.value_counts())

print("\nTest Class Distribution:")
print(y_test_clf.value_counts())

# Check predicted vs actual classes
predicted_classes = clf_pipeline.predict(X_test_clf)
print("\nPredicted Classes:", np.unique(predicted_classes))
print("Actual Classes:", np.unique(y_test_clf))

# Price prediction metrics
predictions = reg_pipeline.predict(X_test_reg)
print("\nPrice Prediction Performance:")
print(f"MAE: {mean_absolute_error(y_test_reg, predictions):.2f}")
print(f"R²: {r2_score(y_test_reg, predictions):.2f}")

# Feature importance plot
feature_names = reg_pipeline.named_steps['preprocessor'].get_feature_names_out()
importances = reg_pipeline.named_steps['regressor'].feature_importances_

plt.figure(figsize=(12, 8))
sns.barplot(x=importances, y=feature_names)
plt.title('Feature Importance for Price Prediction')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

# Actual vs Predicted prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test_reg, predictions, alpha=0.6)
plt.plot([min(y_test_reg), max(y_test_reg)], [min(y_test_reg), max(y_test_reg)], '--r')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()




#possible future  deployment 

#aws deployment

# Save the model to a file
#joblib.dump(clf_pipeline, 'clf_pipeline.pkl')
#joblib.dump(reg_pipeline, 'reg_pipeline.pkl')

# Upload the model to S3
#s3 = boto3.client('s3')
#bucket_name = 'your-bucket-name'

#s3.upload_file('clf_pipeline.pkl', bucket_name, 'models/clf_pipeline.pkl')
#s3.upload_file('reg_pipeline.pkl', bucket_name, 'models/reg_pipeline.pkl')

#print("Models uploaded to S3 successfully.")

