import pandas as pd
import numpy as np
import ast
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer # <--- NEW
from sklearn.compose import ColumnTransformer # <--- NEW
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# --- 1. DATA PREPARATION (Priprema podataka) ---

def load_and_clean_data(filepath):
    # Added 'Reviewer' to the columns being loaded
    df = pd.read_csv(filepath, usecols=['Review', 'Score', 'genres', 'Reviewer']) 
    df['genres'] = df['genres'].fillna("[]")
    df = df.dropna(subset=['Review', 'Score', 'Reviewer'])
    
    # Convert string representation of list to actual list
    df['genres'] = df['genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
    
    return df

# --- 2. ML PROBLEM SOLUTION (Rješenje ML problema) ---

def train_model(df):
    # Split data - Now including 'Reviewer'
    X = df[['Review', 'Reviewer']] 
    y = df['Score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # MLflow Tracking & Versioning
    mlflow.set_experiment("Rotten_Tomatoes_Personalized_Analysis")

    with mlflow.start_run(run_name="Ridge_Personalized_V2"):
        # Hyperparameters
        alpha = 1.0
        max_features = 5000
        
        mlflow.log_param("model_type", "Ridge")
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("tfidf_max_features", max_features)

        # --- INTEGRATED PIPELINE START ---
        
        # 1. Define the ColumnTransformer
        # This handles the Review text and Reviewer names in parallel
        preprocessor = ColumnTransformer(
            transformers=[
                ('text', TfidfVectorizer(stop_words='english', max_features=max_features), 'Review'),
                ('reviewer', OneHotEncoder(handle_unknown='ignore'), ['Reviewer']) # <--- PERSONALIZATION
            ]
        )

        # 2. Define the full Pipeline
        # This bundles preprocessing and the model into one object
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', Ridge(alpha=alpha))
        ])

        print("Training personalized model pipeline...")
        model_pipeline.fit(X_train, y_train)

        # --- INTEGRATED PIPELINE END ---

        # Evaluation
        predictions = model_pipeline.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        # --- 3. TRACKING & VERSIONING (Praćenje i verzioniranje) ---
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)

        # Log the ENTIRE pipeline (preprocessor + model)
        # This makes deployment in Streamlit much easier
        mlflow.sklearn.log_model(model_pipeline, "personalized_sentiment_model")
        
        print(f"Run Finished.\nMAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR2: {r2:.2f}")
        return model_pipeline

if __name__ == "__main__":
    data = load_and_clean_data('rottentomatoes_with_genres.csv')
    # Using a sample to speed up training, remove .sample() for the final run
    model = train_model(data)