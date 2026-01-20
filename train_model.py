import pandas as pd
import joblib
import ast
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from textblob import TextBlob
import numpy as np
import lightgbm as lgb

def engineer_text_features(text):
    """Extract advanced text features for better predictions"""
    if not isinstance(text, str) or len(text) == 0:
        return {
            'review_length': 0,
            'word_count': 0,
            'avg_word_length': 0,
            'sentiment_polarity': 0,
            'sentiment_subjectivity': 0.5,
            'exclamation_count': 0,
            'question_count': 0,
            'uppercase_ratio': 0
        }
    
    blob = TextBlob(text)
    words = text.split()
    word_count = len(words)
    
    return {
        'review_length': len(text),
        'word_count': word_count,
        'avg_word_length': np.mean([len(w) for w in words]) if word_count > 0 else 0,
        'sentiment_polarity': blob.sentiment.polarity,
        'sentiment_subjectivity': blob.sentiment.subjectivity,
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
    }

def train_critic_model():
    # 1. Setup MLflow
    # Set the experiment name so all runs are grouped together
    mlflow.set_experiment("RottenTomatoes_Score_Predictor")

    print("Loading data...")
    df = pd.read_csv('rottentomatoes_with_genres.csv')

    # Data Cleaning
    df = df.dropna(subset=['Review', 'Score', 'genres', 'Reviewer'])
    
    # SPEED OPTIMIZATION: Sample 100k rows for faster training
    # Remove .sample() to train on full dataset
    print(f"Total dataset: {len(df):,} rows")
    if len(df) > 100000:
        df = df#.sample(n=100000, random_state=42)
        print(f"Using sample: {len(df):,} rows for faster training")

    # Feature Engineering: Genres
    def process_genres(genre_str):
        try:
            genre_list = ast.literal_eval(genre_str)
            return " ".join(genre_list)
        except:
            return ""

    print("Processing genres...")
    df['genres_flat'] = df['genres'].apply(process_genres)

    print("Engineering text features (sentiment, length, etc.)...")
    text_features = pd.DataFrame(df['Review'].apply(engineer_text_features).tolist())
    df = pd.concat([df, text_features], axis=1)
    
    print("Calculating reviewer statistics...")
    reviewer_stats = df.groupby('Reviewer')['Score'].agg(['mean', 'std', 'count']).reset_index()
    reviewer_stats.columns = ['Reviewer', 'reviewer_mean_score', 'reviewer_std_score', 'reviewer_review_count']
    reviewer_stats['reviewer_std_score'] = reviewer_stats['reviewer_std_score'].fillna(0)
    df = df.merge(reviewer_stats, on='Reviewer', how='left')
    df[['reviewer_mean_score', 'reviewer_std_score', 'reviewer_review_count']] = \
        df[['reviewer_mean_score', 'reviewer_std_score', 'reviewer_review_count']].fillna(0)

    # Features and Target
    text_features_cols = ['review_length', 'word_count', 'avg_word_length', 'sentiment_polarity', 
                          'sentiment_subjectivity', 'exclamation_count', 'question_count', 'uppercase_ratio']
    reviewer_features_cols = ['reviewer_mean_score', 'reviewer_std_score', 'reviewer_review_count']
    
    X = df[['Review', 'Reviewer', 'genres_flat'] + text_features_cols + reviewer_features_cols]
    y = df['Score']
    
    # Critical: Fill NaN values in text columns before vectorization
    X['Review'] = X['Review'].fillna('Unknown')
    X['Reviewer'] = X['Reviewer'].fillna('Unknown')
    X['genres_flat'] = X['genres_flat'].fillna('')
    
    # Fill NaN values in numeric columns with 0
    for col in text_features_cols + reviewer_features_cols:
        X[col] = X[col].fillna(0)
    
    # Remove any rows with empty reviews (vectorizer can't handle them)
    mask = X['Review'].str.strip().str.len() > 0
    X = X[mask]
    y = y[mask]
    
    # Remove rows where target (Score) is NaN
    mask_y = y.notna()
    X = X[mask_y]
    y = y[mask_y]
    
    print(f"Final dataset size after cleaning: {len(X):,} rows")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define Hyperparameters for LightGBM
    tfidf_max_features = 3000  # word-level vocab size
    char_max_features = 20000  # char-level vocab size
    word_ngram_range = (1, 2)
    char_ngram_range = (3, 5)
    lgb_params = dict(
        n_estimators=800,
        learning_rate=0.06,
        num_leaves=96,
        max_depth=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_samples=20,
        objective='regression',
        random_state=42,
        n_jobs=-1
    )

    # Build Enhanced Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('review_word_tfidf', TfidfVectorizer(
                stop_words='english',
                max_features=tfidf_max_features,
                ngram_range=word_ngram_range,
                min_df=3,
                max_df=0.95
            ), 'Review'),
            ('review_char_tfidf', TfidfVectorizer(
                analyzer='char',
                ngram_range=char_ngram_range,
                min_df=5,
                max_features=char_max_features
            ), 'Review'),
            ('reviewer_id', OneHotEncoder(handle_unknown='ignore'), ['Reviewer']),
            ('genre_bow', CountVectorizer(token_pattern=r'(?u)\b\w+\b'), 'genres_flat'),
            ('numeric_features', StandardScaler(), text_features_cols + reviewer_features_cols)
        ]
    )

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', lgb.LGBMRegressor(**lgb_params))
    ])

    # 2. Start MLflow Run
    print("Starting MLflow run...")
    with mlflow.start_run():
        
        # Log Parameters
        mlflow.log_param("model_type", "Gradient Boosting Regressor")
        mlflow.log_param("tfidf_max_features", tfidf_max_features)
        mlflow.log_param("char_max_features", char_max_features)
        mlflow.log_param("word_ngram_range", str(word_ngram_range))
        mlflow.log_param("char_ngram_range", str(char_ngram_range))
        mlflow.log_param("lgb_params", lgb_params)
        mlflow.log_param("added_text_features", True)
        mlflow.log_param("added_reviewer_stats", True)

        # Train
        print("Training LightGBM model...")
        print(f"Dataset size: {len(X_train)} training samples, {len(X_test)} test samples")
        model.fit(X_train, y_train)
        print("✓ Training completed")

        # Evaluate
        print("Evaluating model...")
        predictions = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        
        # Log Metrics
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        print(f"R² Score: {r2:.4f}")
        
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)

        # Skip cross-validation for speed (comment out to re-enable)
        # Uncomment below for more robust evaluation:
        # print("Running 3-fold cross-validation...")
        # cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_absolute_error', verbose=1)
        # cv_mae = -cv_scores.mean()
        # print(f"Cross-Validation MAE: {cv_mae:.2f}")
        # mlflow.log_metric("cv_mae", cv_mae)
        # print("✓ Cross-validation completed")

        # Log the Model
        mlflow.sklearn.log_model(model, "model")

        # 3. Save for Streamlit
        print("Saving local model_v3.pkl for Streamlit app...")
        joblib.dump(model, 'model_v3.pkl')

    print("Training Complete. Run 'mlflow ui' in your terminal to view results.")

if __name__ == "__main__":
    train_critic_model()