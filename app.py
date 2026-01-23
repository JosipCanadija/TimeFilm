import streamlit as st
import pandas as pd
import joblib
from textblob import TextBlob 
import ast

st.set_page_config(page_title="Personalized Movie Critic", page_icon="🍅")

@st.cache_resource
def load_assets():
    model = joblib.load("model_v3.pkl")
    # Load more columns now so we can calculate reviewer averages
    df = pd.read_csv('rottentomatoes_with_genres.csv')
    
    # Pre-calculate reviewer stats so we can look them up during prediction
    rev_stats = df.groupby('Reviewer')['Score'].agg(['mean', 'std', 'count']).reset_index()
    rev_stats.columns = ['Reviewer', 'reviewer_mean_score', 'reviewer_std_score', 'reviewer_review_count']
    # Fill NaN for critics with only 1 review
    rev_stats['reviewer_std_score'] = rev_stats['reviewer_std_score'].fillna(0)

    reviewers = sorted(df['Reviewer'].dropna().unique().tolist())
    
    all_genres = set()
    for g in df['genres'].dropna():
        try:
            all_genres.update(ast.literal_eval(g))
        except:
            continue
    
    return model, reviewers, sorted(list(all_genres)), rev_stats

model, reviewers, genres_list, reviewer_stats_df = load_assets()

st.title("🍅 Complete Score Predictor")
st.write("Predicting scores based on **Review Content**, **Critic Identity**, and **Movie Genres**.")

selected_reviewer = st.selectbox("Select a Critic", reviewers)
selected_genres = st.multiselect("Select Movie Genres", genres_list, default=["Drama"])
movie_title = st.text_input("Movie Title")
review_text = st.text_area("Review Text", height=150)

if st.button("Predict Score"):
    if review_text:
        # 1. Feature Engineering (The missing piece!)
        blob = TextBlob(review_text)
        words = review_text.split()
        
        # Calculate text-based features
        feat_dict = {
            'review_length': len(review_text),
            'word_count': len(words),
            'avg_word_length': sum(len(w) for w in words) / len(words) if words else 0,
            'exclamation_count': review_text.count('!'),
            'question_count': review_text.count('?'),
            'uppercase_ratio': sum(1 for c in review_text if c.isupper()) / len(review_text) if review_text else 0,
            'sentiment_polarity': blob.sentiment.polarity,
            'sentiment_subjectivity': blob.sentiment.subjectivity,
        }
        
        # 2. Lookup Reviewer Stats
        rev_info = reviewer_stats_df[reviewer_stats_df['Reviewer'] == selected_reviewer].iloc[0]
        feat_dict['reviewer_mean_score'] = rev_info['reviewer_mean_score']
        feat_dict['reviewer_std_score'] = rev_info['reviewer_std_score']
        feat_dict['reviewer_review_count'] = rev_info['reviewer_review_count']

        # 3. Create Input DataFrame
        input_df = pd.DataFrame([feat_dict])
        input_df['Review'] = review_text
        input_df['Reviewer'] = selected_reviewer
        input_df['genres_flat'] = " ".join(selected_genres)
        # Movie title is collected for display; not used by the model
        input_df['Movie'] = movie_title

        # 4. Predict
        # Note: Ensure columns are in the EXACT order your training set used if not using a Pipeline
        prediction = model.predict(input_df)[0]
        final_score = max(0, min(100, float(prediction)))

        st.metric("Predicted Score", f"{final_score:.1f} / 100", help=f"Movie: {movie_title or 'N/A'}")
        if final_score >= 60:
            st.success("Verdict: FRESH 🍅")
        else:
            st.error("Verdict: ROTTEN 🤢")
    else:
        st.warning("Please enter a review.")