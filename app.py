import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Personalized Movie Critic", page_icon="🍅")

@st.cache_resource
def load_assets():
    model = joblib.load("model_v3.pkl")
    df = pd.read_csv('rottentomatoes_with_genres.csv', usecols=['Reviewer', 'genres'])
    
    # Get unique reviewers
    reviewers = sorted(df['Reviewer'].dropna().unique().tolist())
    
    # Get unique genres for the multiselect
    import ast
    all_genres = set()
    for g in df['genres'].dropna():
        try:
            all_genres.update(ast.literal_eval(g))
        except:
            continue
    
    return model, reviewers, sorted(list(all_genres))

model, reviewers, genres_list = load_assets()

st.title("🍅 Complete Score Predictor")
st.write("Predicting scores based on **Review Content**, **Critic Identity**, and **Movie Genres**.")

# Input UI
selected_reviewer = st.selectbox("Select a Critic", reviewers)
selected_genres = st.multiselect("Select Movie Genres", genres_list, default=["Drama"])
review_text = st.text_area("Review Text", height=150)

if st.button("Predict Score"):
    if review_text:
        # 1. Format genres the same way as training (space-separated string)
        genre_string = " ".join(selected_genres)
        
        # 2. Create Input DataFrame
        input_df = pd.DataFrame({
            'Review': [review_text],
            'Reviewer': [selected_reviewer],
            'genres_flat': [genre_string] # Matches the column name in training
        })

        # 3. Predict
        prediction = model.predict(input_df)[0]
        final_score = max(0, min(100, float(prediction)))

        # Display
        st.metric("Predicted Score", f"{final_score:.1f} / 100")
        if final_score >= 60:
            st.success("Verdict: FRESH 🍅")
        else:
            st.error("Verdict: ROTTEN 🤢")
    else:
        st.warning("Please enter a review.")