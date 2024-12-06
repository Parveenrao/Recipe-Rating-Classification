import streamlit as st
import pickle
import pandas as pd

# Load the tuned logistic regression pipeline
with open('lr_tuned.pkl', 'rb') as f:
    lr_pipeline_tuned = pickle.load(f)

# Streamlit app title
st.title("Recipe Rating Prediction")

# Layout with two columns for inputs
col1, col2 = st.columns(2)

# Collect user inputs
with col1:
    user_reputation = st.number_input(
        "User Reputation", min_value=0.0, help="The reputation score of the user posting the recipe."
    )
    reply_count = st.number_input("Reply Count", min_value=0, help="Number of replies on the recipe.")
    thumbs_up = st.number_input("Thumbs Up Count", min_value=0, help="Number of thumbs up the recipe received.")

with col2:
    thumbs_down = st.number_input("Thumbs Down Count", min_value=0, help="Number of thumbs down the recipe received.")
    best_score = st.number_input("Best Score", min_value=0.0, help="Best score achieved by the recipe.")
    recipe_name = st.text_input("Recipe Name", help="The name of the recipe.")

# Collect recipe review as text input
recipe_review = st.text_area("Recipe Review", help="A brief review of the recipe.")

# Predict button
if st.button("Predict"):
    # Validation for required fields
    if recipe_name.strip() == "" or recipe_review.strip() == "":
        st.error("Please provide both Recipe Name and Recipe Review.")
    else:
        try:
            # Prepare the input data
            input_data = pd.DataFrame([[
                user_reputation, reply_count, thumbs_up, thumbs_down, best_score, recipe_name, recipe_review
            ]], columns=['UserReputation', 'ReplyCount', 'ThumbsUpCount',
                         'ThumbsDownCount', 'BestScore', 'RecipeName', 'Recipe_Review'])

            # Make prediction using the loaded pipeline
            prediction = lr_pipeline_tuned.predict(input_data)

            # Show the result
            st.success(f"The predicted rating is: {prediction[0]}")

            # Optionally show the prediction probability (confidence)
            prediction_prob = lr_pipeline_tuned.predict_proba(input_data)
            st.info(f"Prediction confidence: {max(prediction_prob[0]) * 100:.2f}%")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
