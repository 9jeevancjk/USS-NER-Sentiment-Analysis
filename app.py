import streamlit as st
import openai
import spacy

# Set the API key for OpenAI
openai.api_key = "OPENAI_API_KEY"

# Load the spaCy NER model (English model)
nlp = spacy.load("en_core_web_sm")

# Function to analyze sentiment with word-level contributions using GPT-4
def analyze_sentiment_with_words(review, category):
    prompt = f"Analyze the sentiment of the following {category} review and provide sentiment contributions for each word (percentage):\n\nReview: {review}"
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a sentiment analysis assistant."},
                {"role": "user", "content": prompt},
            ]
        )

        sentiment_analysis = response['choices'][0]['message']['content']
        return sentiment_analysis.strip()
    except Exception as e:
        return f"An error occurred during sentiment analysis: {e}"

# Function to perform Named Entity Recognition (NER)
def extract_entities(review):
    doc = nlp(review)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Streamlit app
st.title("Sentiment Analysis and Named Entity Recognition")

# User inputs
category = st.selectbox("Select the category of the review:", ["Food", "Product", "Place", "Other"])
review = st.text_area("Enter your review:")

# Analyze button
if st.button("Analyze"):
    if review:
        # Perform sentiment analysis
        st.subheader("Sentiment Analysis with Word-Level Contributions")
        sentiment_with_contributions = analyze_sentiment_with_words(review, category)
        st.write(sentiment_with_contributions)

        # Perform Named Entity Recognition (NER)
        st.subheader("Named Entities in the Review")
        entities = extract_entities(review)
        if entities:
            for entity, label in entities:
                st.write(f"**Entity**: {entity} - **Label**: {label}")
        else:
            st.write("No entities found.")
    else:
        st.warning("Please enter a review to analyze.")
