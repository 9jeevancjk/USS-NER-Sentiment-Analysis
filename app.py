import streamlit as st
import openai
import spacy

# Initialize OpenAI client using Streamlit's secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Load the spaCy model for NER (English model)
nlp = spacy.load("en_core_web_sm")

# # Set up OpenAI API key
# openai.api_key = "your_openai_api_key"
# Function to use OpenAI's API to refine entity recognition or provide additional insights
# Define the function to analyze sentiment with word-level contributions using GPT
def analyze_sentiment_with_words(review, category):
    prompt = f"Analyze the sentiment of the following {category} review and provide sentiment contributions for each word (percentage):\n\nReview: {review}"

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a sentiment analysis assistant."},
            {"role": "user", "content": prompt},
        ]
    )

    sentiment_analysis = response['choices'][0]['message']['content']
    return sentiment_analysis.strip()

# Streamlit App
st.title("Named Entity Recognition App with spaCy and OpenAI")
st.write("Enter some text, and this app will identify named entities using spaCy and optionally enhance them with OpenAI's API.")

# Text input
input_text = st.text_area("Enter text here", "")

# Option to use OpenAI for additional insights
use_openai = st.checkbox("Enhance with OpenAI")

# Button to trigger NER
if st.button("Extract Entities"):
    if input_text:
        # Process text with spaCy
        doc = nlp(input_text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        # Display entities identified by spaCy
        st.write("### Entities identified by spaCy:")
        if entities:
            for ent, label in entities:
                st.write(f"**Entity**: {ent} - **Label**: {label}")
        else:
            st.write("No entities found by spaCy.")

        # Use OpenAI for additional insights if checkbox is checked
        if use_openai:
            with st.spinner("Enhancing entities with OpenAI..."):
                openai_entities = get_openai_entities(input_text)
                st.write("### OpenAI-enhanced Entities:")
                st.write(openai_entities)
    else:
        st.warning("Please enter some text before extracting entities.")
