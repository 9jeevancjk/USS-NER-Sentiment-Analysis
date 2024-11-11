import streamlit as st
import openai
import spacy

# Load the spaCy model for NER (English model)
nlp = spacy.load("en_core_web_sm")

# Set up OpenAI API key
openai.api_key = "your_openai_api_key"

# Function to use OpenAI's API to refine entity recognition or provide additional insights
def get_openai_entities(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an NER model that enhances spaCy's output with extra insights."},
            {"role": "user", "content": f"Provide additional insights on the named entities in the following text: {text}"}
        ],
        max_tokens=150,
    )
    return response['choices'][0]['message']['content']

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
