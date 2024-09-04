import streamlit as st
import spacy_streamlit as spt
import spacy
from transformers import pipeline

nlp = spacy.load("en_core_web_sm")
nlp_transformer_model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", grouped_entities=True)

def main():
    st.title("Named Entity Recognition App")
    
    menu = ["HOME", "NER with spaCy", "NER with Transformer"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "HOME":
        st.subheader("Word Tokenization")
        raw_text = st.text_area("Text to tokenize")
        docs = nlp(raw_text)
        if st.button("Tokenize"):
            spt.visualize_tokens(docs)
    elif choice == "NER with spaCy":  
        st.subheader("Named Entity Recognition")
        raw_text = st.text_area("Text to analyze")
        if st.button("Enter"):
            docs = nlp(raw_text)
            spt.visualize_ner(docs)
    elif choice == "NER with Transformer":
        st.subheader("Named Entity Recognition")
        raw_text = st.text_area("Text to analyze")
        if st.button("Enter"):
            result = nlp_transformer_model(raw_text)
            for entity in result:
                st.write(f"{entity['word']} -> {entity['entity_group']}")

if __name__ == "__main__":
    main()
