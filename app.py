import streamlit as st
import pickle
from helper import text_transform

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('vectorizer.pkl', 'rb') as file:
    tv = pickle.load(file)


def main():
    st.title("SMS/EMAIL Spam Classifier App")

    text_input = st.text_area("Enter your SMS/Email:")
    

    if st.button("Classify"):
        X = tv.transform([text_transform(text_input)]).toarray()
        y = model.predict(X)
        if y == 1:
            st.subheader("Spam")
        else:
            st.subheader("Not Spam")


if __name__ == "__main__":
    main()
