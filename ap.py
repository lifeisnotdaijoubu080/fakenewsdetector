import streamlit as st
import joblib

vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")

st.title("Fake News Detector")
st.write("Enter a News Article below to check whether it is fake or real. ")

news_input = st.text_area("News Article:","")

if st.button("Check News"):
    if news_input.strip():
        transform_input = vectorizer.transform([news_input])
        prediction = model.predict(transform_input)
        print("prediction:",prediction)

        if prediction[0]==1:
            st.success("The News is Real! ")
        elif prediction[0]==0:
            st.error("the news is fake! ")
        else:
            st.warning("Please enter some text to analyze. ")

            # some comments