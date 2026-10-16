import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
load_dotenv()



model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7
)


st.header('ResearchTool')

user_input = st.text_input("Enter Your Prompt")

if st.button('summarize'):
    result = model.invoke(user_input)
    st.write(result.content)

