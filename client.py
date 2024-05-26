import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.title('Chat with Mashwara AI Bot 🤖')

input_question = st.text_input("Ask a question:")
if input_question:
    response = requests.post(f"{API_URL}/ask", json={"question": input_question})
    if response.status_code == 200:
        answer = response.json()['response']
        st.write("Response:", answer)
    else:
        st.error("Failed to get response from the server.")

if st.button("Process PDFs"):
    response = requests.post(f"{API_URL}/process-pdfs")
    if response.status_code == 200:
        message = response.json()['message']
        st.success(message)
    else:
        st.error("Failed to process PDFs.")
