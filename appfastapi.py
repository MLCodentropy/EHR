import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import glob
from dotenv import load_dotenv
import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
# Placeholder for Google Generative AI configuration
# genai.configure(api_key=api_key)

def get_latest_modification_time(folder_path):
    """Get the latest modification time of PDFs in a folder."""
    pdf_paths = glob.glob("{folder_path}/*.pdf")
    if not pdf_paths:
        return 0
    return max(os.path.getmtime(pdf_path) for pdf_path in pdf_paths)

def should_process_pdfs(folder_path, last_processed_time):
    """Check if PDFs have been modified since last processing based on modification time."""
    latest_mod_time = get_latest_modification_time(folder_path)
    return latest_mod_time > last_processed_time

def get_pdf_text_from_folder(folder_path):
    text = ""
    for pdf_path in glob.glob(f"{folder_path}/*.pdf"):
        try:
            pdf_reader = PdfReader(pdf_path)
            for page in pdf_reader.pages:
                page_text = page.extract_text() or ""
                text += page_text
        except Exception as e:
            st.error(f"Error processing {pdf_path}: {e}")
    return text

def get_text_chunks(text, max_tokens=1000):
    # Adjust the chunking strategy to respect token limits
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=max_tokens, chunk_overlap=128)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks, force=False):
    # Assume embedding and vector store setup; replace placeholders as needed
    #embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
  
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-pro")
   
    if force or not os.path.exists("faiss_index"):
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        st.success("Vector store created successfully.")
    else:
        st.info("Using existing vector store. No reprocessing needed.")

def user_input(user_question):
    # Placeholder for user input handling; adjust as needed for token limits
    #embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-pro")
   
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question, top_k=5)  # Adjust top_k as needed

    chain = get_conversational_chain()

    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    st.write("Reply: ", response["output_text"])

def get_conversational_chain():
    prompt_template = PromptTemplate(template="""As your AI health assistant, I'm here to help you with your health-related questions. 
    I will use the medical context provided in your documents to offer informed, clear, and concise responses. 
    Please remember that while I can provide information and support, my advice does not replace professional medical consultation. 
    For serious or urgent health issues, always consult a healthcare professional. 
    Now, let's address your query.
    Context from your documents: 
    {context}

    Your Question: {question}
    My Response: 

    Please note that this is general information and should not be taken as specific medical advice. Thanks for asking!""", input_variables=["context", "question"])

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, api_key=api_key)  # Ensure this is correctly configured with your API key
    

    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt_template)
    return chain



def main():
    st.set_page_config(page_title="Chat with Mushwara Ai General Surgery Bot 🤖", layout="wide")
    st.header("Chat with Mushwara Ai General Surgery Bot 🤖")

    pdf_folder_path = "BooksFirst5/"
    last_processed_time = st.session_state.get('last_processed_time', 0)

    if should_process_pdfs(pdf_folder_path, last_processed_time):
        with st.spinner("Processing PDFs..."):
            raw_text = get_pdf_text_from_folder(pdf_folder_path)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks, force=True)
            st.session_state['last_processed_time'] = time.time()
    else:
        st.info("Loaded from existing processing. Ready for queries.")

    user_question = st.text_input("Enter your question here:")
    if user_question:
        try:
            user_input(user_question)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question to get a response.")

if __name__ == "__main__":
    main()
