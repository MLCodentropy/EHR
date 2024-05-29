import os
import streamlit as st

from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

def main():
    """
    Main entry point of the application. Sets up the Groq client, Streamlit interface, and handles chat interaction.
    """
    groq_api_key = "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"
    hardcoded_prompt = """
    As your Virtual Health Assistant, I am programmed to provide informed, precise, and clear answers based on the medical knowledge extracted from the documents you've shared. Please be aware that while I aim to deliver accurate advice, my responses are strictly confined to the information from your documents and general medical principles.

**Note:** Please note, my capabilities are confined to interpreting and analyzing the medical information contained within your documents alongside established medical principles. My responses do not extend to generating programming language code, solving mathematical problems, or addressing queries outside the specified medical domain. For any health issues beyond the scope of this information, it is imperative to consult with a qualified healthcare professional directly.


Now, let's focus on your health-related question, ensuring it pertains strictly to the medical context provided by you.

**Medical Context from Your Documents:**
{context}

**Your Question:**
{question}

**My Analysis and Recommendations:**

Please remember that while I offer insights based on the medical data shared, these are general guidelines. For personalized health advice, diagnoses, or treatments, it is crucial to consult a healthcare professional. Thank you for trusting me to assist with your medical inquiries.
    """
    model_name = 'mixtral-8x7b-32768'
    conversational_memory_length = 5
    
    st.title("Medical Mashwara LLM Chatbot")
    st.write("Hello! I'm your friendly Medical Mashwara LLM chatbot.")

    # Memory setup
    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)

    # Text input for user questions
    user_question = st.text_input("Ask a question:", key='user_input')

    # Initialize Groq Langchain chat object
    groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model_name)

    submit_button = st.button("Ask")

    if submit_button and user_question:
        # Handle chat history in session state
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        # Save user's question to session history
        st.session_state.chat_history.append({'human': user_question, 'AI': ''})

        # Construct a chat prompt template using various components
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=hardcoded_prompt
                ),  # This is the persistent system prompt that is always included at the start of the chat.

                MessagesPlaceholder(
                    variable_name="chat_history"
                ),  # This placeholder will be replaced by the actual chat history during the conversation.

                HumanMessagePromptTemplate.from_template(
                    "{human_input}"
                ),  # This template is where the user's current input will be injected into the prompt.
            ]
        )

        # Conversation chain
        conversation = LLMChain(
            llm=groq_chat,
            prompt=prompt,
            verbose=True,
            memory=memory
        )

        # Generate response
        response = conversation.predict(human_input=user_question)
        # Save chatbot's response to session history
        if st.session_state.chat_history:
            st.session_state.chat_history[-1]['AI'] = response

        # Display chat history
        for chat_pair in st.session_state.chat_history:
            st.write("You:", chat_pair['human'])
            if chat_pair['AI']:
                st.write("Chatbot:", chat_pair['AI'])

        # Clear the input field by resetting the value after processing
        st.session_state.user_input = ""

if __name__ == "__main__":
    main()
