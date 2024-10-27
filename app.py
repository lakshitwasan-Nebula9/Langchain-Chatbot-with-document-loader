import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Load the environment variables
load_dotenv()

# Set the working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

# Load the document


def load_document(file_path):
    loader = UnstructuredPDFLoader(file_path)
    documents = loader.load()
    return documents

# Setup the vector store


def setup_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings()
    text_splitter = CharacterTextSplitter(
        separator="/n",
        chunk_size=1000,
        chunk_overlap=200
    )
    doc_chunks = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(doc_chunks, embeddings)
    return vectorstore

# Create the chain


def create_chain(vectorstore):
    llm = ChatGroq(
        # model="llama-3.1-70b-versatile",
        model="llama-3.1-8b-instant",
        temperature=0,
    )
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(
        llm=llm,
        output_key="answer",
        memory_key="chat_history",
        return_messages=True
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        # chain_type="map_reduce",
        memory=memory,
        verbose=True
    )
    return chain


# Setup the streamlit app
st.set_page_config(
    page_title="Langchain Document Chatbot",
    page_icon="🤖",
    layout="centered",

)

st.title("Langchain Document Chatbot")

# Initialize the chat history in streamlit session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader("Upload your pdf file", type=["pdf"])

# If the file is uploaded, we will save it to the working directory
if uploaded_file:
    file_path = f"{working_dir}/{uploaded_file.name}"
    # file_path = "D:\Lakshit Wasan\Internships\Nebula9\Langchain\RAG-Chatbot\attention.pdf"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # vectorstore = setup_vectorstore(load_document(file_path))

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = setup_vectorstore(
            load_document(file_path))

    # To avoid losing the data of the conversation chain, we need to check if it already exists
    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = create_chain(
            st.session_state.vectorstore)

# Display the chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask AI")

# If the user input is not empty, we will pass it to the conversation chain
if user_input:
    response = st.session_state.conversation_chain(user_input)
    st.session_state.chat_history.append(
        {"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        reponse = st.session_state.conversation_chain({"question": user_input})
        assistant_response = response["answer"]
        st.markdown(assistant_response)
        st.session_state.chat_history.append(
            {"role": "assistant", "content": assistant_response})
