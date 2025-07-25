import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_TRACKING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with OLLAMA"

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that answers questions about the LangChain project."),
    ("user", "Question: {question}"),
])

# Function to generate response using Ollama
def generate_response(question, model_name, temperature, max_tokens):
    llm = Ollama(model=model_name, temperature=temperature)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    response = chain.invoke({"question": question})
    return response

# Streamlit UI
st.title("🤖 Q&A Chatbot with Ollama (Local LLMs)")

# Sidebar settings
model_name = st.sidebar.selectbox(
    "🧠 Select LLM Model",
    ["mistral", "gemma:2b", "llama3", "llama3:8b", "llama3:70b"]
)

temperature = st.sidebar.slider("🌡️ Temperature", 0.0, 1.0, 0.7)
max_tokens = st.sidebar.slider("🧠 Max Tokens", 50, 1000, 300)

# User input
st.write("Go ahead and ask any question about LangChain:")
question = st.text_input("💬 Your Question")

# Generate response
if question:
    try:
        response = generate_response(question, model_name, temperature, max_tokens)
        st.success("🧠 Response:")
        st.write(response)
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
else:
    st.info("👈 Please enter a question to get a response.")
