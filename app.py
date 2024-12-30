import os
import gc
import tempfile
import uuid
import pandas as pd
import google.generativeai as genai

from llama_index.core import Settings, Document
from llama_index.llms.gemini import Gemini
from llama_index.core import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex

import streamlit as st

# Initialize session state
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
if "file_cache" not in st.session_state:
    st.session_state.file_cache = {}
if "query_engine" not in st.session_state:
    st.session_state.query_engine = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Configure Google API
os.environ["GOOGLE_API_KEY"] = "AIzaSyD_x23GFJ76FCBTJYf9rDtGnN-IbRK6psw"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

@st.cache_resource
def load_llm():
    llm = Gemini(
        model="models/gemini-pro",
        max_tokens=2048,
        temperature=0.7,
    )
    return llm

def reset_chat():
    st.session_state.messages = []
    gc.collect()

def excel_to_text(df):
    """Convert DataFrame to text representation"""
    text = "Excel File Contents:\n\n"
    # Add column names with their data types
    text += "Columns and their types:\n"
    for col in df.columns:
        text += f"- {col} ({df[col].dtype})\n"
    text += "\nData rows:\n"
    
    # Convert each row to a string with column names
    for idx, row in df.iterrows():
        text += f"\nRow {idx + 1}:\n"
        for col, val in row.items():
            text += f"{col}: {val}\n"
    return text

def display_excel(file):
    st.markdown("### Excel Preview")
    try:
        # Read the Excel file
        df = pd.read_excel(file)
        # Display the dataframe
        st.dataframe(df, use_container_width=True)
        return df
    except Exception as e:
        st.error(f"Error reading Excel file: {str(e)}")
        return None

with st.sidebar:
    st.header("Add your documents!")
    
    uploaded_file = st.file_uploader("Choose your `.xlsx` file", type=["xlsx", "xls"])

    if uploaded_file:
        try:
            # Read and display Excel
            df = display_excel(uploaded_file)
            
            if df is not None:
                file_key = f"{st.session_state.id}-{uploaded_file.name}"
                st.write("Indexing your document...")

                if file_key not in st.session_state.file_cache:
                    # Convert Excel to text format
                    text_content = excel_to_text(df)
                    
                    # Create a Document object
                    doc = Document(text=text_content)
                    
                    # setup llm & embedding model
                    llm = load_llm()
                    embed_model = HuggingFaceEmbedding(
                        model_name="BAAI/bge-large-en-v1.5",
                        trust_remote_code=True
                    )
                    
                    # Configure settings
                    Settings.embed_model = embed_model
                    Settings.llm = llm
                    
                    # Create index and query engine
                    index = VectorStoreIndex.from_documents([doc], show_progress=True)
                    query_engine = index.as_query_engine(
                        streaming=True,
                        similarity_top_k=5
                    )

                    # Customize prompt template
                    qa_prompt_tmpl_str = (
                    "Below is an Excel spreadsheet's content:\n"
                    "---------------------\n"
                    "{context_str}\n"
                    "---------------------\n"
                    "Based on the Excel data above, answer the following question.\n"
                    "Be precise and include specific values from the data.\n"
                    "If the information isn't in the data, say 'I don't know!'\n"
                    "Question: {query_str}\n"
                    "Answer: "
                    )
                    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

                    query_engine.update_prompts(
                        {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
                    )
                    
                    # Store in session state
                    st.session_state.file_cache[file_key] = query_engine
                    st.session_state.query_engine = query_engine
                    st.success("✅ Ready to Chat!")
                else:
                    st.session_state.query_engine = st.session_state.file_cache[file_key]
                    st.success("✅ Ready to Chat!")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.stop()

col1, col2 = st.columns([6, 1])

with col1:
    st.header("Chat with Excel using Gemini 1.5")

with col2:
    st.button("Clear ↺", on_click=reset_chat)

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What would you like to know about the data?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            if st.session_state.query_engine is not None:
                streaming_response = st.session_state.query_engine.query(prompt)
                
                for chunk in streaming_response.response_gen:
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            else:
                message_placeholder.markdown("⚠️ Please upload an Excel file first!")
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")