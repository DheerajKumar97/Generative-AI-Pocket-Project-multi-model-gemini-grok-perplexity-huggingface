import streamlit as st
import PyPDF2
from groq import Groq

# Initialize Groq client with API key
client = Groq(api_key="gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def chat_with_groq(messages):
    """Send messages to Groq API"""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ **Error**: {str(e)}"

# Page configuration
st.title("📄 Chat with PDF using Groq AI")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = None

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.write("Chat with PDF documents using Groq's free AI models.")
    st.info("✅ Free tier: 30 requests/minute")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.pdf_text = None
        st.rerun()

# File upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    # Extract text from PDF
    if st.session_state.pdf_text is None:
        with st.spinner("Reading PDF..."):
            st.session_state.pdf_text = extract_text_from_pdf(uploaded_file)
        
        if st.session_state.pdf_text:
            st.success(f"✅ PDF loaded! ({len(st.session_state.pdf_text)} characters)")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the PDF"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Prepare messages for API
        api_messages = [
            {"role": "system", "content": f"You are a helpful assistant. Answer questions based on the following PDF content:\n\n{st.session_state.pdf_text[:15000]}"},
        ]
        api_messages.extend(st.session_state.messages)
        
        # Get response from Groq
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat_with_groq(api_messages)
                st.markdown(response)
        
        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.info("👆 Upload a PDF file to start chatting!")