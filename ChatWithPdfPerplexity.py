import streamlit as st
import PyPDF2
from io import BytesIO
from openai import OpenAI

# Perplexity API configuration using OpenAI-compatible client
PERPLEXITY_API_KEY = "pplx-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Initialize Perplexity client
client = OpenAI(
    api_key=PERPLEXITY_API_KEY,
    base_url="https://api.perplexity.ai"
)

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def chat_with_perplexity(messages, model="sonar"):
    """Send chat request to Perplexity API using OpenAI client"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI
st.set_page_config(page_title="Chat with PDF", page_icon="📄", layout="wide")
st.title("📄 Chat with PDF using Perplexity")
st.markdown("Upload a PDF and ask questions about its content!")

# Initialize session state
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for PDF upload
with st.sidebar:
    st.header("📁 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
    
    if uploaded_file:
        if st.button("Process PDF"):
            with st.spinner("Extracting text from PDF..."):
                pdf_text = extract_text_from_pdf(uploaded_file)
                if pdf_text:
                    st.session_state.pdf_text = pdf_text
                    st.session_state.chat_history = []
                    st.success(f"✅ PDF processed! ({len(pdf_text)} characters)")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This app uses Perplexity's free model to answer questions about your PDF content.")
    
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# Main chat interface
if st.session_state.pdf_text:
    st.success("PDF loaded! Ask me anything about it.")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your PDF..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Prepare messages for API
        messages = [
            {
                "role": "system",
                "content": f"You are a helpful assistant that answers questions based on the following PDF content. Only answer based on the provided content.\n\nPDF Content:\n{st.session_state.pdf_text[:4000]}"
            }
        ]
        
        # Add recent chat history (last 10 messages)
        recent_history = st.session_state.chat_history[-10:]
        for msg in recent_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Get response from Perplexity
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat_with_perplexity(messages)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})

else:
    st.info("👈 Please upload a PDF file from the sidebar to get started!")
    
    # Example usage
    st.markdown("### How to use:")
    st.markdown("""
    1. Upload a PDF file using the sidebar
    2. Click 'Process PDF' to extract the text
    3. Ask questions about the PDF content
    4. The AI will answer based on the document
    """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and Perplexity API")