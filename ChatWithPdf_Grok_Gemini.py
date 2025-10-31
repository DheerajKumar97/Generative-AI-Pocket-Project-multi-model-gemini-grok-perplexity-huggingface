import streamlit as st
import PyPDF2
from groq import Groq
from google import genai

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def chat_with_groq(api_key, messages):
    """Send messages to Groq API"""
    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ **Groq Error**: {str(e)}"

def chat_with_gemini(api_key, pdf_text, chat_history, user_question):
    """Send messages to Gemini API"""
    try:
        client = genai.Client(api_key=api_key)
        chat = client.chats.create(model="gemini-2.0-flash-exp")
        
        # Send PDF context
        system_prompt = f"Here is the text extracted from the uploaded PDF:\n{pdf_text[:15000]}"
        chat.send_message(system_prompt)
        
        # Send chat history for context
        for role, msg in chat_history:
            if role != "System":
                prefix = "User" if role == "User" else "AI"
                chat.send_message(f"{prefix}: {msg}")
        
        # Send current question
        response = chat.send_message(user_question)
        return response.text
    except Exception as e:
        return f"⚠️ **Gemini Error**: {str(e)}"

# Page configuration
st.title("📄 Chat with PDF - Groq & Gemini AI")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = None
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = ""
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = ""
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "Groq (Llama 3.3)"

# Sidebar
with st.sidebar:
    st.header("🤖 Model Selection")
    
    # Model selector
    model_option = st.radio(
        "Choose AI Model:",
        ["Groq (Llama 3.3)", "Gemini 2.0 Flash"],
        index=0 if st.session_state.selected_model == "Groq (Llama 3.3)" else 1
    )
    st.session_state.selected_model = model_option
    
    st.markdown("---")
    
    # API Key inputs based on selected model
    if st.session_state.selected_model == "Groq (Llama 3.3)":
        st.subheader("🔑 Groq API Key")
        st.info("Get your free API key from: [console.groq.com](https://console.groq.com)")
        groq_key = st.text_input(
            "Enter Groq API Key:",
            value=st.session_state.groq_api_key,
            type="password",
            placeholder="gsk_..."
        )
        st.session_state.groq_api_key = groq_key
    else:
        st.subheader("🔑 Gemini API Key")
        st.info("Get your free API key from: [aistudio.google.com/api-keys](https://aistudio.google.com/api-keys)")
        gemini_key = st.text_input(
            "Enter Gemini API Key:",
            value=st.session_state.gemini_api_key,
            type="password",
            placeholder="AIza..."
        )
        st.session_state.gemini_api_key = gemini_key
    
    st.markdown("---")
    st.header("ℹ️ About")
    st.write("Chat with PDF documents using AI models.")
    st.write("**Current Model:**", st.session_state.selected_model)
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Main content
uploaded_file = st.file_uploader("📤 Upload a PDF file", type="pdf")

# Check if all requirements are met
pdf_uploaded = uploaded_file is not None
api_key_provided = False

if st.session_state.selected_model == "Groq (Llama 3.3)":
    api_key_provided = bool(st.session_state.groq_api_key)
    current_api_key = st.session_state.groq_api_key
else:
    api_key_provided = bool(st.session_state.gemini_api_key)
    current_api_key = st.session_state.gemini_api_key

# Show status messages
if not pdf_uploaded:
    st.warning("⚠️ Please upload a PDF file and enter the approriate model's API Key to continue")
elif not api_key_provided:
    st.warning(f"⚠️ Please enter your {st.session_state.selected_model.split()[0]} API key in the sidebar")
else:
    # Extract PDF text
    if st.session_state.pdf_text is None:
        with st.spinner("📖 Reading PDF..."):
            st.session_state.pdf_text = extract_text_from_pdf(uploaded_file)
        
        if st.session_state.pdf_text:
            st.success(f"✅ PDF loaded! ({len(st.session_state.pdf_text)} characters)")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input (only enabled when all requirements are met)
    if prompt := st.chat_input(f"Ask a question about the PDF using {st.session_state.selected_model}"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response based on selected model
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                if st.session_state.selected_model == "Groq (Llama 3.3)":
                    # Prepare messages for Groq
                    api_messages = [
                        {"role": "system", "content": f"You are a helpful assistant. Answer questions based on the following PDF content:\n\n{st.session_state.pdf_text[:15000]}"},
                    ]
                    api_messages.extend(st.session_state.messages)
                    response = chat_with_groq(current_api_key, api_messages)
                else:
                    # Use Gemini
                    response = chat_with_gemini(
                        current_api_key,
                        st.session_state.pdf_text,
                        [(msg["role"].capitalize(), msg["content"]) for msg in st.session_state.messages[:-1]],
                        prompt
                    )
                
                st.markdown(response)
        
        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": response})