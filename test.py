import streamlit as st
import PyPDF2
from groq import Groq
from google import genai
from openai import OpenAI
from huggingface_hub import InferenceClient
import json
from docx import Document
from bs4 import BeautifulSoup

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

def extract_text_from_docx(docx_file):
    """Extract text from uploaded DOCX file"""
    try:
        doc = Document(docx_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {str(e)}")
        return None

def extract_text_from_html(html_file):
    """Extract text from uploaded HTML file"""
    try:
        content = html_file.read().decode('utf-8')
        soup = BeautifulSoup(content, 'html.parser')
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        text = soup.get_text()
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        return text
    except Exception as e:
        st.error(f"Error reading HTML: {str(e)}")
        return None

def extract_text_from_txt(txt_file):
    """Extract text from uploaded TXT file"""
    try:
        text = txt_file.read().decode('utf-8')
        return text
    except Exception as e:
        st.error(f"Error reading TXT: {str(e)}")
        return None

def extract_text_from_json(json_file):
    """Extract text from uploaded JSON file"""
    try:
        content = json_file.read().decode('utf-8')
        data = json.loads(content)
        # Convert JSON to readable text format
        text = json.dumps(data, indent=2)
        return text
    except Exception as e:
        st.error(f"Error reading JSON: {str(e)}")
        return None

def extract_text_from_file(uploaded_file):
    """Extract text based on file type"""
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    if file_type == 'pdf':
        return extract_text_from_pdf(uploaded_file)
    elif file_type == 'docx':
        return extract_text_from_docx(uploaded_file)
    elif file_type in ['html', 'htm']:
        return extract_text_from_html(uploaded_file)
    elif file_type == 'txt':
        return extract_text_from_txt(uploaded_file)
    elif file_type == 'json':
        return extract_text_from_json(uploaded_file)
    else:
        st.error(f"Unsupported file type: {file_type}")
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
        
        # Send document context
        system_prompt = f"Here is the text extracted from the uploaded document:\n{pdf_text[:15000]}"
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

def chat_with_perplexity(api_key, messages):
    """Send messages to Perplexity API"""
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.perplexity.ai"
        )
        response = client.chat.completions.create(
            model="sonar",
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ **Perplexity Error**: {str(e)}"

def chat_with_huggingface(api_key, messages, model_name):
    """Send messages to HuggingFace API using InferenceClient"""
    try:
        # Initialize the Inference Client
        client = InferenceClient(token=api_key)
        
        # Model options with their full IDs
        model_map = {
            "Qwen 2.5": "Qwen/Qwen2.5-72B-Instruct",
            "Llama 3.1": "meta-llama/Llama-3.1-70B-Instruct",
            "Mistral Nemo": "mistralai/Mistral-Nemo-Instruct-2407"
        }
        
        model_id = model_map.get(model_name, model_map["Qwen 2.5"])
        
        # Format messages for chat completion
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Call the chat completion API
        response = client.chat_completion(
            messages=formatted_messages,
            model=model_id,
            max_tokens=2000,
            temperature=0.7
        )
        
        # Extract the response content
        if response and response.choices:
            return response.choices[0].message.content
        else:
            return "No response generated from the model."
            
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "unauthorized" in error_msg.lower():
            return "⚠️ **Authentication Error**: Invalid API key. Please check your HuggingFace token."
        elif "404" in error_msg:
            return f"⚠️ **Model Not Found**: Try switching to a different model. Current: {model_name}"
        elif "503" in error_msg or "loading" in error_msg.lower():
            return "⚠️ **Model Loading**: The model is warming up. Please wait 20-30 seconds and try again."
        elif "rate" in error_msg.lower() or "limit" in error_msg.lower():
            return "⚠️ **Rate Limit**: You've hit the free tier rate limit. Please wait a moment and try again."
        else:
            return f"⚠️ **HuggingFace Error**: {error_msg}"

# Page configuration
st.set_page_config(page_title="Multi-Model Document Chatbot", page_icon="📄", layout="wide")

st.title("📄 LLM-Powered Multi-Model PDF Chatbot")
st.markdown("### (Groq, Gemini, Perplexity AI, and HuggingFace)")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = None
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = ""
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = ""
if "perplexity_api_key" not in st.session_state:
    st.session_state.perplexity_api_key = ""
if "huggingface_api_key" not in st.session_state:
    st.session_state.huggingface_api_key = ""
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "Groq (Llama 3.3)"
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None
if "hf_model_choice" not in st.session_state:
    st.session_state.hf_model_choice = "Qwen 2.5"

# Sidebar
with st.sidebar:
    st.header("🤖 Model Selection")
    
    # Model selector
    model_option = st.radio(
        "Choose AI Model:",
        ["Groq (Llama 3.3)", "Gemini 2.0 Flash", "Perplexity (Sonar)", "HuggingFace (Mistral 7B)"],
        index=["Groq (Llama 3.3)", "Gemini 2.0 Flash", "Perplexity (Sonar)", "HuggingFace (Mistral 7B)"].index(st.session_state.selected_model) if st.session_state.selected_model in ["Groq (Llama 3.3)", "Gemini 2.0 Flash", "Perplexity (Sonar)", "HuggingFace (Mistral 7B)"] else 0
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
    elif st.session_state.selected_model == "Gemini 2.0 Flash":
        st.subheader("🔑 Gemini API Key")
        st.info("Get your free API key from: [aistudio.google.com/api-keys](https://aistudio.google.com/api-keys)")
        gemini_key = st.text_input(
            "Enter Gemini API Key:",
            value=st.session_state.gemini_api_key,
            type="password",
            placeholder="AIza..."
        )
        st.session_state.gemini_api_key = gemini_key
    elif st.session_state.selected_model == "Perplexity (Sonar)":
        st.subheader("🔑 Perplexity API Key")
        st.info("Get your API key from: [perplexity.ai/settings/api](https://www.perplexity.ai/settings/api)")
        perplexity_key = st.text_input(
            "Enter Perplexity API Key:",
            value=st.session_state.perplexity_api_key,
            type="password",
            placeholder="pplx-..."
        )
        st.session_state.perplexity_api_key = perplexity_key
    else:  # HuggingFace
        st.subheader("🔑 HuggingFace API Key")
        st.info("Get your free API key from: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)")
        st.warning("⚠️ Create a 'Read' token (simple, not Fine-grained)")
        huggingface_key = st.text_input(
            "Enter HuggingFace API Key:",
            value=st.session_state.huggingface_api_key,
            type="password",
            placeholder="hf_..."
        )
        st.session_state.huggingface_api_key = huggingface_key
        
        # Model selector for HuggingFace
        st.markdown("---")
        st.subheader("🤖 HuggingFace Model")
        hf_model = st.selectbox(
            "Choose Model:",
            ["Qwen 2.5", "Llama 3.1", "Mistral Nemo"],
            index=["Qwen 2.5", "Llama 3.1", "Mistral Nemo"].index(st.session_state.hf_model_choice) if st.session_state.hf_model_choice in ["Qwen 2.5", "Llama 3.1", "Mistral Nemo"] else 0
        )
        st.session_state.hf_model_choice = hf_model
    
    st.markdown("---")
    st.header("ℹ️ About")
    st.write("Chat with your documents using multiple AI models.")
    st.write("**Supported Formats:**")
    st.write("• PDF • DOCX • HTML • TXT • JSON")
    st.write("**Current Model:**", st.session_state.selected_model)
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Main content
uploaded_file = st.file_uploader(
    "📤 Upload a document file", 
    type=["pdf", "docx", "html", "htm", "txt", "json"],
    help="Supported formats: PDF, DOCX, HTML, TXT, JSON"
)

# Check if all requirements are met
doc_uploaded = uploaded_file is not None
api_key_provided = False

if st.session_state.selected_model == "Groq (Llama 3.3)":
    api_key_provided = bool(st.session_state.groq_api_key)
    current_api_key = st.session_state.groq_api_key
elif st.session_state.selected_model == "Gemini 2.0 Flash":
    api_key_provided = bool(st.session_state.gemini_api_key)
    current_api_key = st.session_state.gemini_api_key
elif st.session_state.selected_model == "Perplexity (Sonar)":
    api_key_provided = bool(st.session_state.perplexity_api_key)
    current_api_key = st.session_state.perplexity_api_key
else:  # HuggingFace
    api_key_provided = bool(st.session_state.huggingface_api_key)
    current_api_key = st.session_state.huggingface_api_key

# Show status messages
if not doc_uploaded:
    st.warning("⚠️ Please upload a document file and enter the appropriate model's API Key to continue")
elif not api_key_provided:
    st.warning(f"⚠️ Please enter your {st.session_state.selected_model.split()[0]} API key in the sidebar")
else:
    # Extract document text if new file is uploaded
    if st.session_state.uploaded_file_name != uploaded_file.name:
        with st.spinner("📖 Reading document..."):
            st.session_state.pdf_text = extract_text_from_file(uploaded_file)
            st.session_state.uploaded_file_name = uploaded_file.name
        
        if st.session_state.pdf_text:
            file_type = uploaded_file.name.split('.')[-1].upper()
            st.success(f"✅ {file_type} loaded! ({len(st.session_state.pdf_text)} characters)")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input (only enabled when all requirements are met)
    if prompt := st.chat_input(f"Ask a question about the document using {st.session_state.selected_model}"):
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
                        {"role": "system", "content": f"You are a helpful assistant. Answer questions based on the following document content:\n\n{st.session_state.pdf_text[:15000]}"},
                    ]
                    api_messages.extend(st.session_state.messages)
                    response = chat_with_groq(current_api_key, api_messages)
                elif st.session_state.selected_model == "Gemini 2.0 Flash":
                    # Use Gemini
                    response = chat_with_gemini(
                        current_api_key,
                        st.session_state.pdf_text,
                        [(msg["role"].capitalize(), msg["content"]) for msg in st.session_state.messages[:-1]],
                        prompt
                    )
                elif st.session_state.selected_model == "Perplexity (Sonar)":
                    # Prepare messages for Perplexity
                    api_messages = [
                        {"role": "system", "content": f"You are a helpful assistant. Answer questions based on the following document content:\n\n{st.session_state.pdf_text[:15000]}"},
                    ]
                    api_messages.extend(st.session_state.messages)
                    response = chat_with_perplexity(current_api_key, api_messages)
                else:  # HuggingFace
                    # Prepare messages for HuggingFace
                    api_messages = [
                        {"role": "system", "content": f"You are a helpful assistant. Answer questions based on the following document content:\n\n{st.session_state.pdf_text[:15000]}"},
                    ]
                    api_messages.extend(st.session_state.messages)
                    response = chat_with_huggingface(current_api_key, api_messages, st.session_state.hf_model_choice)
                
                st.markdown(response)
        
        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center;'>
        <p><strong>Made by Dheeraj Kumar K</strong></p>
        <p>
            <a href="https://www.linkedin.com/in/dheerajkumar1997/" target="_blank">LinkedIn</a> • 
            <a href="https://github.com/DheerajKumar97?tab=repositories" target="_blank">GitHub</a> • 
            <a href="https://dheeraj-kumar-k.lovable.app/" target="_blank">Website</a>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)