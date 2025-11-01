"""
app.py - Streamlit UI for the Multi-Model Document Chatbot
"""

import streamlit as st
from main import ChatbotManager


class StreamlitUI:
    """Handles all Streamlit UI components and interactions"""
    
    def __init__(self):
        self.chatbot_manager = ChatbotManager()
        self.setup_page_config()
        self.initialize_session_state()
    
    @staticmethod
    def setup_page_config():
        """Configure the Streamlit page"""
        st.set_page_config(
            page_title="Multi-Model Document Chatbot",
            page_icon="📄",
            layout="wide"
        )
    
    @staticmethod
    def initialize_session_state():
        """Initialize session state variables"""
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
    
    def render_header(self):
        """Render the main header"""
        st.title("📄 LLM-Powered Multi-Model Document Chatbot")
        st.markdown("### (Groq, Gemini, Perplexity AI, and HuggingFace)")
    
    def render_sidebar(self):
        """Render the sidebar with model selection and API key inputs"""
        with st.sidebar:
            st.header("🤖 Model Selection")
            
            # Model selector
            model_options = [
                "Groq (Llama 3.3)",
                "Gemini 2.0 Flash",
                "Perplexity (Sonar)",
                "HuggingFace (Mistral 7B)"
            ]
            
            current_index = (
                model_options.index(st.session_state.selected_model)
                if st.session_state.selected_model in model_options
                else 0
            )
            
            model_option = st.radio(
                "Choose AI Model:",
                model_options,
                index=current_index
            )
            st.session_state.selected_model = model_option
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Render API key input based on selected model
            self._render_api_key_input()
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # About section
            st.header("ℹ️ About")
            st.write("Chat with your documents using multiple AI models.")
            st.write("**Supported:** PDF, DOCX, HTML, TXT, JSON")
            
            # Clear chat button
            if st.button("Clear Chat History"):
                st.session_state.messages = []
                st.rerun()
    
    def _render_api_key_input(self):
        """Render API key input based on selected model"""
        if st.session_state.selected_model == "Groq (Llama 3.3)":
            self._render_groq_input()
        elif st.session_state.selected_model == "Gemini 2.0 Flash":
            self._render_gemini_input()
        elif st.session_state.selected_model == "Perplexity (Sonar)":
            self._render_perplexity_input()
        else:  # HuggingFace
            self._render_huggingface_input()
    
    @staticmethod
    def _render_groq_input():
        """Render Groq API key input"""
        st.subheader("🔑 API Key")
        st.info("Get your free API key from: [console.groq.com](https://console.groq.com)")
        groq_key = st.text_input(
            "Enter Groq API Key:",
            value=st.session_state.groq_api_key,
            type="password",
            placeholder="gsk_..."
        )
        st.session_state.groq_api_key = groq_key
    
    @staticmethod
    def _render_gemini_input():
        """Render Gemini API key input"""
        st.subheader("🔑 API Key")
        st.info("Get your free API key from: [aistudio.google.com/api-keys](https://aistudio.google.com/api-keys)")
        gemini_key = st.text_input(
            "Enter Gemini API Key:",
            value=st.session_state.gemini_api_key,
            type="password",
            placeholder="AIza..."
        )
        st.session_state.gemini_api_key = gemini_key
    
    @staticmethod
    def _render_perplexity_input():
        """Render Perplexity API key input"""
        st.subheader("🔑 API Key")
        st.info("Get your API key from: [perplexity.ai/settings/api](https://www.perplexity.ai/settings/api)")
        perplexity_key = st.text_input(
            "Enter Perplexity API Key:",
            value=st.session_state.perplexity_api_key,
            type="password",
            placeholder="pplx-..."
        )
        st.session_state.perplexity_api_key = perplexity_key
    
    @staticmethod
    def _render_huggingface_input():
        """Render HuggingFace model selector and API key input"""
        st.subheader("🤖 Model")
        hf_models = ["Qwen 2.5", "Llama 3.1", "Mistral Nemo"]
        current_index = (
            hf_models.index(st.session_state.hf_model_choice)
            if st.session_state.hf_model_choice in hf_models
            else 0
        )
        
        hf_model = st.selectbox(
            "Choose HuggingFace Model:",
            hf_models,
            index=current_index
        )
        st.session_state.hf_model_choice = hf_model
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🔑 API Key")
        st.info("Get your free API key from: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)")
        st.warning("⚠️ Create a 'Read' token")
        huggingface_key = st.text_input(
            "Enter HuggingFace API Key:",
            value=st.session_state.huggingface_api_key,
            type="password",
            placeholder="hf_..."
        )
        st.session_state.huggingface_api_key = huggingface_key
    
    def render_file_uploader(self):
        """Render file uploader"""
        return st.file_uploader(
            "📤 Upload a document file",
            type=["pdf", "docx", "html", "htm", "txt", "json"],
            help="Supported formats: PDF, DOCX, HTML, TXT, JSON"
        )
    
    def get_current_api_key(self):
        """Get the API key for the currently selected model"""
        api_key_map = {
            "Groq (Llama 3.3)": st.session_state.groq_api_key,
            "Gemini 2.0 Flash": st.session_state.gemini_api_key,
            "Perplexity (Sonar)": st.session_state.perplexity_api_key,
            "HuggingFace (Mistral 7B)": st.session_state.huggingface_api_key
        }
        return api_key_map.get(st.session_state.selected_model, "")
    
    def check_requirements(self, uploaded_file):
        """Check if all requirements are met"""
        doc_uploaded = uploaded_file is not None
        api_key_provided = bool(self.get_current_api_key())
        return doc_uploaded, api_key_provided
    
    def handle_document_upload(self, uploaded_file):
        """Handle document upload and extraction"""
        if st.session_state.uploaded_file_name != uploaded_file.name:
            with st.spinner("📖 Reading document..."):
                success, message, text = self.chatbot_manager.load_document(uploaded_file)
                st.session_state.uploaded_file_name = uploaded_file.name
                st.session_state.pdf_text = text
            
            if success:
                st.success(message)
            else:
                st.error(message)
    
    def render_chat_history(self):
        """Render chat message history"""
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    def handle_user_input(self):
        """Handle user chat input and generate response"""
        prompt = st.chat_input(
            f"Ask a question about the document using {st.session_state.selected_model}"
        )
        
        if prompt:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate assistant response
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    # Initialize the model
                    api_key = self.get_current_api_key()
                    self.chatbot_manager.set_model(
                        st.session_state.selected_model,
                        api_key,
                        hf_model_choice=st.session_state.hf_model_choice
                    )
                    
                    # Get response
                    response = self.chatbot_manager.get_response(
                        st.session_state.messages,
                        st.session_state.selected_model,
                        prompt
                    )
                    
                    st.markdown(response)
            
            # Add assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    @staticmethod
    def render_footer():
        """Render footer with attribution"""
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; padding: 20px 0;'>
                <p style='margin: 5px 0;'><strong>Made by Dheeraj Kumar K</strong></p>
                <p style='margin: 5px 0;'>
                    <a href="https://www.linkedin.com/in/dheerajkumar1997/" target="_blank" style='margin: 0 10px;'>LinkedIn</a> • 
                    <a href="https://github.com/DheerajKumar97?tab=repositories" target="_blank" style='margin: 0 10px;'>GitHub</a> • 
                    <a href="https://dheeraj-kumar-k.lovable.app/" target="_blank" style='margin: 0 10px;'>Website</a>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    def run(self):
        """Main application flow"""
        # Render header
        self.render_header()
        
        # Render sidebar
        self.render_sidebar()
        
        # File uploader
        uploaded_file = self.render_file_uploader()
        
        # Check requirements
        doc_uploaded, api_key_provided = self.check_requirements(uploaded_file)
        
        # Show status messages
        if not doc_uploaded:
            st.warning("⚠️ Please upload a document file and enter the appropriate model's API Key to continue")
        elif not api_key_provided:
            model_name = st.session_state.selected_model.split()[0]
            st.warning(f"⚠️ Please enter your {model_name} API key in the sidebar")
        else:
            # Handle document upload
            self.handle_document_upload(uploaded_file)
            
            # Render chat history
            self.render_chat_history()
            
            # Handle user input
            self.handle_user_input()
        
        # Render footer
        self.render_footer()


def main():
    """Entry point for the Streamlit application"""
    app = StreamlitUI()
    app.run()


if __name__ == "__main__":
    main()