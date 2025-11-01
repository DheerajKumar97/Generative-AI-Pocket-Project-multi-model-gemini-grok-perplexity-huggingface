"""
main.py - Backend module containing all business logic and AI model integrations
"""

import PyPDF2
from groq import Groq
from google import genai
from openai import OpenAI
from huggingface_hub import InferenceClient
import json
from docx import Document
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Tuple


class DocumentExtractor:
    """Handles extraction of text from various document formats"""
    
    @staticmethod
    def extract_from_pdf(pdf_file) -> Optional[str]:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            return text
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
    
    @staticmethod
    def extract_from_docx(docx_file) -> Optional[str]:
        """Extract text from DOCX file"""
        try:
            doc = Document(docx_file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error reading DOCX: {str(e)}")
    
    @staticmethod
    def extract_from_html(html_file) -> Optional[str]:
        """Extract text from HTML file"""
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
            raise Exception(f"Error reading HTML: {str(e)}")
    
    @staticmethod
    def extract_from_txt(txt_file) -> Optional[str]:
        """Extract text from TXT file"""
        try:
            text = txt_file.read().decode('utf-8')
            return text
        except Exception as e:
            raise Exception(f"Error reading TXT: {str(e)}")
    
    @staticmethod
    def extract_from_json(json_file) -> Optional[str]:
        """Extract text from JSON file"""
        try:
            content = json_file.read().decode('utf-8')
            data = json.loads(content)
            text = json.dumps(data, indent=2)
            return text
        except Exception as e:
            raise Exception(f"Error reading JSON: {str(e)}")
    
    def extract_text(self, uploaded_file) -> Optional[str]:
        """Extract text based on file type"""
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        extractors = {
            'pdf': self.extract_from_pdf,
            'docx': self.extract_from_docx,
            'html': self.extract_from_html,
            'htm': self.extract_from_html,
            'txt': self.extract_from_txt,
            'json': self.extract_from_json
        }
        
        extractor = extractors.get(file_type)
        if extractor:
            return extractor(uploaded_file)
        else:
            raise Exception(f"Unsupported file type: {file_type}")


class AIModelInterface:
    """Base class for AI model interactions"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def chat(self, messages: List[Dict], **kwargs) -> str:
        """Abstract method to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement chat method")


class GroqModel(AIModelInterface):
    """Groq AI model integration"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.model_name = "llama-3.3-70b-versatile"
    
    def chat(self, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """Send messages to Groq API"""
        try:
            client = Groq(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"⚠️ **Groq Error**: {str(e)}"


class GeminiModel(AIModelInterface):
    """Google Gemini model integration"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.model_name = "gemini-2.0-flash-exp"
    
    def chat(self, document_text: str, chat_history: List[Tuple[str, str]], 
             user_question: str) -> str:
        """Send messages to Gemini API"""
        try:
            client = genai.Client(api_key=self.api_key)
            chat = client.chats.create(model=self.model_name)
            
            # Send document context
            system_prompt = f"Here is the text extracted from the uploaded document:\n{document_text[:15000]}"
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


class PerplexityModel(AIModelInterface):
    """Perplexity AI model integration"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.model_name = "sonar"
        self.base_url = "https://api.perplexity.ai"
    
    def chat(self, messages: List[Dict]) -> str:
        """Send messages to Perplexity API"""
        try:
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"⚠️ **Perplexity Error**: {str(e)}"


class HuggingFaceModel(AIModelInterface):
    """HuggingFace model integration"""
    
    MODEL_MAP = {
        "Qwen 2.5": "Qwen/Qwen2.5-72B-Instruct",
        "Llama 3.1": "meta-llama/Llama-3.1-70B-Instruct",
        "Mistral Nemo": "mistralai/Mistral-Nemo-Instruct-2407"
    }
    
    def __init__(self, api_key: str, model_name: str = "Qwen 2.5"):
        super().__init__(api_key)
        self.model_name = model_name
        self.model_id = self.MODEL_MAP.get(model_name, self.MODEL_MAP["Qwen 2.5"])
    
    def chat(self, messages: List[Dict], temperature: float = 0.7, 
             max_tokens: int = 2000) -> str:
        """Send messages to HuggingFace API"""
        try:
            client = InferenceClient(token=self.api_key)
            
            # Format messages for chat completion
            formatted_messages = [
                {"role": msg["role"], "content": msg["content"]} 
                for msg in messages
            ]
            
            # Call the chat completion API
            response = client.chat_completion(
                messages=formatted_messages,
                model=self.model_id,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            if response and response.choices:
                return response.choices[0].message.content
            else:
                return "No response generated from the model."
                
        except Exception as e:
            return self._handle_error(str(e))
    
    def _handle_error(self, error_msg: str) -> str:
        """Handle and format error messages"""
        if "401" in error_msg or "unauthorized" in error_msg.lower():
            return "⚠️ **Authentication Error**: Invalid API key. Please check your HuggingFace token."
        elif "404" in error_msg:
            return f"⚠️ **Model Not Found**: Try switching to a different model. Current: {self.model_name}"
        elif "503" in error_msg or "loading" in error_msg.lower():
            return "⚠️ **Model Loading**: The model is warming up. Please wait 20-30 seconds and try again."
        elif "rate" in error_msg.lower() or "limit" in error_msg.lower():
            return "⚠️ **Rate Limit**: You've hit the free tier rate limit. Please wait a moment and try again."
        else:
            return f"⚠️ **HuggingFace Error**: {error_msg}"


class ChatbotManager:
    """Manages the chatbot conversation and model interactions"""
    
    def __init__(self):
        self.document_extractor = DocumentExtractor()
        self.current_model = None
        self.document_text = None
    
    def load_document(self, uploaded_file) -> Tuple[bool, str, Optional[str]]:
        """Load and extract text from document"""
        try:
            self.document_text = self.document_extractor.extract_text(uploaded_file)
            file_type = uploaded_file.name.split('.')[-1].upper()
            char_count = len(self.document_text) if self.document_text else 0
            return True, f"✅ {file_type} loaded! ({char_count} characters)", self.document_text
        except Exception as e:
            return False, str(e), None
    
    def set_model(self, model_type: str, api_key: str, **kwargs):
        """Initialize the selected AI model"""
        if model_type == "Groq (Llama 3.3)":
            self.current_model = GroqModel(api_key)
        elif model_type == "Gemini 2.0 Flash":
            self.current_model = GeminiModel(api_key)
        elif model_type == "Perplexity (Sonar)":
            self.current_model = PerplexityModel(api_key)
        elif model_type == "HuggingFace (Mistral 7B)":
            hf_model_name = kwargs.get('hf_model_choice', 'Qwen 2.5')
            self.current_model = HuggingFaceModel(api_key, hf_model_name)
    
    def get_response(self, messages: List[Dict], model_type: str, prompt: str) -> str:
        """Get response from the current model"""
        if not self.current_model:
            return "⚠️ Model not initialized"
        
        if not self.document_text:
            return "⚠️ No document loaded"
        
        # Handle Gemini differently
        if model_type == "Gemini 2.0 Flash":
            chat_history = [
                (msg["role"].capitalize(), msg["content"]) 
                for msg in messages[:-1]
            ]
            return self.current_model.chat(
                self.document_text,
                chat_history,
                prompt
            )
        else:
            # Prepare messages with document context
            api_messages = [
                {
                    "role": "system", 
                    "content": f"You are a helpful assistant. Answer questions based on the following document content:\n\n{self.document_text[:15000]}"
                }
            ]
            api_messages.extend(messages)
            return self.current_model.chat(api_messages)