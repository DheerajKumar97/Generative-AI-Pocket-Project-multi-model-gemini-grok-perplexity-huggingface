# 📄 LLM-Powered Multi-Model Document Chatbot

A powerful Streamlit-based chatbot application that allows you to interact with your documents (PDF, DOCX, HTML, TXT, JSON) using multiple state-of-the-art Large Language Models (LLMs).

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-green)

## 🌟 Features

- **Multiple AI Models Support**
  - 🦙 **Groq (Llama 3.3 70B)** - Ultra-fast inference
  - 🤖 **Google Gemini 2.0 Flash** - Advanced AI capabilities
  - 🔍 **Perplexity AI (Sonar)** - Real-time web search integration
  - 🤗 **HuggingFace Models** - Qwen 2.5, Llama 3.1, Mistral Nemo

- **Multi-Format Document Support**
  - 📕 PDF (Portable Document Format)
  - 📝 DOCX (Microsoft Word Documents)
  - 🌐 HTML (Web Pages)
  - 📄 TXT (Plain Text)
  - 📊 JSON (Structured Data)

- **User-Friendly Interface**
  - Clean and intuitive Streamlit UI
  - Real-time chat interface
  - Chat history management
  - Model switching on-the-fly
  - Responsive design

## 🚀 Demo

![Demo Screenshot](screenshot.png)

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- API Keys for the models you want to use:
  - [Groq API Key](https://console.groq.com)
  - [Google Gemini API Key](https://aistudio.google.com/api-keys)
  - [Perplexity API Key](https://www.perplexity.ai/settings/api)
  - [HuggingFace API Token](https://huggingface.co/settings/tokens)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/DheerajKumar97/llm-multi-model-chatbot.git
   cd llm-multi-model-chatbot
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📦 Dependencies

```txt
streamlit
PyPDF2
groq
google-genai
openai
python-docx
beautifulsoup4
huggingface-hub
```

## 🎯 Usage

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Access the web interface**
   - The application will automatically open in your default browser
   - Default URL: `http://localhost:8501`

3. **Setup and Chat**
   - Select your preferred AI model from the sidebar
   - Enter the corresponding API key
   - Upload a document (PDF, DOCX, HTML, TXT, or JSON)
   - Start chatting with your document!

## 🔑 Getting API Keys

### Groq (Free)
1. Visit [console.groq.com](https://console.groq.com)
2. Sign up for a free account
3. Navigate to API Keys section
4. Create a new API key

### Google Gemini (Free)
1. Visit [aistudio.google.com](https://aistudio.google.com/api-keys)
2. Sign in with your Google account
3. Click "Get API Key"
4. Create a new API key

### Perplexity AI
1. Visit [perplexity.ai/settings/api](https://www.perplexity.ai/settings/api)
2. Sign up for an account
3. Navigate to API settings
4. Generate a new API key

### HuggingFace (Free)
1. Visit [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Sign up for a free account
3. Click "New token"
4. Select **"Read"** token type (not Fine-grained)
5. Generate and copy the token

## 💡 How It Works

1. **Document Processing**: The application extracts text from your uploaded document based on its file type
2. **Context Injection**: The extracted text is injected into the system prompt as context
3. **AI Processing**: Your questions are sent to the selected AI model along with the document context
4. **Response Generation**: The AI model generates responses based on the document content
5. **Chat History**: Conversation history is maintained for context-aware responses

## 🎨 Supported Models

| Model | Provider | Size | Speed | Best For |
|-------|----------|------|-------|----------|
| Llama 3.3 | Groq | 70B | ⚡ Ultra Fast | General queries, code |
| Gemini 2.0 Flash | Google | - | ⚡ Fast | Complex reasoning |
| Sonar | Perplexity | - | 🔍 Real-time | Web-connected queries |
| Qwen 2.5 | HuggingFace | 72B | 🐢 Slower | Advanced analysis |
| Llama 3.1 | HuggingFace | 70B | 🐢 Slower | Open-source alternative |
| Mistral Nemo | HuggingFace | 12B | ⚡ Fast | Efficient processing |

## 🔒 Security & Privacy

- **API Keys**: All API keys are stored only in session state and are never logged or saved permanently
- **Document Privacy**: Documents are processed in-memory and are not stored on any server
- **Local Processing**: Text extraction happens locally on your machine

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🐛 Known Issues

- HuggingFace models may experience cold start delays (20-30 seconds)
- Large documents (>15,000 characters) are truncated for context window limits
- Some PDF files with complex formatting may have text extraction issues

## 🔮 Future Enhancements

- [ ] Add support for more document formats (PPT, Excel, etc.)
- [ ] Implement document chunking for large files
- [ ] Add conversation export functionality
- [ ] Multi-document chat support
- [ ] Custom model fine-tuning options
- [ ] Voice input/output capabilities

## 📧 Contact & Support

**Made by Dheeraj Kumar K**

- 💼 [LinkedIn](https://www.linkedin.com/in/dheerajkumar1997/)
- 🐙 [GitHub](https://github.com/DheerajKumar97?tab=repositories)
- 🌐 [Website](https://dheeraj-kumar-k.lovable.app/)

For issues and questions, please use the [GitHub Issues](https://github.com/DheerajKumar97/llm-multi-model-chatbot/issues) page.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing web framework
- [Groq](https://groq.com/) for lightning-fast LLM inference
- [Google](https://ai.google.dev/) for Gemini API
- [Perplexity AI](https://www.perplexity.ai/) for real-time AI capabilities
- [HuggingFace](https://huggingface.co/) for open-source models

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

---

**Built with ❤️ using Streamlit and Multiple LLMs**
