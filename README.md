# Multi-PDF Chat AI Agent 📚🤖

Chat seamlessly with multiple PDFs using **LangChain**, **Google Gemini**, and **FAISS Vector DB** via a **Streamlit web app**. Ask questions directly from your uploaded PDFs and get accurate responses instantly.

---

## 📝 Description

This app allows you to upload multiple PDF documents, extract their content, and interact with a chatbot trained on that content. It uses **HuggingFace embeddings** for local vector storage and **Google Gemini** for response generation. Perfect for research, study, or document analysis.

---

## 🎯 How It Works

1. **PDF Loading** – Upload multiple PDFs; text is extracted.
2. **Text Chunking** – Text is split into manageable chunks for better processing.
3. **Vector Embeddings** – Chunks are converted to embeddings using **HuggingFace `all-MiniLM-L6-v2`**.
4. **Similarity Search** – Queries are matched with the most relevant chunks using **FAISS**.
5. **Response Generation** – The matched chunks are passed to **Google Gemini** to generate detailed answers.

---

## 🌟 Key Features

* Multi-document conversational QA
* Local embeddings with HuggingFace (free & fast)
* Google Gemini for high-quality responses
* Supports PDF and TXT files
* Adaptive text chunking for accurate retrieval

---

## ⚡ Requirements

* `streamlit` – Web interface
* `PyPDF2` – PDF reading
* `langchain` – LLM and chain management
* `langchain-google-genai` – Google Gemini integration
* `langchain-community` – FAISS vector store & embeddings
* `sentence-transformers` – HuggingFace embeddings
* `faiss-cpu` – Local vector search
* `python-dotenv` – Environment variables

---

## ▶️ Installation

```bash
git clone https://github.com/abhikant12/Multi-PDFs_ChatApp_AI-Agent.git
cd Multi-PDFs_ChatApp_AI-Agent
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```txt
GOOGLE_API_KEY=<your-google-api-key>
```

Run the app:

```bash
streamlit run app.py
```

---

## 💡 Usage

1. Upload PDFs in the sidebar and click **Submit & Process**.
2. Ask questions in the text input field; responses appear in real-time.
3. The chatbot retrieves answers using **HuggingFace embeddings** + **Google Gemini**.

---

#### Made with ❤️ by [Abhikant Kumar](https://github.com/abhikant12)

⭐ If you like this project, support it on GitHub!

