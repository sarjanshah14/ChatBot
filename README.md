# HR Chatbot Implementations

This repository features two distinct approaches to building an HR chatbot powered by Large Language Models (LLMs). The chatbots are designed to answer employee questions based on the `emp.pdf` (Employee Handbook) file.

## 1. Cloud-Native Chatbot (`app.py`)

This version leverages **Google's Gemini API** to process the document directly.

*   **Technology Stack**: FastAPI, Google GenAI SDK (`google-genai`).
*   **Model**: `gemini-2.0-flash-lite`.
*   **Mechanism**:
    *   On startup, the application uploads `emp.pdf` to Google's servers.
    *   When a question is asked, the API sends both the file reference and the prompt to Gemini.
    *   This approach avoids complex local indexing by relying on Gemini's large context window to "read" the entire document for every query (or use efficient caching).
*   **Pros**: Simple architecture, high accuracy, zero local model maintenance.
*   **Cons**: Requires an API key and internet connection; data leaves the local environment.

### Usage
```bash
# Ensure GEMINI_API_KEY is set in your .env file
python app.py
```
*   Access the UI at: `http://localhost:8000`

---

## 2. Local RAG Chatbot (`app_backup.py` & `chatbot.py`)

This version implements a **Retrieval-Augmented Generation (RAG)** pipeline, designed to run locally or with self-hosted models.

*   **Technology Stack**: FastAPI, LangChain, ChromaDB, Ollama.
*   **Model**: Uses a local LLM via Ollama (e.g., Llama 3, Mistral) and HuggingFace embeddings.
*   **Mechanism**:
    1.  **Ingestion**: Extracts text from `emp.pdf`, splits it into semantic chunks, and creates vector embeddings.
    2.  **Storage**: Stores embeddings in a local **ChromaDB** vector store (`chroma_db/`).
    3.  **Retrieval**: Uses an **Adaptive Retrieval** strategy:
        *   Classifies questions as "narrow", "medium", or "broad".
        *   Adjusts the number of chunks retrieved ($k$) based on classification.
        *   Uses **Maximal Marginal Relevance (MMR)** to ensure diverse search results.
    4.  **Generation**: Sends the retrieved context + question to the local Ollama model to generate an answer.
    5.  **Caching**: Implements an LRU cache to store and instantly serve frequent answers.

*   **Pros**: Complete data privacy (runs locally), no API costs, highly customizable retrieval logic (MMR, adaptive $k$).
*   **Cons**: Higher system requirements (RAM/GPU for Ollama), more complex setup.

### Usage
Prerequisite: Ensure [Ollama](https://ollama.com/) is installed and running with your desired model.

```bash
# Start the RAG server
python app_backup.py
```
*   Access the API documentation at: `http://localhost:8000/docs`

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Environment Variables**:
    Create a `.env` file with the following keys:
    ```env
    GEMINI_API_KEY=your_api_key_here
    ```

3.  **Data**:
    Ensure the `emp.pdf` file is present in the root directory.
