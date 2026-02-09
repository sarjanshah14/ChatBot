# HR Chatbot

A voice-enabled HR assistant powered by Google's Gemini API, designed to answer employee questions based on the `emp.pdf` (Employee Handbook) file.

## Architecture

This application leverages **Google's Gemini API** to process the employee handbook directly.

*   **Technology Stack**: FastAPI, Google GenAI SDK (`google-generativeai`).
*   **Model**: `gemini-2.0-flash`.
*   **Mechanism**:
    *   On startup, the application uploads `emp.pdf` to Google's servers.
    *   When a question is asked, the API sends both the file reference and the prompt to Gemini.
    *   This approach avoids complex local indexing by relying on Gemini's large context window.

## Files

- `index.html`: The main standalone chatbot interface.
- `widget.html`: An embeddable version of the chatbot for existing portals.
- `app.py`: The FastAPI backend handling the chat logic.
- `emp.pdf`: The source document for HR policies.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Environment Variables**:
    Create a `.env` file with:
    ```env
    GEMINI_API_KEY=your_api_key_here
    ```

3.  **Run Locally**:
    ```bash
    python app.py
    ```
    - Standalone: `http://localhost:8000/`
    - Widget: `http://localhost:8000/widget.html` (Note: Backend serves index.html at root)
