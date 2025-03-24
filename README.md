# Document Management and RAG-based Q&A API

This is a Flask-based API for document management and retrieval-augmented generation (RAG) Q&A using vector embeddings and an LLM backend.

## Features

- **Document Uploading:** Store documents along with embeddings in a PostgreSQL database.
- **Document Retrieval:** Fetch documents by document ID and user ID.
- **Retrieval-Augmented Generation (RAG):** Answer questions using the most relevant documents based on cosine similarity.
- **LLM Integration:** Uses `deepseek-ai/DeepSeek-V3` via the Together API for answering questions.

## Installation

### Prerequisites

- Python 3.8+
- PostgreSQL
- Virtual environment (optional but recommended)

### Steps

1. Clone the repository:
   ```sh
   git clone https://github.com/nikhilreddy2k22/auro-assignment.git
   cd auro-assignment
   ```
2. Create a virtual environment and activate it:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Set up environment variables:
   Create a `.env` file in the project directory and add:
   ```sh
   DB_HOST=<your_database_host>
   DB_PORT=<your_database_port>
   DB_NAME=<your_database_name>
   DB_USER=<your_database_user>
   DB_PASSWORD=<your_database_password>
   T_KEY=<your_together_api_key>
   ```
5. Run the application:
   ```sh
   python app.py
   ```
   The server will start at `http://127.0.0.1:5000/`.

## API Endpoints

### Health Check

- **`GET /`**
  - **Response:** `{ "message": "Server is running" }`

### Document Upload

- **`POST /upload`**
  - **Request Body:**
    ```json
    {
      "document_id": 1,
      "content": "Sample text",
      "uid": 123
    }
    ```
  - **Response:** `{ "message": "Document uploaded successfully" }`

### Retrieve Documents

- **`POST /documents`**
  - **Request Body:**
    ```json
    {
      "document_ids": [1, 2],
      "uid": 123,
      "question": "What is the summary?"
    }
    ```
  - **Response:** `{ "message": "Generated response from documents" }`

### Question Answering (RAG)

- **`POST /qa`**
  - **Request Body:**
    ```json
    {
      "question": "What is AI?",
      "uid": 123
    }
    ```
  - **Response:** `{ "answer": "AI is..." }`

## Dependencies

The dependencies are listed in `requirements.txt`:
