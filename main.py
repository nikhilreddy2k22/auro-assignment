from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import psycopg2
import ollama
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v6")

USER = os.getenv("user")
PASSWORD = os.getenv("password")
HOST = os.getenv("host")
PORT = os.getenv("port")
DBNAME = os.getenv("dbname")
# Database connection
DB_CONFIG = {
    "dbname": DBNAME,
    "user": USER,
    "password": PASSWORD,
    "host": HOST,
    "port": PORT
}

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

# Document Ingestion API
@app.route("/ingest", methods=["POST"])
def ingest_document():
    try:
        data = request.json
        document_id = data.get("document_id")
        content = data.get("content")
        
        if not document_id or not content:
            return jsonify({"error": "Missing document_id or content"}), 400
        
        embedding = model.encode(content).tolist()
        
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO documents (document_id, content, embedding) VALUES (%s, %s, %s)",
            (document_id, content, np.array(embedding))
        )
        conn.commit()
        cur.close()
        conn.close()
        
        return jsonify({"message": "Document ingested successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Q&A API
@app.route("/qa", methods=["POST"])
def rag_qa():
    try:
        data = request.json
        question = data.get("question")
        selected_docs = data.get("selected_docs", [])
        
        if not question:
            return jsonify({"error": "Missing question"}), 400
        
        question_embedding = model.encode(question).tolist()
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        if selected_docs:
            cur.execute("SELECT content, embedding FROM documents WHERE document_id = ANY(%s)", (selected_docs,))
        else:
            cur.execute("SELECT content, embedding FROM documents")
        
        results = cur.fetchall()
        cur.close()
        conn.close()
        
        best_match = None
        best_score = -1
        
        for content, embedding in results:
            embedding = np.array(embedding)
            score = np.dot(question_embedding, embedding) / (np.linalg.norm(question_embedding) * np.linalg.norm(embedding))
            if score > best_score:
                best_score = score
                best_match = content
        
        return jsonify({"answer": best_match if best_match else "No relevant document found"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Document Selection API
@app.route("/documents", methods=["GET"])
def get_documents():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT document_id, content FROM documents")
        documents = [{"document_id": row[0], "content": row[1]} for row in cur.fetchall()]
        cur.close()
        conn.close()
        return jsonify(documents)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    



def get_embedding(doc):

    #Loading model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v6')
    
    # Generate embedding
    vec = model.encode(doc, convert_to_numpy=True)

    return vec


def retrive(query,selected_doc):

    return []


def retrive_doc(query):

    ans = []
    
    return ans

def gen_response(query,documents):
     
    context = "\n".join(documents)

    # Query Ollama with retrieved document context
    response = ollama.chat(
        model="mistral",  # You can change to "llama3" or another model
        messages=[
            {"role": "system", "content": "You are an AI assistant that provides answers based on retrieved documents."},
            {"role": "user", "content": f"Use the following documents to answer the query '{query}':\n{context}"}
        ]
    )
    return response["message"]["content"]

if __name__ == "__main__":
    app.run(debug=True)
