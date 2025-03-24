from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import psycopg2
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from together import Together
import os
from connection import DB_CONFIG
# Initialize Flask app
app = Flask(__name__)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")



def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)



@app.route("/")
def health():
        return jsonify({"message": "Server is running"})
# Document Ingestion API
@app.route("/upload", methods=["POST"])
def upload_document():
    try:
        data = request.json
        document_id = data.get("document_id")
        content = data.get("content")
        user_id = data.get("uid")  # User ID is now required
        
        if not document_id or not content or not user_id:
            return jsonify({"error": "Missing details"}), 400
        
        embedding = model.encode(content).tolist()
        
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO documents (document_id, user_id, content, embedding) VALUES (%s, %s, %s, %s::vector)",
            (document_id, user_id, content, embedding)
        )
        conn.commit()
        cur.close()
        conn.close()
        
        return jsonify({"message": "Document uploaded successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Q&A APIfrom sklearn.metrics.pairwise import cosine_similarity

@app.route("/qa", methods=["POST"])
def rag_qa():
    try:
        data = request.json
        question = data.get("question")
        user_id = data.get("uid")
        
        if not question or not user_id:
            return jsonify({"error": "Missing details"}), 400
        
        # Generate embedding for the question
        question_embedding = np.array(model.encode(question)).reshape(1, -1)  # Reshape for sklearn

        # Fetch user's documents from the database
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT content, embedding FROM documents WHERE user_id = %s", (user_id,))
        results = cur.fetchall()
        cur.close()
        conn.close()

        if not results:
            return jsonify({"answer": "No relevant document found"})

        # Prepare embeddings and contents
        contents = []
        embeddings = []
        for content, embedding_str in results:
            try:
                embedding = np.array(json.loads(embedding_str)).reshape(1, -1)  # Convert from string to float array
                contents.append(content)
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error converting embedding: {e}")

        # Compute cosine similarity between question and document embeddings
        similarities = [cosine_similarity(question_embedding, emb)[0][0] for emb in embeddings]

        # Find the best match
        best_index = np.argmax(similarities)
        best_match = contents[best_index] if similarities[best_index] > 0 else None

        return jsonify({"answer": gen_response(question,[best_match]) if best_match else "No relevant document found"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
# Document Selection API
@app.route("/documents", methods=["POST"])
def get_documents():
    try:
        data = request.get_json()

        document_ids = data["document_ids"]  # Expecting a list of document IDs
        user_id = data["uid"]
        question=data["question"]
        # Validate input
        if not document_ids or not user_id or not question:
            return jsonify({"error": "Missing 'document_ids' or 'user_id' in request body"}), 400

        # Ensure document_ids is a non-empty list
        if not isinstance(document_ids, list) or not document_ids:
            return jsonify({"error": "'document_ids' must be a non-empty list"}), 400

        conn = get_db_connection()
        cur = conn.cursor()

        # Query only the requested documents for the specific user
        query = """
        SELECT document_id, content FROM documents 
        WHERE document_id IN %s AND user_id = %s
        """
        cur.execute(query, (tuple(document_ids), user_id))
        documents = [{"document_id": row[0], "content": row[1]} for row in cur.fetchall()]
        print(documents)
        cur.close()
        conn.close()

        return jsonify({
            "message":
            gen_response(question,documents)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500




def gen_response(query, documents):
    # context = "\n".join(documents)

    context = "\n".join(doc["content"] for doc in documents if "content" in doc)

    print(context)
    client=Together(api_key=os.getenv("T_KEY"))
    # client=OpenAI(api_key=os.getenv("API_KEY"),base_url=os.getenv("API_URL"))
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3",
        messages= [
            {"role": "system", "content": "You are an AI assistant that provides answers based on retrieved documents."},
            {"role": "user", "content": f"Use the following documents to answer the question '{query}':\n{context}"}
        ],
    
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    app.run(port=5000,debug=True)
