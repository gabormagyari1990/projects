import os
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import List, Dict
import openai
from dotenv import load_dotenv
import numpy as np
from pathlib import Path
import json

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create knowledge directory if it doesn't exist
KNOWLEDGE_DIR = Path("knowledge")
KNOWLEDGE_DIR.mkdir(exist_ok=True)

# Store document embeddings
document_embeddings = {}

def get_embedding(text: str) -> List[float]:
    """Get embeddings for a given text using OpenAI's API."""
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_relevant_content(query: str, threshold: float = 0.7) -> str:
    """Find relevant content from the knowledge base using embeddings."""
    if not document_embeddings:
        return ""
    
    query_embedding = get_embedding(query)
    relevant_content = []
    
    for doc_name, doc_data in document_embeddings.items():
        similarity = cosine_similarity(query_embedding, doc_data["embedding"])
        if similarity > threshold:
            relevant_content.append(doc_data["content"])
    
    return "\n".join(relevant_content)

@app.post("/upload")
async def upload_document(file: UploadFile):
    """Upload a document to the knowledge base."""
    try:
        content = await file.read()
        content = content.decode("utf-8")
        
        # Save the file
        file_path = KNOWLEDGE_DIR / file.filename
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        # Generate and store embedding
        embedding = get_embedding(content)
        document_embeddings[file.filename] = {
            "embedding": embedding,
            "content": content
        }
        
        return {"message": "Document uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(data: Dict):
    """Handle chat interactions."""
    try:
        user_message = data.get("message", "")
        if not user_message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        # Find relevant content from knowledge base
        context = find_relevant_content(user_message)
        
        # Prepare system message
        system_message = """You are a helpful customer service assistant. 
        Use the following context to answer the user's questions. 
        If you can't find relevant information in the context, 
        provide a general helpful response."""
        
        if context:
            system_message += f"\n\nContext:\n{context}"
        
        # Get response from OpenAI
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
        )
        
        return {"response": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve static files
app.mount("/", StaticFiles(directory="static", html=True), name="static")

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
