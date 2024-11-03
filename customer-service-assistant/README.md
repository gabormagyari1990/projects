# Customer Service Assistant

A simple chatbot application that uses document-based knowledge to provide customer service assistance. The application uses OpenAI's GPT-4 for generating responses and embeddings for finding relevant context from uploaded documents.

## Features

- Upload documents to create a knowledge base
- Chat interface for user interactions
- Document similarity search using OpenAI embeddings
- Real-time responses using GPT-4
- Simple and intuitive web interface

## Setup

1. Clone the repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and add your OpenAI API key:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` and add your OpenAI API key.

## Running the Application

1. Start the server:
   ```bash
   python main.py
   ```
2. Open your browser and navigate to `http://localhost:8000`

## Usage

1. Upload Documents:

   - Use the upload section on the right to add documents to the knowledge base
   - Supported file formats: .txt, .md, .doc, .docx

2. Chat Interface:
   - Type your questions in the chat input
   - The assistant will use the uploaded documents as context to provide relevant answers
   - If no relevant context is found, it will provide general helpful responses

## How it Works

1. Document Processing:

   - When documents are uploaded, they are stored in the knowledge directory
   - The system generates embeddings for each document using OpenAI's embedding model

2. Chat Process:
   - When a user sends a message, the system generates an embedding for the query
   - It finds relevant content from the knowledge base using cosine similarity
   - The relevant content is used as context for GPT-4 to generate an appropriate response

## Notes

- The application uses a simple file-based storage system for documents
- Embeddings are stored in memory and will reset when the server restarts
- Make sure your OpenAI API key has access to both the embedding and GPT-4 models
