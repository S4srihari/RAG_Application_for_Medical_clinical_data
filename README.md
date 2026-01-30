This project is a high-fidelity Clinical Q&A system designed to assist healthcare providers and patients in navigating complex clinical protocol documentation. Using a Retrieval-Augmented Generation (RAG) pipeline, the system extracts precise answers from technical documents while providing source citations to ensure medical accuracy.

## Core FeaturesSemantic Document Processing: 
Uses RecursiveCharacterTextSplitter to maintain clinical context during document chunking.Local Vector Embeddings: Leverages sentence-transformers/all-MiniLM-L6-v2 for efficient, local vectorization of clinical text.Vector Database: Utilizes ChromaDB for persistent storage and high-speed retrieval of document embeddings.Audience-Aware Generation: Powered by Groq-hosted LLMs (e.g., Llama 3.1) to tailor responses for either "Clinician" or "Patient" audiences.Interactive React Frontend: A modern, responsive chat interface built with React and Tailwind CSS.

## System Architecture
The system operates through four primary layers:Ingestion: Processes sample_protocol.txt into overlapping chunks.Indexing: Generates numerical vectors and stores them in a local Chroma database.Retrieval: Finds the top-k most relevant clinical segments based on a user's query.Augmentation: Passes the query and retrieved context to the LLM to generate a cited response.

## Setup Instructions

### 1. Prerequisites
Python 3.9
A Groq API Key (for LLM generation).

### 2. Environment SetupBash# Clone the repository

git clone <https://github.com/S4srihari/RAG_backed_Chatbot_for_Clinical_data.git>
cd project_folder

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn langchain chromadb sentence-transformers langchain-groq python-dotenv
### 3. Configuration
Create a .env file in the root directory with:
GROQ_API_KEY=your_api_key_here

## Running the Application
Start the Backend Server:The system will automatically load and index sample_protocol.txt on startup.

Bash/CMD/Powershell
python main.py

Access the Interface:Navigate to http://127.0.0.1:8000 in your browser.

## Technical Stack
Backend: FastAPI
LLM Orchestration: LangChain
Embeddings: Sentence-Transformers (Local)
Frontend: React (v18), Tailwind CSS, FontAwesomeDatabase: ChromaDB