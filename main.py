# CIBA Health - RAG Pipeline Assessment
# ========================================
# 
# Welcome! Your task is to build a clinical Q&A system using RAG.
# 
# TIME: 60 minutes
# 
# REQUIREMENTS:
# 1. Load and chunk the clinical protocol document (sample_protocol.txt)
# 2. Generate embeddings and store them for retrieval
# 3. Implement the /ask endpoint to answer questions with source citations
# 4. Implement the /health endpoint for basic health check
#
# EVALUATION CRITERIA:
# - Chunking strategy (semantic vs arbitrary)
# - Retrieval quality
# - Code structure and error handling
# - Healthcare context awareness
#
# NOTES:
# - OpenRouter API key is available in Secrets (already configured as OPENROUTER_API_KEY)
# - OpenRouter base URL: https://openrouter.ai/api/v1
# - You may use langchain, chromadb, or build from scratch
# - For embeddings: use sentence-transformers (free, local) since OpenRouter doesn't support embeddings
# - For LLM: use OpenRouter with model like "anthropic/claude-3-haiku" or "openai/gpt-4o-mini"
# - Focus on working code first, then refine
#
# Good luck!

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Any
import chromadb
import uuid
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

app = FastAPI(
    title="CIBA Clinical Q&A",
    description="RAG-powered clinical protocol assistant"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# MODELS
# =============================================================================

class Query(BaseModel):
    question: str
    audience: Optional[str] = "clinician"  # "clinician" or "patient"

class Answer(BaseModel):
    answer: str
    sources: list[str]
    confidence: Optional[float] = None

class HealthCheck(BaseModel):
    status: str
    chunks_cnt : int
    documents_loaded: int
    embedding_model: str

# =============================================================================
# TODO: Implement your RAG components below
# =============================================================================

# Hint: You'll need to:
# 1. Load the document from sample_protocol.txt
# 2. Chunk it appropriately (consider clinical context)
# 3. Create embeddings (OpenAI or sentence-transformers)
# 4. Store in a vector database (ChromaDB recommended for simplicity)
# 5. Implement retrieval logic

# Your code here...
class DocumentProcessor:
    """Handles the ingestion and transformation of documents."""
    
    @staticmethod
    def process_document(filepath: str, chunk_size: int = 200, chunk_overlap: int = 50):
        """
        Loads and chunks a document using a recursive strategy to maintain context.
        """
        if not os.path.exists(filepath):
            print(f"Error: File not found at {filepath}")
            return []

        try:
            # Ingestion
            loader = TextLoader(filepath)
            document = loader.load()
            print(f"Document loaded from path: {filepath}")

            # Transformation 
            # Using a larger overlap by default, because for medical protocols, context is critical.
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )

            chunks = text_splitter.split_documents(document)
            print(f"Created {len(chunks)} chunks from {filepath}")
            
            return chunks

        except Exception as e:
            print(f"Failed to process document: {str(e)}")
            return []

#chunks = Document_processing.process_document("sample_protocol.txt")
#print(chunks)

class EmbeddingManager:
    """
    Handles the transformation of text data into numerical vectors.
    Strictly isolated from database and API logic.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding manager.
        """
        self.model_name = model_name
        self.model = self._load_model()

    def _load_model(self):
        try:
            print(f"Loading embedding model: {self.model_name}")
            return SentenceTransformer(self.model_name)
        except Exception as e:
            print(f"Error loading embedding model: {str(e)}")
            return None
        
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        '''
        converts text to a vector embedding
        Uses: To convert text data into embeddings to store in db, generating query embedding  
        '''
        if not self.model:
            raise ValueError("Model not loaded")
        print(f"generate_embeddings for {len(texts)}")
        embeddings = self.model.encode(texts, show_progress_bar = False)
        print("Embeddings generated")
        return embeddings



class VectorManager:
    """
    Manages the vector DB lifecycle and document indexing.
    Responsibility: Storage, metadata management, and retrieval.
    """
    def __init__(self, collection_name: str = "clinical_protocol", persist_dir: str = "./chroma_db"):
        self.collection_name = collection_name
        self.persist_directory = persist_dir
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        """Initializes the persistent ChromaDB client and collection."""
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Clinical protocol embeddings for RAG"}
            )
            print(f"Vector store initialized. Collection: {self.collection_name}")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """
        Pairs LangChain documents with their corresponding embeddings and adds to Chroma.
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents does not match number of embeddings")
        
        ids = []
        metadatas = []
        documents_text = []
        # Convert numpy array to list of lists for Chroma compatibility
        embeddings_list = embeddings.tolist()

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # Generate unique ID for each chunk
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            # Metadata extraction and enhancement
            metadata = dict(doc.metadata) if hasattr(doc, 'metadata') else {}
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)

            documents_text.append(doc.page_content)

        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text
            )
            print(f"Successfully added {len(documents)} documents to vector store")
        except Exception as e:
            print(f"Error adding to vector store: {e}")
            raise

    def query(self, query_embedding: List[float], k: int = 5):
        """
        Searches the collection using a pre-generated query vector.
        """
        if not self.collection:
            return []
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        return results

    def get_count(self) -> int:
        """Returns total number of chunks in the collection."""
        return self.collection.count() if self.collection else 0
    

class RAGRetriever:
    """Handles query-based retrieval from the vector store"""
    
    def __init__(self, vector_store: VectorManager, embedding_manager: EmbeddingManager):
        """
        Initialize the retriever
        Args:
            vector_store: Vector store containing document embeddings
            embedding_manager: Manager for generating query embeddings
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.01) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query
        Args:
            query: The search query
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold
        Returns:
            List of dictionaries containing retrieved documents and metadata
        """
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top K: {top_k}, Score threshold: {score_threshold}")
        
        # Generate query embedding
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        
        # Search in vector store
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            
            # Process results
            retrieved_docs = []
            
            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]
                
                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    # Convert distance to similarity score 
                    similarity_score = 1 - distance
                    
                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'distance': distance,
                            'rank': i + 1
                        })
                
                print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
            else:
                print("No documents found")
            
            return retrieved_docs
            
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []




class LLM_Manager:
    def __init__(self, model_name: str = "llama-3.1-8b-instant", api_key: str =None):
        """
        Initialize Groq LLM
        
        Args:
            model_name: Groq model name
            api_key: Groq API key (environment variable)
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        
        if not self.api_key:
            raise ValueError("Groq API key is required. Set GROQ_API_KEY environment variable or pass api_key parameter.")
        
        self.llm = ChatGroq(
            groq_api_key=self.api_key,
            model_name=self.model_name,
            temperature=0.1,
            max_tokens=1024
        )
        
        print(f"Initialized Groq LLM with model: {self.model_name}")

    def generate_response(self, query: str, context: str, audience: str = "patient", max_length: int = 500) -> str:
        """
        Generate response using retrieved context
        
        Args:
            query: User question
            context: Retrieved document context
            max_length: Maximum response length
            
        Returns:
            Generated response string
        """
        
        # Create prompt template
        prompt_template = PromptTemplate(
            input_variables=["query","context", "audience"],
            template=f"""
            SYSTEM ROLE:
            You are a highly accurate Clinical Protocol Advising Assistant. Your goal is to provide evidence-based answers for {audience} derived strictly from the provided technical context.

            CONTEXT PROVIDED:
            ---------------------
            {context}
            ---------------------

            INSTRUCTIONS:
            1. Use ONLY the context provided above to answer the question. 
            2. If the answer is not explicitly stated in the context, respond with: "I'm sorry, but the provided protocol documentation does not contain information to answer that question."
            3. Do not use outside medical knowledge or assumptions.
            5. If medical dosages or risks are mentioned, ensure they are copied exactly as written.

            QUESTION: {query}

            ANSWER:""")
        
        # Format the prompt
        formatted_prompt = prompt_template.format(context=context, query=query, audience=audience)
        
        try:
            # Generate response
            messages = [HumanMessage(content=formatted_prompt)]
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Internal RAG Error: {str(e)}")    



# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """
    TODO: Return system health status
    - Confirm documents are loaded
    - Report number of chunks
    - Report embedding model used
    """
    # Your implementation here
    try:
        count = vector_store.get_count()
        return HealthCheck(
            status = "healthy" if count > 0 else "uninitialized",
            documents_loaded = count,
            chunks_count = vector_store.get_count(),
            embedding_model = embedding_manager.model_name
        )
    except Exception as e:
        raise f"error with health_check: {e}"

@app.post("/ask", response_model=Answer)
async def ask_question(query: Query):
    """
    TODO: Answer clinical questions using RAG
    
    Steps:
    1. Embed the question
    2. Retrieve relevant chunks
    3. Generate answer with LLM
    4. Return answer with source citations
    
    BONUS: Adjust response style based on audience (clinician vs patient)
    """
    # Your implementation here
    # connecting with vectors in vector store
    try:
        retrieved_docs = rag_retriever.retrieve(query.question)
        if not retrieved_docs:
            return Answer(answer="No relevant clinical data found.", sources=[])
        
        context_text = "\n\n-----\n\n".join([doc['content'] for doc in retrieved_docs])

    except Exception as e:
        print(f"error while retrieving the vectors: {e}")

    # generate llm output
    try:
        llm_answer = llm.generate_response(
            query= query.question, 
            context= context_text, 
            audience= query.audience
        )
        
        sources = list(set([doc['metadata'].get('source', 'Unknown') for doc in retrieved_docs]))

        return Answer(
            answer = llm_answer,
            sources = sources,
            confidence = retrieved_docs[0]['similarity_score']
        )

    except Exception as e:
        print(f"error generating answer with llm: {e}")
        raise HTTPException(status_code=500, detail=f"Internal RAG Error: {str(e)}")

@app.get("/")
async def root():
    try:
        with open("frontend/index.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return "<h1>Frontend file not found</h1><p>Ensure index.html is in a 'frontend' folder.</p>"

    return {
        "message": "CIBA Clinical Q&A API",
        "docs": "/docs",
        "endpoints": {
            "health": "GET /health",
            "ask": "POST /ask"
        }
    }


# =============================================================================
# STARTUP (optional - for preloading documents)
# =============================================================================

embedding_manager = EmbeddingManager()
vector_store = VectorManager()
rag_retriever = None
llm = None

@app.on_event("startup")
async def startup_event():
    """
    TODO (optional): Preload documents and create embeddings on startup
    """
    print("Starting CIBA Clinical Q&A...")
    # Your initialization code here
    global rag_retriever, llm
    
    # Load and Process
    chunks = DocumentProcessor.process_document("sample_protocol.txt")
    
    if chunks:
        # Embed and Store
        texts = [doc.page_content for doc in chunks]
        embeddings = embedding_manager.generate_embeddings(texts)
        vector_store.add_documents(chunks, embeddings)
        
        # Initialize Retrieval/Generation
        rag_retriever = RAGRetriever(vector_store, embedding_manager)
        llm = LLM_Manager()
        print(f"Indexed {len(chunks)} clinical chunks.")
        print(f"RaG Model is ready for your queries")
    else:
        print("Warning: No documents found to index.")
    


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
