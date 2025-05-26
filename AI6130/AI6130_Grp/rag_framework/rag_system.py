import os
import torch
import faiss
import numpy as np
import gc
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Import the package logger
from .utils import logger

# For document loading and processing
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# For embeddings
from sentence_transformers import SentenceTransformer

class RAGSystem:

  def __init__(self, embedding_model_name: str = "BAAI/bge-small-en-v1.5", chunk_size: int = 1500, chunk_overlap: int = 300):

    #init param
    self.embedding_model_name = embedding_model_name
    self.chunk_size = chunk_size
    self.chunk_overlap = chunk_overlap

    #init storage for document chunks and metadata

    self.documents = []
    self.meatadatas = []
    self.embeddings = None
    self.embedding_model = None
    self.index = None

    #load embedding model
    self._load_embedding_model()

    logger.info(f"Initialized RAG System with {embedding_model_name}")
    logger.info(f" Chunk size: {chunk_size}, Overlap is {chunk_overlap}")

  def _load_embedding_model(self):
    #Load embedding model
    if torch.cuda.is_available():
      device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
      device = "mps"  # For Apple Silicon Macs
    else:
      device = "cpu"

    try:
      logger.info(f"Loading embedding model: {self.embedding_model_name}")
      self.embedding_model = SentenceTransformer(self.embedding_model_name, device=device)
      logger.info(f"Embedding model laoded!")
    except Exception as e:
        logger.error(f"Failed to load embedding model: {str(e)}")
        raise

  def load_documents(self, papers_dir: str)-> Dict[str, str]:

    papers_dir = Path(papers_dir)
    if not papers_dir.exists():
      logger.error(f"Directory not found: {papers_dir}")
      return {}

    paper_files = [ f for f in os.listdir(papers_dir) if f.endswith('.pdf')]

    if not paper_files:
      logger.error(f"No PDF files found in {papers_dir}")
      return{}


    paper_contents = {}

    for paper_file in paper_files:
      paper_path = os.path.join(papers_dir, paper_file)
      logger.info(f"Loading document: {paper_path}")

      try:

        #Use PyPDFLoader to load pdf doc
        loader = PyPDFLoader(paper_path)
        doc_sections = loader.load()

        if doc_sections:
          content = "\n\n".join([section.page_content for section in doc_sections]) #double newlines to preserve para.
          paper_contents[paper_file] = content
          logger.info(f"Successfully loaded {paper_file} with ({len(content)} characters)")
        else:
          logger.warning(f"No content loaded from {paper_file}")
      except Exception as e:
        logger.error(f"Error loading file {paper_file}: {str(e)}")

    logger.info(f"Loaded {len(paper_contents)} documents")
    return paper_contents

  def process_documents(self, paper_contents: Dict[str, str]):
    logger.info(f"Start Processing Doc....")

    #init text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, separators=["\n\n", "\n","","."," "])

    self.documents = []
    self.metadata  = []

    #Process each paper

    for paper_name, content in paper_contents.items():
      logger.info(f"Splitting document: {paper_name}")

      chunks = text_splitter.split_text(content)

      #Store chunks w metadata

      for i, chunk in enumerate(chunks):
        self.documents.append(chunk)
        self.metadata.append({
            "source": paper_name,
            "chunk_id": i,
            "total_chunks": len(chunks)
        })

      logger.info(f"Created {len(chunks)} chunks from {paper_name}")


    logger.info(f"Processed {len(paper_contents)} documents into {len(self.documents)} chunks")

  def create_embedding(self, batch_size=32):
    if not self.documents:
      logger.erro("No documents to create embeddings for")
      return

    logger.info(f"Creating embeddings for {len(self.documents)} chunks...")

    try:
      #generate embeddings for chunks
      #This convert text chunks into numerical vectors
      self.embeddings = self.embedding_model.encode(self.documents, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)

      #Convert to float32 for FAISS

      self.embeddings = self.embeddings.astype('float32')

      # Memory cleanup if using GPU
      if hasattr(self, 'device') and self.device in ["cuda", "mps"]:
          torch.cuda.empty_cache() if self.device == "cuda" else None
          gc.collect()  # Call garbage collector

      logger.info(f"Created embeddings with shape: {self.embeddings.shape}")
    except Exception as e:
      logger.error(f"Error creating embeddings: {str(e)}")
      raise


  def build_faiss_index(self):

    if self.embeddings is None or len(self.embeddings) == 0:
      logger.error("No embeddings to build index with")
      return

    try:
      #Get embedding dimension
      dimension = self.embeddings.shape[1]

      #CPU version
      logger.info(f"Building FAISS index using CPU with dimension {dimension}")
      self.index = faiss.IndexFlatL2(dimension)
      self.index.add(self.embeddings)

      logger.info(f"Built FAISS index with {self.index.ntotal} vectors")
    except Exception as e:
      logger.error(f"Error building FAISS index: {str(e)}")
      raise

  def query(self, question: str , top_k: int = 3) -> List[Dict[str, Any]]:

    if self.index is None:
      logger.error("No index available for querying")
      return[]

    try:

      #create embedding for question
      question_embedding = self.embedding_model.encode([question])
      question_embedding = np.array(question_embedding).astype('float32')

      #search the index
      distances, indices = self.index.search(question_embedding, top_k)

      #get relevant doc & metadata
      results = []
      for i, idx in enumerate(indices[0]):
        if idx < len(self.documents):
          results.append({
              "content": self.documents[idx],
              "metadata": self.metadata[idx],
              "score": float(distances[0][i])
              })
      logger.info(f"Query: '{question}' returned {len(results)}")
      return results
    except Exception as e:
      logger.error(f"Error querying index: {str(e)}")
      return[]

  def ingest_and_index(self, papers_dir: str, batch_size: int = 32)-> bool:

    try:

      #Step 1: Load Docs
      paper_contents = self.load_documents(papers_dir)
      if not paper_contents:
        logger.error(f" No documents loaded. Aborting")
        return False

      #Step 2: Process documents into chunks
      self.process_documents(paper_contents)

      #Step 3: Create embeddings
      self.create_embedding(batch_size=batch_size)

      #Step 4: Build FAISS index
      self.build_faiss_index()

      logger.info("Successfully completed ingestion and indexing")
      return True
    except Exception as e:
      logger.error(f" Error in ingest and index: {str(e)}")
      return False

def run_rag_example(papers_dir: str, use_gpu: bool = False,batch_size: int = 32):
  #papers_dir = get_papers_dir()
  #use_gpu = torch.cuda.is_available()

  #Initialize RAG
  rag_system = RAGSystem(
      embedding_model_name = "BAAI/bge-small-en-v1.5",
      chunk_size = 1500,
      chunk_overlap = 300
  )

    #Ingest and index doc
  success = rag_system.ingest_and_index(papers_dir, batch_size=batch_size)

  if success:
    example_questions = [
      "What is the equation defining force according to Newton?",
      "What is Newton’s First Law also called?",
      "How is force defined according to Newton?",
      "What transformation preserves Newton’s laws?"
    ]

    #Run queries and display results:
    for question in example_questions:
      print("\n" + "="*80)
      print(f"Question: {question}\n")
      print("="*80)

      results = rag_system.query(question, top_k=4)

      for i, result in enumerate(results):
        print(f"\nRESULT {i+1} (Score: {result['score']:.4f})")
        print(f"Source: {result['metadata']['source']}, " + f"Chunk: {result['metadata']['chunk_id']+1}/{result['metadata']['total_chunks']}" )
        print("-"*80)
        print(result['content'][:800] + "...." if len(result['content']) > 800 else result['content'])
        print("-"*80)

  return
