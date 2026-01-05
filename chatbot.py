from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import PyPDF2
import os
import config
import re

class HandbookChatbot:
    def __init__(self):
        self.vectorstore = None
        self.llm = None
        self.qa_chain = None
        self.is_initialized = False
    
    def extract_pdf_text(self):
        """Extract text from PDF"""
        print("📄 Extracting text from PDF...")
        text = ""
        with open(config.PDF_PATH, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for i, page in enumerate(pdf_reader.pages):
                text += page.extract_text()
                if (i + 1) % 10 == 0:
                    print(f"   Processed {i + 1} pages...")
        print(f"✅ Extracted {len(text)} characters")
        return text
    
    def create_chunks(self, text):
        """
        Split text into chunks optimized for policy documents.
        Larger chunks with better overlap to preserve section context.
        """
        print("✂️ Splitting into chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            # Prioritize natural breaks in policy documents
            separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_text(text)
        print(f"✅ Created {len(chunks)} chunks")
        return chunks
    
    def classify_question_breadth(self, question: str) -> str:
        """
        Classify question as narrow, medium, or broad based on linguistic patterns.
        This determines how many chunks to retrieve.
        
        NO keyword matching - uses linguistic structure only.
        """
        question_lower = question.lower().strip()
        
        # Broad indicators: plural, "all", "what are", list requests
        broad_patterns = [
            r'\ball\b',
            r'\bevery\b',
            r'\blist\b',
            r'^what are',
            r'^tell me about all',
            r'^what.*benefits',  # "What benefits" tends to be broad
            r'^explain.*policies',
            r'give me.*list',
            r'what.*types of',
            r'how many.*types',
        ]
        
        # Narrow indicators: specific singular nouns, "the", definite articles
        narrow_patterns = [
            r'^what is the\b',
            r'^how.*do i\b',
            r'^when.*is\b',
            r'^where.*is\b',
            r'specific.*about',
            r'exactly.*what',
        ]
        
        # Check patterns
        for pattern in broad_patterns:
            if re.search(pattern, question_lower):
                return "broad"
        
        for pattern in narrow_patterns:
            if re.search(pattern, question_lower):
                return "narrow"
        
        # Default: medium breadth
        return "medium"
    
    def retrieve_with_scores(self, question: str, k: int):
        """
        Retrieve chunks with similarity scores for confidence assessment.
        Returns: list of (chunk, score) tuples
        """
        # Use similarity_search_with_relevance_scores for score access
        results = self.vectorstore.similarity_search_with_relevance_scores(
            question, 
            k=k
        )
        return results  # Returns [(Document, score), ...]
    
    def adaptive_retrieval(self, question: str):
        """
        Dynamically retrieve appropriate number of chunks based on question breadth.
        Returns: (context_string, confidence_level)
        """
        breadth = self.classify_question_breadth(question)
        
        # Determine k based on breadth
        if breadth == "narrow":
            k = config.RETRIEVAL_K_MIN
        elif breadth == "broad":
            k = config.RETRIEVAL_K_MAX
        else:
            k = config.RETRIEVAL_K_DEFAULT
        
        # Retrieve with scores
        results = self.retrieve_with_scores(question, k)
        
        if not results:
            return "", "none"
        
        # Analyze score distribution
        scores = [score for _, score in results]
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)
        
        # Determine confidence level
        if max_score >= config.SIMILARITY_THRESHOLD_HIGH:
            confidence = "high"  # Strong single match
        elif avg_score >= config.SIMILARITY_THRESHOLD_LOW and len([s for s in scores if s >= config.SIMILARITY_THRESHOLD_LOW]) >= 3:
            confidence = "distributed"  # Multiple moderate matches
        elif max_score >= config.SIMILARITY_THRESHOLD_LOW:
            confidence = "medium"  # Weak single match
        else:
            confidence = "low"  # Nothing relevant
        
        # Build context from all retrieved chunks
        # Filter out very low-scoring chunks
        filtered_results = [(doc, score) for doc, score in results 
                           if score >= config.SIMILARITY_THRESHOLD_LOW * 0.8]
        
        context_parts = [doc.page_content for doc, _ in filtered_results]
        context = "\n\n---\n\n".join(context_parts)
        
        return context, confidence
    
    def initialize(self, force_rebuild=False):
        """Initialize the chatbot"""
        try:
            # Check if already built
            if os.path.exists(config.CHROMA_DB_PATH) and not force_rebuild:
                print("📦 Loading existing vector database...")
                embeddings = HuggingFaceEmbeddings(
                    model_name=config.EMBEDDING_MODEL,
                    model_kwargs={'device': 'cpu'}
                )
                self.vectorstore = Chroma(
                    persist_directory=config.CHROMA_DB_PATH,
                    embedding_function=embeddings
                )
                print("✅ Loaded from disk")
            else:
                # Build from scratch
                print("🔧 Building vector database (first time only)...")
                
                # Extract and chunk
                text = self.extract_pdf_text()
                chunks = self.create_chunks(text)
                
                # Create embeddings
                print("🧠 Creating embeddings (this takes 2-3 minutes)...")
                embeddings = HuggingFaceEmbeddings(
                    model_name=config.EMBEDDING_MODEL,
                    model_kwargs={'device': 'cpu'}
                )
                
                # Create vector store
                print("💾 Storing in ChromaDB...")
                self.vectorstore = Chroma.from_texts(
                    texts=chunks,
                    embedding=embeddings,
                    persist_directory=config.CHROMA_DB_PATH
                )
                print("✅ Vector database created and saved")
            
            # Initialize LLM
            print("🤖 Connecting to Ollama...")
            self.llm = Ollama(
                model=config.LLM_MODEL,
                base_url=config.OLLAMA_BASE_URL,
                temperature=config.LLM_TEMPERATURE,
                num_ctx=config.NUM_CTX,
                num_predict=config.NUM_PREDICT,
                num_thread=config.OLLAMA_NUM_THREAD,
                num_gpu=config.OLLAMA_NUM_GPU,
            )
            
            # Create improved prompt template
            # This prompt allows summarization across sections while preventing hallucination
            template = """You are an HR assistant for Navneet Education Limited's employee handbook.

CONTEXT FROM HANDBOOK:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Answer ONLY using information from the context above
2. If the question asks about multiple things or general categories:
   - Summarize the relevant information from ALL sections provided
   - Group related information logically
   - Be comprehensive but concise
3. If the question is specific:
   - Give a direct answer with relevant details
4. If the information is NOT in the context:
   - Say "I don't have that specific information in the handbook"
   - Do NOT guess or use external knowledge
5. Always cite section names when relevant (e.g., "According to Section 4.1...")

ANSWER:"""

            self.prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )
            
            # Create LLM chain (more transparent than RetrievalQA)
            self.qa_chain = LLMChain(
                llm=self.llm,
                prompt=self.prompt
            )
            
            self.is_initialized = True
            print("✅ Chatbot ready!\n")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing: {str(e)}")
            return False
    
    def ask(self, question):
        """
        Get answer for a question using adaptive retrieval.
        Returns: dict with success, answer, and metadata
        """
        if not self.is_initialized:
            return {"error": "Chatbot not initialized"}
        
        try:
            # Truncate overly long questions
            if len(question) > 200:
                question = question[:200]
            
            # Adaptive retrieval with confidence assessment
            context, confidence = self.adaptive_retrieval(question)
            
            # Handle low confidence case
            if confidence == "none" or confidence == "low":
                return {
                    "success": True,
                    "answer": "I don't have that specific information in the employee handbook. Could you rephrase or ask about a different topic?"
                }
            
            # Generate answer using LLM
            result = self.qa_chain.run(
                context=context,
                question=question
            )
            
            return {
                "success": True,
                "answer": result.strip()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }