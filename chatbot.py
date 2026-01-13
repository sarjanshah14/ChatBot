from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from functools import lru_cache
from collections import OrderedDict
import PyPDF2
import os
import config
import re
import logging
import hashlib

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)


class LRUCache:
    """Simple LRU cache for Q&A responses"""
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key: str):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key: str, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


class HandbookChatbot:
    def __init__(self):
        self.vectorstore = None
        self.llm = None
        self.qa_chain = None
        self.is_initialized = False
        self.embeddings = None  # Reuse embedding model
        
        # Response cache
        if config.ENABLE_RESPONSE_CACHE:
            self.response_cache = LRUCache(config.CACHE_MAX_SIZE)
            logger.info(f"✅ Response cache enabled (capacity: {config.CACHE_MAX_SIZE})")
        else:
            self.response_cache = None
            logger.info("❌ Response cache disabled")
    
    def extract_pdf_text(self) -> str:
        """Extract text from PDF with error handling"""
        logger.info("📄 Extracting text from PDF...")
        
        if not os.path.exists(config.PDF_PATH):
            raise FileNotFoundError(f"PDF not found: {config.PDF_PATH}")
        
        text = ""
        try:
            with open(config.PDF_PATH, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                for i, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:  # Skip empty pages
                        text += page_text + "\n\n"
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"   Processed {i + 1}/{total_pages} pages")
                
                logger.info(f"✅ Extracted {len(text)} characters from {total_pages} pages")
        except Exception as e:
            logger.error(f"Error extracting PDF: {e}")
            raise
        
        return text
    
    def create_chunks(self, text: str) -> list:
        """
        Create optimized chunks with semantic separators.
        Uses hierarchical splitting for better context preservation.
        """
        logger.info("✂️ Creating semantic chunks...")
        
        # Custom separators for policy documents
        separators = [
            "\n\n\n",  # Major section breaks
            "\n\n",    # Paragraph breaks
            "\n",      # Line breaks
            ". ",      # Sentence ends
            "? ",      # Questions
            "! ",      # Exclamations
            "; ",      # Semi-colons
            ", ",      # Commas
            " ",       # Spaces
            ""         # Characters
        ]
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=separators,
            keep_separator=True  # Preserve separators for context
        )
        
        chunks = splitter.split_text(text)
        logger.info(f"✅ Created {len(chunks)} chunks")
        
        # Log chunk statistics
        avg_length = sum(len(c) for c in chunks) / len(chunks)
        logger.info(f"   Average chunk length: {avg_length:.0f} characters")
        
        return chunks
    
    @lru_cache(maxsize=1000)
    def classify_question_breadth(self, question: str) -> str:
        """
        Enhanced question classification with caching.
        Determines retrieval strategy based on question scope.
        """
        q_lower = question.lower().strip()
        q_words = q_lower.split()
        
        # Broad query indicators
        broad_signals = 0
        broad_patterns = [
            r'\ball\b', r'\bevery\b', r'\blist\b', r'^what are',
            r'^tell me about all', r'what.*benefits', r'explain.*policies',
            r'give me.*list', r'what.*types of', r'how many.*types',
            r'different.*options', r'available.*options', r'various'
        ]
        
        for pattern in broad_patterns:
            if re.search(pattern, q_lower):
                broad_signals += 1
        
        # Check for plural nouns (often indicates broad queries)
        if any(word.endswith('ies') or word.endswith('s') for word in q_words[-3:]):
            broad_signals += 0.5
        
        # Narrow query indicators
        narrow_signals = 0
        narrow_patterns = [
            r'^what is the\b', r'^how do i\b', r'^when is\b',
            r'^where is\b', r'specific.*about', r'exactly',
            r'^define\b', r'^explain the\b'
        ]
        
        for pattern in narrow_patterns:
            if re.search(pattern, q_lower):
                narrow_signals += 1
        
        # Check for singular definite articles
        if re.match(r'^(what is|how does|when does|where is) (the|a)\b', q_lower):
            narrow_signals += 1
        
        # Decision logic
        if broad_signals >= 2 or (broad_signals >= 1 and narrow_signals == 0):
            return "broad"
        elif narrow_signals >= 1 and broad_signals == 0:
            return "narrow"
        else:
            return "medium"
    
    def retrieve_with_mmr(self, question: str, k: int):
        """
        Retrieve chunks using similarity search (MMR not available in this ChromaDB version).
        Returns relevant chunks with distance scores converted to similarity.
        """
        # Use basic similarity search (works with all ChromaDB versions)
        try:
            # Try similarity_search_with_score first (returns distance, not relevance)
            results = self.vectorstore.similarity_search_with_score(question, k=k)
            # Convert distance to similarity: smaller distance = higher similarity
            # Normalize: similarity = 1 / (1 + distance)
            converted_results = []
            for doc, distance in results:
                # Convert distance to similarity score (0-1 range)
                similarity = 1.0 / (1.0 + abs(distance))
                converted_results.append((doc, similarity))
            return converted_results
        except Exception as e:
            logger.warning(f"Similarity search with score failed: {e}")
            # Fallback to basic search without scores
            docs = self.vectorstore.similarity_search(question, k=k)
            return [(doc, 0.5) for doc in docs]  # Assign default score
    
    def adaptive_retrieval(self, question: str):
        """
        Multi-stage adaptive retrieval with confidence scoring.
        Returns: (context, confidence, metadata)
        """
        breadth = self.classify_question_breadth(question)
        
        # Determine k based on breadth
        k_map = {
            "narrow": config.RETRIEVAL_K_MIN,
            "medium": config.RETRIEVAL_K_DEFAULT,
            "broad": config.RETRIEVAL_K_MAX
        }
        k = k_map.get(breadth, config.RETRIEVAL_K_DEFAULT)
        
        # Retrieve with MMR for diversity
        results = self.retrieve_with_mmr(question, k)
        
        if not results:
            return "", "none", {}
        
        # Analyze score distribution
        scores = [score for _, score in results]
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)
        
        high_quality_count = len([s for s in scores if s >= 0.25])  # Adjusted for distance-based scores
        
        # Enhanced confidence assessment with adjusted thresholds for distance-based scores
        # Since we converted distance to similarity, typical good scores are 0.3-0.7
        if max_score >= 0.4 and high_quality_count >= 2:
            confidence = "high"
        elif max_score >= 0.3 and high_quality_count >= 3:
            confidence = "distributed"
        elif max_score >= 0.3:
            confidence = "medium"
        elif max_score >= 0.2:
            confidence = "low"
        else:
            confidence = "none"
        
        # Filter and build context with adjusted threshold
        # For distance-based scores, lower threshold (0.15-0.2 range is acceptable)
        adaptive_threshold = max(0.15, avg_score * 0.5)  # More permissive threshold
        
        filtered_results = [
            (doc, score) for doc, score in results 
            if score >= adaptive_threshold
        ]
        
        # Build context with separators - CLEAN format without section metadata
        context_parts = []
        for i, (doc, score) in enumerate(filtered_results, 1):
            # Just add the content, no metadata in the context
            context_parts.append(doc.page_content)
        
        context = "\n\n".join(context_parts)  # Clean separator
        
        # Metadata for logging/debugging
        metadata = {
            "breadth": breadth,
            "k_retrieved": k,
            "k_used": len(filtered_results),
            "max_score": max_score,
            "avg_score": avg_score,
            "confidence": confidence
        }
        
        if config.LOG_RETRIEVAL_DETAILS:
            logger.info(f"Retrieval: {metadata}")
        
        return context, confidence, metadata
    
    def _get_cache_key(self, question: str) -> str:
        """Generate cache key from question"""
        normalized = question.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def initialize(self, force_rebuild=False) -> bool:
        """Initialize chatbot with optimizations"""
        try:
            # Initialize embeddings once
            logger.info("🧠 Loading embedding model...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=config.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'batch_size': config.BATCH_EMBEDDING_SIZE}
            )
            
            # Load or create vector store
            if os.path.exists(config.CHROMA_DB_PATH) and not force_rebuild:
                logger.info("📦 Loading existing vector database...")
                self.vectorstore = Chroma(
                    persist_directory=config.CHROMA_DB_PATH,
                    embedding_function=self.embeddings
                )
                logger.info("✅ Loaded from disk")
            else:
                logger.info("🔧 Building vector database...")
                
                # Extract and chunk
                text = self.extract_pdf_text()
                chunks = self.create_chunks(text)
                
                # Create vector store with batching
                logger.info("💾 Creating embeddings and storing...")
                self.vectorstore = Chroma.from_texts(
                    texts=chunks,
                    embedding=self.embeddings,
                    persist_directory=config.CHROMA_DB_PATH
                )
                logger.info("✅ Vector database created")
            
            # Initialize LLM with optimizations
            logger.info("🤖 Initializing LLM...")
            
            self.llm = Ollama(
                model=config.LLM_MODEL,
                base_url=config.OLLAMA_BASE_URL,
                temperature=config.LLM_TEMPERATURE,
                num_ctx=config.NUM_CTX,
                num_predict=config.NUM_PREDICT,
                num_thread=config.OLLAMA_NUM_THREAD,
                num_gpu=config.OLLAMA_NUM_GPU
                # Note: top_p, repeat_penalty, keep_alive not supported in langchain-community 0.0.x
                # Upgrade to langchain-ollama for these features
            )
            
            # Enhanced prompt template - OPTIMIZED FOR CONCISE ANSWERS
            template = """You are a helpful HR assistant for Navneet Education Limited's employee handbook.

CONTEXT FROM HANDBOOK:
{context}

EMPLOYEE QUESTION: {question}

INSTRUCTIONS:
1. Give a SHORT, DIRECT answer - maximum 2-3 sentences
2. Answer ONLY from the context above - never use external knowledge
3. If asking for a number/time period/specific fact: give ONLY that specific answer
4. DO NOT include phrases like "According to Section" or "Relevance" 
5. DO NOT repeat the question in your answer
6. DO NOT list multiple sections - synthesize into ONE clear answer
7. If the context has contradictory information, use the most specific/recent one
8. If information is NOT in context, say: "I don't have that information in the handbook"

ANSWER (keep it short and direct):"""

            self.prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )
            
            # Create chain
            self.qa_chain = LLMChain(
                llm=self.llm,
                prompt=self.prompt,
                verbose=False
            )
            
            self.is_initialized = True
            logger.info("✅ Chatbot initialized and ready!\n")
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization error: {str(e)}")
            return False
    
    def ask(self, question: str) -> dict:
        """
        Process question with caching and optimized retrieval.
        Returns: dict with success, answer, and metadata
        """
        if not self.is_initialized:
            return {"success": False, "error": "Chatbot not initialized"}
        
        try:
            # Input validation
            question = question.strip()
            if not question:
                return {"success": False, "error": "Question cannot be empty"}
            
            if len(question) > 500:
                question = question[:500]
            
            # Check cache
            if self.response_cache:
                cache_key = self._get_cache_key(question)
                cached = self.response_cache.get(cache_key)
                if cached:
                    logger.info("✓ Cache hit")
                    return {
                        "success": True,
                        "answer": cached["answer"],
                        "cached": True
                    }
            
            # Adaptive retrieval
            context, confidence, metadata = self.adaptive_retrieval(question)
            
            # Handle low confidence
            if confidence in ["none", "low"]:
                answer = (
                    "I couldn't find specific information about that in the employee handbook. "
                    "Could you rephrase your question or ask about a different topic?"
                )
                return {"success": True, "answer": answer}
            
            # Generate answer
            result = self.qa_chain.run(
                context=context,
                question=question
            )
            
            answer = result.strip()
            
            # Cache response
            if self.response_cache:
                self.response_cache.put(cache_key, {"answer": answer})
            
            return {
                "success": True,
                "answer": answer,
                "cached": False,
                "metadata": metadata  # Include for debugging
            }
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }