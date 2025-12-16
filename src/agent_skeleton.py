"""
ESILV Smart Assistant - Agent Skeleton
========================================
This file contains the blueprint for all 5 core agents.

Date: 2025-12-16
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import json
from enum import Enum


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Document:
    """Represents a chunk from the vector store"""
    content: str
    source: str
    metadata: Dict[str, Any]
    embedding_score: float = 0.0


@dataclass
class UserProfile:
    """Stores collected user information"""
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    program_interest: Optional[str] = None
    education_level: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'first_name': self.first_name,
            'last_name': self.last_name,
            'email': self.email,
            'phone': self.phone,
            'program_interest': self.program_interest,
            'education_level': self.education_level,
        }
    
    def get_missing_fields(self) -> List[str]:
        """Return list of None fields"""
        return [k for k, v in self.to_dict().items() if v is None]


class IntentType(str, Enum):
    """Valid intent types"""
    PROGRAM_INFO = "program_info"
    ADMISSION_HELP = "admission_help"
    COURSE_DETAILS = "course_details"
    CONTACT_COLLECTION = "contact_collection"
    GENERAL_INFO = "general_info"
    SMALL_TALK = "small_talk"
    UNKNOWN = "unknown"


@dataclass
class Intent:
    """Result of intent classification"""
    type: IntentType
    confidence: float
    entities: Dict[str, Any]


@dataclass
class ChatTurn:
    """Single message in conversation"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ChatState:
    """State passed through the LangGraph"""
    user_message: str
    chat_history: List[ChatTurn]
    user_profile: UserProfile = None
    detected_intent: Intent = None
    retrieved_context: List[Document] = None
    generated_response: str = None
    form_data: Dict = None
    
    def __post_init__(self):
        if self.user_profile is None:
            self.user_profile = UserProfile()


# ============================================================================
# BASE AGENT CLASS
# ============================================================================

class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = self._setup_logger()
    
    @abstractmethod
    def execute(self, state: ChatState) -> ChatState:
        """Execute agent logic and update state"""
        pass
    
    def _setup_logger(self):
        """Setup logging for this agent"""
        import logging
        logger = logging.getLogger(f"ESILV.{self.name}")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            f'[{self.name}] %(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger


# ============================================================================
# AGENT 1: ROUTER
# ============================================================================

class RouterAgent(BaseAgent):
    """
    Classifies user intent and decides which agent to invoke next.
    
    INPUTS:
    - state.user_message: Current user query
    - state.chat_history: Previous messages (for context)
    
    OUTPUTS:
    - state.detected_intent: Intent classification with confidence
    """
    
    def __init__(self, llm_client):
        super().__init__("Router")
        self.llm = llm_client
        
        # Intent definitions
        self.INTENT_DEFINITIONS = {
            IntentType.PROGRAM_INFO: {
                'description': 'User wants info on programs/specializations',
                'keywords': ['program', 'specialization', 'track', 'major', 'curriculum'],
            },
            IntentType.ADMISSION_HELP: {
                'description': 'User asks about admission process',
                'keywords': ['admission', 'apply', 'requirement', 'prerequisite', 'deadline'],
            },
            IntentType.COURSE_DETAILS: {
                'description': 'User wants course information',
                'keywords': ['course', 'class', 'subject', 'module', 'semester', 'credits'],
            },
            IntentType.CONTACT_COLLECTION: {
                'description': 'User wants to register/contact school',
                'keywords': ['contact', 'register', 'sign up', 'email', 'phone', 'get in touch'],
            },
            IntentType.GENERAL_INFO: {
                'description': 'General ESILV information',
                'keywords': ['esilv', 'school', 'campus', 'history', 'facilities', 'location'],
            },
            IntentType.SMALL_TALK: {
                'description': 'Greeting, small talk',
                'keywords': ['hello', 'hi', 'how are you', 'thanks', 'bye'],
            },
        }
    
    def execute(self, state: ChatState) -> ChatState:
        """
        Step 1: Classify intent using LLM
        Step 2: Extract entities (program name, course code, etc.)
        Step 3: Assign confidence score
        Step 4: Update state
        """
        self.logger.info(f"Classifying intent for: {state.user_message[:50]}...")
        
        # TODO: Implement intent classification
        # Approach:
        # 1. Build prompt with intent definitions
        # 2. Call LLM (self.llm.chat(...))
        # 3. Parse JSON response
        # 4. Validate and assign confidence
        
        # PLACEHOLDER
        intent = Intent(
            type=IntentType.PROGRAM_INFO,
            confidence=0.95,
            entities={'program_name': None}
        )
        
        state.detected_intent = intent
        self.logger.info(f"Intent: {intent.type} (confidence: {intent.confidence:.2f})")
        return state
    
    def _build_classification_prompt(self, user_message: str) -> str:
        """Build prompt for intent classification"""
        # TODO: Implement prompt building
        pass
    
    def _parse_intent_response(self, response: str) -> Tuple[IntentType, float, Dict]:
        """Parse LLM response into structured intent"""
        # TODO: Implement parsing
        pass


# ============================================================================
# AGENT 2: RETRIEVER (RAG)
# ============================================================================

class RetrieverAgent(BaseAgent):
    """
    Searches the knowledge base for relevant documents using RAG.
    
    INPUTS:
    - state.user_message: User query
    - state.detected_intent: Intent (for filtering/boosting)
    
    OUTPUTS:
    - state.retrieved_context: List of relevant documents
    """
    
    def __init__(self, vector_store, embedding_model, llm_client):
        super().__init__("Retriever")
        self.vector_store = vector_store  # FAISS or Chromadb
        self.embedding_model = embedding_model  # Ollama embeddings
        self.llm = llm_client
        self.top_k = 5
        self.rerank = True
    
    def execute(self, state: ChatState) -> ChatState:
        """
        Step 1: Optionally expand the query
        Step 2: Generate query embedding
        Step 3: Search vector store
        Step 4: Optionally re-rank results
        Step 5: Return top-K documents
        """
        self.logger.info(f"Retrieving context for: {state.user_message[:50]}...")
        
        # TODO: Implement retrieval pipeline
        # 1. call self.augment_query(state.user_message)
        # 2. call self.embed_query(augmented_queries)
        # 3. call self.search_vector_store(embeddings)
        # 4. call self.rerank_results(query, candidates) [optional]
        # 5. return top_k
        
        # PLACEHOLDER
        context = []
        
        state.retrieved_context = context
        self.logger.info(f"Retrieved {len(context)} documents")
        return state
    
    def augment_query(self, query: str) -> List[str]:
        """
        Generate query variations to improve retrieval.
        
        Techniques:
        - Paraphrase the query
        - Add synonyms
        - Break complex queries into sub-queries
        """
        # TODO: Implement query augmentation
        # Use LLM to generate 2-3 variations
        pass
    
    def embed_query(self, queries: List[str]) -> List[List[float]]:
        """Generate embeddings for queries using Ollama"""
        # TODO: Implement embedding
        # Use self.embedding_model to get vectors
        pass
    
    def search_vector_store(self, embeddings: List[List[float]]) -> List[Document]:
        """Search vector store using cosine similarity"""
        # TODO: Implement vector search
        # Use self.vector_store.query(embeddings, top_k=self.top_k)
        pass
    
    def rerank_results(self, query: str, candidates: List[Document]) -> List[Document]:
        """
        Re-rank candidates using cross-encoder or LLM.
        Optional but improves precision.
        """
        # TODO: Implement re-ranking
        # Option 1: Use HuggingFace cross-encoder
        # Option 2: Use LLM to score relevance
        pass


# ============================================================================
# AGENT 3: QA GENERATOR
# ============================================================================

class QAGeneratorAgent(BaseAgent):
    """
    Generates answers grounded in retrieved documents.
    
    INPUTS:
    - state.user_message: User query
    - state.retrieved_context: Relevant documents from retriever
    - state.detected_intent: Intent (for tone/format)
    
    OUTPUTS:
    - state.generated_response: Natural language answer
    """
    
    def __init__(self, llm_client):
        super().__init__("QAGenerator")
        self.llm = llm_client
    
    def execute(self, state: ChatState) -> ChatState:
        """
        Step 1: Build prompt with context
        Step 2: Call LLM to generate answer
        Step 3: Post-process response (citations, formatting)
        Step 4: Return answer
        """
        self.logger.info("Generating answer...")
        
        # TODO: Implement answer generation
        # 1. call self.build_generation_prompt(...)
        # 2. call self.llm.chat(prompt)
        # 3. call self.post_process_response(response)
        # 4. return response
        
        # PLACEHOLDER
        response = "I don't have information about that yet."
        
        state.generated_response = response
        self.logger.info(f"Generated response: {response[:50]}...")
        return state
    
    def build_generation_prompt(self, 
                               query: str,
                               context: List[Document],
                               intent: Intent) -> str:
        """
        Build system + context + query prompt.
        
        Template:
        ---
        You are an ESILV school assistant chatbot.
        
        CONTEXT (from documentation):
        [retrieved documents]
        
        USER QUERY: [query]
        
        ANSWER:
        ---
        """
        # TODO: Implement prompt building
        pass
    
    def post_process_response(self, response: str, context: List[Document]) -> str:
        """
        Clean up and format response.
        - Add citations to sources
        - Format bullet points, lists
        - Fix grammar/typos
        """
        # TODO: Implement post-processing
        pass


# ============================================================================
# AGENT 4: FORM-FILLER
# ============================================================================

class FormFillerAgent(BaseAgent):
    """
    Extracts and validates user contact information.
    
    INPUTS:
    - state.user_message: Current user message
    - state.chat_history: Full conversation (multi-turn dialogue)
    - state.user_profile: Already-collected info
    
    OUTPUTS:
    - state.form_data: Extracted/validated structured data
    - state.generated_response: Follow-up question if needed
    """
    
    def __init__(self, llm_client):
        super().__init__("FormFiller")
        self.llm = llm_client
        
        # Define required fields
        self.REQUIRED_FIELDS = {
            'first_name': str,
            'last_name': str,
            'email': str,
            'phone': str,
            'program_interest': str,
            'education_level': str,
        }
    
    def execute(self, state: ChatState) -> ChatState:
        """
        Step 1: Extract user info from current + past messages
        Step 2: Validate extracted data (email format, etc.)
        Step 3: Identify missing fields
        Step 4: Generate follow-up question if needed
        Step 5: Update state
        """
        self.logger.info("Extracting user information...")
        
        # TODO: Implement form extraction
        # 1. build extraction prompt
        # 2. call llm.chat(prompt) to extract JSON
        # 3. validate each field
        # 4. check for missing fields
        # 5. generate follow-up question if needed
        
        # PLACEHOLDER
        extracted = {}
        state.form_data = extracted
        self.logger.info(f"Extracted fields: {extracted}")
        return state
    
    def extract_info_from_conversation(self, messages: List[ChatTurn]) -> Dict:
        """
        Use LLM to extract structured info from multi-turn dialogue.
        
        Output JSON:
        {
            "first_name": "...",
            "last_name": "...",
            "email": "...",
            "phone": "...",
            "program_interest": "Engineering" | "MBA" | "...",
            "education_level": "Bac" | "Master" | "..."
        }
        """
        # TODO: Implement extraction
        pass
    
    def validate_field(self, field_name: str, value: str) -> Tuple[bool, Optional[str]]:
        """
        Validate individual field.
        Returns: (is_valid, error_message)
        """
        # TODO: Implement field validation
        # - email: regex match
        # - phone: format check (French format?)
        # - program_interest: enum check
        # - education_level: enum check
        pass
    
    def get_next_question(self, missing_fields: List[str]) -> str:
        """
        Generate a natural follow-up question for missing info.
        
        Example:
        missing_fields = ['email', 'phone']
        → "Could you please share your email address so we can follow up?"
        """
        # TODO: Implement question generation
        pass


# ============================================================================
# AGENT 5: CONVERSATION MANAGER
# ============================================================================

class ConversationManagerAgent(BaseAgent):
    """
    Manages conversation state, memory, and user profile.
    
    INPUTS:
    - state.chat_history: All messages so far
    - state.user_profile: Collected user info
    - state.form_data: Newly extracted data
    - state.generated_response: Current response to send
    
    OUTPUTS:
    - Updated state with:
      - chat_history (add new messages)
      - user_profile (merge new data)
      - conversation goal/next step
    """
    
    def __init__(self):
        super().__init__("ConversationManager")
        self.max_history = 10  # Keep last N messages
    
    def execute(self, state: ChatState) -> ChatState:
        """
        Step 1: Add current response to chat history
        Step 2: Merge form_data into user_profile
        Step 3: Infer conversation goal/next step
        Step 4: Format response for output
        Step 5: Return updated state
        """
        self.logger.info("Managing conversation state...")
        
        # TODO: Implement state management
        # 1. append assistant message to chat_history
        # 2. merge form_data into user_profile
        # 3. trim history to max_history
        # 4. infer next step/goal
        
        self.logger.info(f"Conversation updated. Goal: {self._infer_goal(state)}")
        return state
    
    def add_turn(self, state: ChatState, role: str, message: str) -> ChatState:
        """Add a message to chat history"""
        # TODO: Implement
        pass
    
    def merge_form_data(self, profile: UserProfile, form_data: Dict) -> UserProfile:
        """Merge newly extracted data into user profile"""
        # TODO: Implement
        pass
    
    def _infer_goal(self, state: ChatState) -> str:
        """
        Infer high-level conversation goal.
        
        Examples:
        - "User is asking about Engineering programs"
        - "User wants to register"
        - "Small talk / greeting"
        """
        # TODO: Implement goal inference
        pass
    
    def trim_history(self, history: List[ChatTurn], max_len: int) -> List[ChatTurn]:
        """Keep only last N messages"""
        # TODO: Implement
        pass


# ============================================================================
# ORCHESTRATION: LANGGRAPH SETUP
# ============================================================================

class ESILVChatGraph:
    """
    Main orchestration graph using LangGraph.
    
    Flow:
    START → ROUTER → [branch based on intent]
                    ├─→ RETRIEVER → QA_GENERATOR → CONVERSATION_MANAGER → END
                    └─→ FORM_FILLER → CONVERSATION_MANAGER → END
    """
    
    def __init__(self, 
                 router: RouterAgent,
                 retriever: RetrieverAgent,
                 qa_generator: QAGeneratorAgent,
                 form_filler: FormFillerAgent,
                 conv_manager: ConversationManagerAgent):
        
        self.router = router
        self.retriever = retriever
        self.qa_generator = qa_generator
        self.form_filler = form_filler
        self.conv_manager = conv_manager
        
        # TODO: Import and build LangGraph
        # from langgraph.graph import StateGraph, END
        # self.graph = StateGraph(ChatState)
        # ... add nodes and edges
    
    def should_collect_info(self, state: ChatState) -> bool:
        """
        Conditional: should we collect user info or answer question?
        
        True if:
        - intent == CONTACT_COLLECTION
        - intent == UNKNOWN (collect info to clarify)
        """
        # TODO: Implement condition
        return False
    
    def run(self, user_message: str, chat_history: List[ChatTurn]) -> str:
        """
        Execute the conversation graph.
        
        Args:
            user_message: New message from user
            chat_history: Previous conversation
        
        Returns:
            Response to send to user
        """
        # TODO: Implement graph execution
        # 1. Create ChatState
        # 2. Invoke compiled_graph.invoke(state)
        # 3. Return state.generated_response
        pass


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

class ESILVSmartAssistant:
    """
    Main chatbot class that ties everything together.
    """
    
    def __init__(self, 
                 vector_store_path: str = "./vectorstore",
                 ollama_host: str = "http://localhost:11434"):
        
        self.ollama_host = ollama_host
        self.vector_store_path = vector_store_path
        
        # TODO: Initialize all components
        # 1. Setup Ollama LLM client
        # 2. Setup Ollama embedding model
        # 3. Load or create vector store
        # 4. Initialize all agents
        # 5. Build LangGraph
        pass
    
    def chat(self, user_message: str) -> Dict[str, Any]:
        """
        Main chat function.
        
        Args:
            user_message: User's input
        
        Returns:
            {
                'response': str,
                'intent': IntentType,
                'user_profile': Dict,
                'confidence': float
            }
        """
        # TODO: Implement
        pass


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example:
    
    # Initialize chatbot
    assistant = ESILVSmartAssistant()
    
    # Single turn
    result = assistant.chat("What programs do you offer?")
    print(result['response'])
    
    # Multi-turn conversation
    chat_history = []
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        
        result = assistant.chat(user_input)
        chat_history.append(ChatTurn("user", user_input))
        chat_history.append(ChatTurn("assistant", result['response']))
        
        print(f"Assistant: {result['response']}")
        print(f"Intent: {result['intent']}")
    """
    pass
