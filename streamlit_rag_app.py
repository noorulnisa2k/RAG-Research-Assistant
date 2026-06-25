import os
import re
import json
import tempfile
import textwrap
from pathlib import Path
from typing import Optional, List
import streamlit as st
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
import autogen

# Load environment variables
load_dotenv()

# ============================================================
# Streamlit Configuration
# ============================================================
st.set_page_config(
    page_title="RAG Research Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔍 Multi-Agent RAG Research Assistant")
st.markdown("Upload documents and ask research questions with intelligent retrieval and synthesis.")

# ============================================================
# Session State Initialization
# ============================================================
if 'chroma_client' not in st.session_state:
    st.session_state.chroma_client = None
if 'collection' not in st.session_state:
    st.session_state.collection = None
if 'all_chunks' not in st.session_state:
    st.session_state.all_chunks = []
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

# ============================================================
# Configuration
# ============================================================
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
COLLECTION_NAME = "academic_knowledge_base"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
TOP_K = 5
PERSIST_DIR = "./chroma_db"

# LLM Configuration
llm_config = {
    "config_list": [
        {
            'model': os.getenv("LLAMA_MODEL", "Llama-3.2-3B-Instruct"),
            'base_url': os.getenv("BASE_URL", "http://localhost:1234/v1"),
            'api_key': os.getenv("API_KEY", "lm-studio"),
        }
    ],
    "temperature": 0.3,
    "timeout": 120,
}

# ============================================================
# Document Processing Functions
# ============================================================
def load_pdf(path: str) -> str:
    """Extract text from a PDF file."""
    try:
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        st.error(f"Error reading PDF {path}: {e}")
        return ""

def load_txt(path: str) -> str:
    """Read a plain text file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        st.error(f"Error reading TXT {path}: {e}")
        return ""

def chunk_text(text: str, source: str,
               chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> List[dict]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if len(chunk) > 50:  # skip very short chunks
            chunks.append({
                "text": chunk,
                "source": source,
                "chunk_id": f"{source}_chunk_{idx}"
            })
            idx += 1
        start += chunk_size - overlap
    return chunks

def initialize_chromadb():
    """Initialize ChromaDB with persistence."""
    if not Path(PERSIST_DIR).exists():
        Path(PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    
    st.session_state.chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
    
    # Use sentence-transformers embedding function
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    
    # Delete existing collection to avoid conflicts
    try:
        st.session_state.chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    
    st.session_state.collection = st.session_state.chroma_client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )
    
    return st.session_state.collection

def index_chunks(chunks: List[dict]):
    """Index chunks into ChromaDB."""
    if not st.session_state.collection:
        st.error("ChromaDB not initialized!")
        return
    
    BATCH_SIZE = 50
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        st.session_state.collection.add(
            ids=[c["chunk_id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[{"source": c["source"]} for c in batch],
        )

def retrieve_chunks(query: str, top_k: int = TOP_K) -> List[dict]:
    """Retrieve relevant chunks for a query."""
    if not st.session_state.collection:
        return []
    
    results = st.session_state.collection.query(
        query_texts=[query],
        n_results=top_k,
    )
    
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "source": meta["source"],
            "distance": round(dist, 4),
        })
    return chunks

# ============================================================
# Tool Definitions
# ============================================================
def format_citation(text: str, source: str) -> str:
    """Format a text snippet with its source as an APA-style citation."""
    name = source.replace(".pdf", "").replace(".txt", "")
    parts = name.split("_")
    year = next((p for p in parts if re.match(r"^\d{4}$", p)), None)
    author = parts[0] if parts else name
    key = f"{author} et al., {year}" if year else name
    return f"{text.strip()} ({key})"

def summarize_text(text: str, max_words: int = 100) -> str:
    """Truncate or summarize text to maximum words."""
    words = text.split()
    if len(words) <= max_words:
        return text
    truncated = " ".join(words[:max_words])
    last_period = max(truncated.rfind("."), truncated.rfind("!"), truncated.rfind("?"))
    if last_period > len(truncated) // 2:
        return truncated[: last_period + 1]
    return truncated + "..."

def rag_retrieve(query: str, top_k: int = 5) -> str:
    """Retrieve relevant chunks from the knowledge base."""
    chunks = retrieve_chunks(query, top_k=top_k)
    if not chunks:
        return "No relevant documents found in the knowledge base."
    output = f"=== Retrieved {len(chunks)} chunks for query: '{query}' ===\n\n"
    for i, chunk in enumerate(chunks, 1):
        output += f"[Chunk {i}] Source: {chunk['source']}\n"
        output += f"{chunk['text']}\n"
        output += "-" * 60 + "\n"
    return output

# ============================================================
# Sidebar - Configuration & Upload
# ============================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Initialize Database
    if st.button("🔄 Initialize/Reset Database", use_container_width=True):
        with st.spinner("Initializing ChromaDB..."):
            initialize_chromadb()
            st.session_state.all_chunks = []
            st.session_state.initialized = True
            st.success("✅ Database initialized!")
    
    st.divider()
    
    # Document Upload
    st.header("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("Process Documents", use_container_width=True):
            with st.spinner("Processing documents..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                temp_dir = tempfile.mkdtemp()
                new_chunks = []
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}")
                    
                    # Save temp file
                    temp_path = Path(temp_dir) / uploaded_file.name
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Load and chunk
                    if uploaded_file.name.endswith(".pdf"):
                        text = load_pdf(str(temp_path))
                    else:
                        text = load_txt(str(temp_path))
                    
                    if text.strip():
                        chunks = chunk_text(text, uploaded_file.name)
                        new_chunks.extend(chunks)
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                # Initialize if not done
                if not st.session_state.initialized:
                    initialize_chromadb()
                    st.session_state.initialized = True
                
                # Index chunks
                if new_chunks:
                    index_chunks(new_chunks)
                    st.session_state.all_chunks.extend(new_chunks)
                    st.session_state.uploaded_files.extend([f.name for f in uploaded_files])
                    st.success(f"✅ Processed {len(new_chunks)} chunks from {len(uploaded_files)} file(s)!")
                else:
                    st.warning("⚠️ No valid text extracted from files.")
    
    # Show indexed files
    if st.session_state.uploaded_files:
        st.divider()
        st.subheader("📚 Indexed Files")
        for file in st.session_state.uploaded_files:
            st.caption(f"✓ {file}")
        st.caption(f"Total chunks: {len(st.session_state.all_chunks)}")
    
    st.divider()
    st.subheader("Settings")
    top_k = st.slider("Top K Results", min_value=1, max_value=10, value=5)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3)
    
    # Update LLM config
    llm_config["temperature"] = temperature

# ============================================================
# Main Content Area
# ============================================================
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Research Query")
    query = st.text_area(
        "Enter your research question:",
        height=120,
        placeholder="E.g., What is Automated Essay Scoring (AES) and how do large language models improve upon traditional machine learning approaches?"
    )

with col2:
    st.header("📊 Status")
    if st.session_state.initialized and st.session_state.all_chunks:
        st.success("✅ Ready")
        st.metric("Documents Indexed", len(st.session_state.uploaded_files))
        st.metric("Chunks Created", len(st.session_state.all_chunks))
    else:
        st.warning("⚠️ No data indexed")
        st.info("Upload documents and click 'Process Documents' to begin.")

# ============================================================
# Query Execution
# ============================================================
if st.button("🚀 Ask Research Assistant", use_container_width=True, type="primary"):
    if not query.strip():
        st.error("Please enter a query!")
    elif not st.session_state.initialized or not st.session_state.all_chunks:
        st.error("Please upload and process documents first!")
    else:
        st.divider()
        st.header("🤖 Research Assistant Response")
        
        # Create placeholder for streaming response
        response_placeholder = st.empty()
        messages_placeholder = st.empty()
        
        with st.spinner("🔍 Retrieving context..."):
            # Quick retrieval to show sources
            initial_chunks = retrieve_chunks(query, top_k=min(top_k, 3))
        
        if initial_chunks:
            # Show retrieved sources
            with st.expander("📖 Retrieved Sources", expanded=True):
                for i, chunk in enumerate(initial_chunks, 1):
                    st.markdown(f"**Source {i}: {chunk['source']}** (Distance: {chunk['distance']})")
                    st.markdown(f"_{chunk['text'][:300]}..._")
                    st.divider()
        
        # Multi-agent pipeline simulation
        with st.spinner("⚙️ Running multi-agent pipeline..."):
            try:
                # Create agents
                planner_agent = autogen.AssistantAgent(
                    name="PlannerAgent",
                    llm_config=llm_config,
                    system_message="""You are an expert academic research planner.
Your role is to DECOMPOSE complex user queries into clear sub-tasks and coordinate a multi-agent pipeline.
When given a query, identify key concepts and specify sub-queries for retrieval.
Format your response as:
=== RESEARCH PLAN ===
Query: <original question>
Sub-queries for retrieval:
  1. <sub-query 1>
  2. <sub-query 2>
====================
After producing the plan, pass control to RetrieverAgent."""
                )
                
                retriever_agent = autogen.AssistantAgent(
                    name="RetrieverAgent",
                    llm_config=llm_config,
                    system_message="""You are a precise academic retrieval agent.
Your role is to fetch relevant information from the knowledge base using the 'rag_retrieve' tool.
When given sub-queries, call rag_retrieve() for EACH sub-query and consolidate the context."""
                )
                
                answer_agent = autogen.AssistantAgent(
                    name="AnswerAgent",
                    llm_config=llm_config,
                    system_message="""You are an expert academic writer and synthesizer.
Your role is to produce a WELL-STRUCTURED, GROUNDED answer using the original query and retrieved context.
Your answer MUST:
1. Be structured with clear sections and headings
2. Include inline citations for every factual claim
3. Contain NO hallucinations — only use information from retrieved context
4. End with a References section listing all cited sources
End your response with TERMINATE to signal completion."""
                )
                
                user_proxy = autogen.UserProxyAgent(
                    name="UserProxy",
                    human_input_mode="NEVER",
                    max_consecutive_auto_reply=10,
                    is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", "").upper(),
                    code_execution_config=False,
                    default_auto_reply="Continue with the next step.",
                )
                
                # Register tools
                autogen.register_function(
                    rag_retrieve,
                    caller=retriever_agent,
                    executor=user_proxy,
                    name="rag_retrieve",
                    description="Retrieve relevant text chunks from the knowledge base."
                )
                
                autogen.register_function(
                    format_citation,
                    caller=answer_agent,
                    executor=user_proxy,
                    name="format_citation",
                    description="Format a text claim with an inline APA citation."
                )
                
                autogen.register_function(
                    summarize_text,
                    caller=answer_agent,
                    executor=user_proxy,
                    name="summarize_text",
                    description="Truncate text to a maximum word count."
                )
                
                # Create group chat
                group_chat = autogen.GroupChat(
                    agents=[user_proxy, planner_agent, retriever_agent, answer_agent],
                    messages=[],
                    max_round=15,
                    speaker_selection_method="auto",
                    allow_repeat_speaker=False,
                )
                
                group_chat_manager = autogen.GroupChatManager(
                    groupchat=group_chat,
                    llm_config=llm_config,
                )
                
                # Run the pipeline
                chat_result = user_proxy.initiate_chat(
                    group_chat_manager,
                    message=query,
                    clear_history=True,
                )
                
                # Extract and display response
                final_response = ""
                if chat_result.chat_history:
                    for msg in chat_result.chat_history:
                        if msg.get("name") == "AnswerAgent":
                            final_response = msg.get("content", "")
                            break
                
                if final_response:
                    # Clean response
                    final_response = final_response.replace("TERMINATE", "").strip()
                    response_placeholder.markdown(final_response)
                else:
                    response_placeholder.warning("No structured response generated.")
                
                st.success("✅ Response generated successfully!")
                
            except Exception as e:
                st.error(f"Error during pipeline execution: {e}")
                st.info("Make sure your local LLM server is running!")

# ============================================================
# Footer
# ============================================================
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("🔬 RAG-Based Research Assistant")
with col2:
    st.caption("Powered by ChromaDB + AutoGen + Local LLM")
with col3:
    st.caption("Upload • Retrieve • Synthesize")
