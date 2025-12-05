"""FastAPI server for Finance Agent.

This module provides a REST API for the Finance Assistant using FastAPI.
Designed for deployment to Google Cloud Run.

Endpoints:
- GET /health - Health check
- POST /chat - Chat with the agent (returns full response)
- POST /chat/stream - Chat with streaming response (SSE)
- GET /agent/info - Get agent configuration info

Usage:
    # Local development
    uvicorn finance_agent.main:app --reload --port 8080
    
    # Or with Python
    python -m finance_agent.main
"""

import json
import os
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Optional

from dotenv import load_dotenv
from fastapi import Cookie, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel, Field

# Setup paths
agent_dir = Path(__file__).parent
project_root = agent_dir.parent
sys.path.insert(0, str(agent_dir))
sys.path.insert(0, str(project_root))

# Load environment variables
env_paths = [agent_dir / '.env', project_root / '.env']
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
        break

# Import agent components
try:
    from workflow_1.agents.root_agent import create_root_agent
    from workflow_1.arize_observability.observability import setup_arize_tracing
except ImportError:
    from src.workflow_1.agents.root_agent import create_root_agent
    from src.workflow_1.arize_observability.observability import setup_arize_tracing

# Initialize Arize tracing BEFORE creating agents (for auto-instrumentation)
# Set ARIZE_ENABLED=false in .env to disable tracing
if os.getenv("ARIZE_ENABLED", "true").lower() != "false":
    setup_arize_tracing()


# =============================================================================
# Pydantic Models
# =============================================================================

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., description="User's message to the agent")
    session_id: str = Field(default="default_session", description="Session identifier")
    model: Optional[str] = Field(default=None, description="Model override")
    # Note: user_id is now handled via cookies (finance_user_token)


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str = Field(..., description="Agent's response")
    user_id: str = Field(..., description="User identifier")
    session_id: str = Field(..., description="Session identifier")
    timestamp: str = Field(..., description="Response timestamp")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(default="healthy")
    service: str = Field(default="finance-agent")
    timestamp: str
    version: str = Field(default="1.0.0")


class AgentInfoResponse(BaseModel):
    """Response model for agent info."""
    name: str
    model: str
    sub_agents: list[str]
    description: str


class UserInfoResponse(BaseModel):
    """Response model for user info."""
    user_id: str
    is_new_user: bool


class MemoryItem(BaseModel):
    """Response model for a single memory."""
    id: str
    user_id: str
    session_id: Optional[str]
    summary: str
    created_at: str
    message_count: Optional[int] = None


class MemoriesResponse(BaseModel):
    """Response model for memories list."""
    memories: list[MemoryItem]
    total: int
    user_id: str


# =============================================================================
# Cookie Configuration
# =============================================================================

USER_COOKIE_NAME = "finance_user_token"
USER_COOKIE_MAX_AGE = 86400 * 365  # 1 year in seconds


def get_or_create_user_id(user_token: Optional[str], response: Response) -> str:
    """Get user ID from cookie or create a new one.
    
    If no cookie exists, generates a UUID and sets it as a persistent cookie.
    Returns the user_id to use for this request.
    """
    if user_token:
        return user_token
    
    # Generate new user token
    new_token = str(uuid.uuid4())
    response.set_cookie(
        key=USER_COOKIE_NAME,
        value=new_token,
        max_age=USER_COOKIE_MAX_AGE,
        httponly=True,  # Not accessible via JavaScript for security
        samesite="lax",  # Allow cross-site requests for GET
        secure=False,  # Set to True in production with HTTPS
    )
    return new_token


# =============================================================================
# Global State
# =============================================================================

# Cache for agent and runners per session
_agent = None
_session_service = None
_memory_service = None
_runners = {}  # Cache runners per session


def get_agent(memory_service=None):
    """Get or create the root agent with dynamic model routing.
    
    The agent uses before_model_callback to dynamically select
    the appropriate model (fast vs complex) for each query.
    
    If memory_service is provided, the agent will have:
    - load_memory tool for cross-session recall
    - after_agent_callback for auto-saving to memory
    """
    global _agent
    if _agent is None:
        # Create agent with dynamic routing enabled and optional memory
        _agent = create_root_agent(use_dynamic_routing=True, memory_service=memory_service)
    return _agent


async def get_session_service():
    """Get or create session service.
    
    Uses InMemorySessionService for ephemeral sessions.
    Session persistence is NOT needed - Arize handles logging/monitoring.
    Cross-session recall is handled by PostgresMemoryService instead.
    """
    global _session_service
    if _session_service is None:
        from google.adk.sessions import InMemorySessionService
        _session_service = InMemorySessionService()
        print("📦 Using InMemorySessionService (ephemeral sessions)")
    return _session_service


async def get_memory_service():
    """Get or create memory service for cross-session recall.
    
    Automatically selects the appropriate memory service:
    - Cloud Run deployment → PostgresMemoryService (persistent)
    - Local development → InMemoryMemoryService (fast, no setup)
    
    Override with MEMORY_STORAGE environment variable:
    - "postgres": PostgresMemoryService (requires SESSION_DB_URL)
    - "memory": InMemoryMemoryService (no persistence)
    - "auto" (default): Postgres on Cloud Run, Memory locally
    """
    global _memory_service
    if _memory_service is None:
        storage_type = os.getenv("MEMORY_STORAGE", "auto").lower()
        
        # Auto-detect: use Postgres on Cloud Run, Memory locally
        if storage_type == "auto":
            # K_SERVICE is set by Cloud Run
            is_cloud_run = os.getenv("K_SERVICE") is not None
            storage_type = "postgres" if is_cloud_run else "memory"
        
        if storage_type == "postgres":
            from workflow_1.memory import PostgresMemoryService
            _memory_service = PostgresMemoryService()
            print("🧠 Using PostgresMemoryService (persistent cross-session recall)")
        else:
            from google.adk.memory import InMemoryMemoryService
            _memory_service = InMemoryMemoryService()
            print("🧠 Using InMemoryMemoryService (no persistence)")
    return _memory_service


async def get_runner(user_id: str, session_id: str):
    """Get or create a runner for the given session."""
    from google.adk.runners import Runner
    
    cache_key = f"{user_id}:{session_id}"
    
    if cache_key not in _runners:
        # Get services first (memory service needed for agent creation)
        memory_service = await get_memory_service()
        session_service = await get_session_service()
        agent = get_agent(memory_service=memory_service)
        
        # Create session if it doesn't exist
        try:
            await session_service.create_session(
                app_name=agent.name,
                user_id=user_id,
                session_id=session_id,
            )
        except Exception:
            # Session might already exist
            pass
        
        _runners[cache_key] = Runner(
            agent=agent,
            app_name=agent.name,
            session_service=session_service,
            memory_service=memory_service,
        )
    
    return _runners[cache_key]


# =============================================================================
# FastAPI App
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup: Initialize the agent WITH memory service
    print("🚀 Starting Finance Agent API...")
    memory_service = await get_memory_service()
    agent = get_agent(memory_service=memory_service)
    print(f"✓ Agent initialized: {agent.name}")
    print(f"✓ Root model (fast routing): {agent.model}")
    if agent.sub_agents:
        for sa in agent.sub_agents:
            print(f"✓ Sub-agent '{sa.name}' model (complex analysis): {sa.model}")
    yield
    # Shutdown: Cleanup
    print("👋 Shutting down Finance Agent API...")


app = FastAPI(
    title="Finance Agent API",
    description="S&P 500 Financial Intelligence Chatbot powered by Google ADK",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint for Cloud Run."""
    return HealthResponse(
        status="healthy",
        service="finance-agent",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0",
    )


@app.get("/", tags=["System"])
async def root():
    """Root endpoint with API info."""
    return {
        "service": "Finance Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "chat_ui": "/chat-ui",
        "health": "/health",
        "endpoints": {
            "chat": "POST /chat",
            "chat_stream": "POST /chat/stream",
            "agent_info": "GET /agent/info",
        }
    }


CHAT_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FinanceBot - S&P 500 Assistant</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .chat-container {
            width: 100%;
            max-width: 800px;
            background: #fff;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .chat-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }
        .chat-header h1 { font-size: 1.5rem; margin-bottom: 5px; }
        .chat-header p { opacity: 0.9; font-size: 0.9rem; }
        .chat-messages {
            height: 400px;
            overflow-y: auto;
            padding: 20px;
            background: #f8f9fa;
        }
        .message {
            margin-bottom: 16px;
            display: flex;
            flex-direction: column;
        }
        .message.user { align-items: flex-end; }
        .message.bot { align-items: flex-start; }
        .message-content {
            max-width: 80%;
            padding: 12px 16px;
            border-radius: 16px;
            line-height: 1.5;
        }
        .message.user .message-content {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-bottom-right-radius: 4px;
        }
        .message.bot .message-content {
            background: white;
            color: #333;
            border-bottom-left-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .message-label {
            font-size: 0.75rem;
            color: #666;
            margin-bottom: 4px;
        }
        .chat-input-container {
            padding: 20px;
            background: white;
            border-top: 1px solid #eee;
            display: flex;
            gap: 12px;
        }
        #messageInput {
            flex: 1;
            padding: 14px 18px;
            border: 2px solid #e0e0e0;
            border-radius: 25px;
            font-size: 1rem;
            outline: none;
            transition: border-color 0.2s;
        }
        #messageInput:focus { border-color: #667eea; }
        #sendBtn {
            padding: 14px 28px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            font-size: 1rem;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        #sendBtn:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4); }
        #sendBtn:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
        .typing { color: #666; font-style: italic; }
        .examples {
            padding: 15px 20px;
            background: #f0f0f0;
            border-top: 1px solid #e0e0e0;
        }
        .examples p { font-size: 0.8rem; color: #666; margin-bottom: 8px; }
        .example-btn {
            background: white;
            border: 1px solid #ddd;
            padding: 6px 12px;
            border-radius: 15px;
            font-size: 0.8rem;
            cursor: pointer;
            margin: 2px;
        }
        .example-btn:hover { background: #667eea; color: white; border-color: #667eea; }
        .memories-toggle {
            position: absolute;
            top: 15px;
            right: 15px;
            background: rgba(255,255,255,0.2);
            border: none;
            color: white;
            padding: 8px 12px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.85rem;
        }
        .memories-toggle:hover { background: rgba(255,255,255,0.3); }
        .session-controls {
            position: absolute;
            top: 15px;
            left: 15px;
            display: flex;
            gap: 8px;
            align-items: center;
        }
        .session-info {
            font-size: 0.7rem;
            color: rgba(255,255,255,0.7);
            max-width: 120px;
            overflow: hidden;
        }
        .new-session-btn {
            background: rgba(255,255,255,0.2);
            border: none;
            color: white;
            padding: 8px 12px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.85rem;
        }
        .new-session-btn:hover { background: rgba(255,255,255,0.3); }
        .new-session-btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .memories-panel {
            display: none;
            background: white;
            border-top: 1px solid #eee;
            max-height: 300px;
            overflow-y: auto;
        }
        .memories-header {
            padding: 12px 20px;
            background: #f8f9fa;
            border-bottom: 1px solid #eee;
            font-weight: 600;
            color: #333;
        }
        .memory-item {
            padding: 12px 20px;
            border-bottom: 1px solid #f0f0f0;
        }
        .memory-item:last-child { border-bottom: none; }
        .memory-time { font-size: 0.75rem; color: #999; margin-bottom: 4px; }
        .memory-summary { font-size: 0.9rem; color: #333; line-height: 1.4; }
        .memory-meta { font-size: 0.75rem; color: #667eea; margin-top: 4px; }
        .memory-loading, .memory-empty, .memory-error { 
            padding: 20px; 
            text-align: center; 
            color: #666; 
        }
        .memory-error { color: #dc3545; }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <div class="session-controls">
                <button class="new-session-btn" id="newSessionBtn" onclick="newSession()">🔄 New Session</button>
                <span class="session-info">ID: <span id="sessionId"></span></span>
            </div>
            <button class="memories-toggle" onclick="toggleMemories()">🧠 Memories</button>
            <h1>📈 FinanceBot</h1>
            <p>S&P 500 Market Intelligence Assistant</p>
        </div>
        <div class="memories-panel" id="memoriesPanel">
            <div class="memories-header">🧠 Long-Term Memories</div>
            <div id="memoriesList"></div>
        </div>
        <div class="chat-messages" id="chatMessages">
            <div class="message bot">
                <span class="message-label">FinanceBot</span>
                <div class="message-content">
                    Hello! I'm your S&P 500 financial assistant. Ask me about stock prices, company info, news, and more!
                </div>
            </div>
        </div>
        <div class="examples">
            <p>Try asking:</p>
            <button class="example-btn" onclick="askExample(this)">What's Apple's stock price?</button>
            <button class="example-btn" onclick="askExample(this)">Tell me about Microsoft</button>
            <button class="example-btn" onclick="askExample(this)">Get Tesla news</button>
            <button class="example-btn" onclick="askExample(this)">Compare AAPL and GOOGL</button>
        </div>
        <div class="chat-input-container">
            <input type="text" id="messageInput" placeholder="Ask about S&P 500 stocks..." onkeypress="handleKeyPress(event)">
            <button id="sendBtn" onclick="sendMessage()">Send</button>
        </div>
    </div>
    <script>
        const chatMessages = document.getElementById('chatMessages');
        const messageInput = document.getElementById('messageInput');
        const sendBtn = document.getElementById('sendBtn');
        
        // Generate unique session ID (user_id is now handled via httpOnly cookie)
        let sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        let userId = null;  // Will be fetched from server
        
        // Initialize user on page load
        async function initUser() {
            try {
                const response = await fetch('/user/info', { credentials: 'include' });
                const data = await response.json();
                userId = data.user_id;
                console.log('User ID:', userId, data.is_new_user ? '(new)' : '(existing)');
            } catch (error) {
                console.error('Failed to get user info:', error);
            }
        }
        initUser();
        
        // Update session display
        function updateSessionDisplay() {
            document.getElementById('sessionId').textContent = sessionId.substring(0, 20) + '...';
        }
        updateSessionDisplay();
        
        function addMessage(content, isUser) {
            const div = document.createElement('div');
            div.className = `message ${isUser ? 'user' : 'bot'}`;
            div.innerHTML = `
                <span class="message-label">${isUser ? 'You' : 'FinanceBot'}</span>
                <div class="message-content">${content}</div>
            `;
            chatMessages.appendChild(div);
            chatMessages.scrollTop = chatMessages.scrollHeight;
            return div;
        }
        
        async function sendMessage() {
            const message = messageInput.value.trim();
            if (!message) return;
            
            addMessage(message, true);
            messageInput.value = '';
            sendBtn.disabled = true;
            
            const typingDiv = addMessage('<span class="typing">Thinking...</span>', false);
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    credentials: 'include',  // Include cookies for user identification
                    body: JSON.stringify({ 
                        message,
                        session_id: sessionId
                    })
                });
                const data = await response.json();
                typingDiv.remove();
                addMessage(data.response || data.detail || 'Sorry, something went wrong.', false);
            } catch (error) {
                typingDiv.remove();
                addMessage('Error: Could not connect to the server.', false);
            }
            
            sendBtn.disabled = false;
            messageInput.focus();
        }
        
        function handleKeyPress(e) {
            if (e.key === 'Enter') sendMessage();
        }
        
        function askExample(btn) {
            messageInput.value = btn.textContent;
            sendMessage();
        }
        
        // Start new session (memories are auto-saved by after_agent_callback)
        function newSession() {
            // Generate new session ID
            sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            updateSessionDisplay();
            
            // Clear chat and show welcome
            chatMessages.innerHTML = `
                <div class="message bot">
                    <span class="message-label">FinanceBot</span>
                    <div class="message-content">
                        New session started! Memories from previous conversations are saved automatically. Ask me about S&P 500 stocks!
                    </div>
                </div>
            `;
            
            // Refresh memories panel if open
            const panel = document.getElementById('memoriesPanel');
            if (panel.style.display !== 'none') {
                loadMemories();
            }
        }
        
        // Memories panel functionality
        async function toggleMemories() {
            const panel = document.getElementById('memoriesPanel');
            const isVisible = panel.style.display !== 'none';
            
            if (isVisible) {
                panel.style.display = 'none';
            } else {
                panel.style.display = 'block';
                await loadMemories();
            }
        }
        
        async function loadMemories() {
            const list = document.getElementById('memoriesList');
            list.innerHTML = '<div class="memory-loading">Loading memories...</div>';
            
            try {
                const response = await fetch('/memories', { credentials: 'include' });
                const data = await response.json();
                
                if (data.memories.length === 0) {
                    list.innerHTML = '<div class="memory-empty">No memories yet. Chat with the bot - memories are saved automatically!</div>';
                    return;
                }
                
                list.innerHTML = data.memories.map(m => `
                    <div class="memory-item">
                        <div class="memory-time">${new Date(m.created_at).toLocaleString()}</div>
                        <div class="memory-summary">${escapeHtml(m.summary)}</div>
                        <div class="memory-meta">${m.message_count || '?'} messages · Session: ${m.session_id ? m.session_id.substring(0, 15) + '...' : 'N/A'}</div>
                    </div>
                `).join('');
            } catch (error) {
                list.innerHTML = '<div class="memory-error">Error loading memories</div>';
            }
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
    </script>
</body>
</html>
"""

@app.get("/chat-ui", response_class=HTMLResponse, tags=["Chat"])
async def chat_ui():
    """Web-based chat interface."""
    return CHAT_HTML


@app.get("/agent/info", response_model=AgentInfoResponse, tags=["Agent"])
async def agent_info():
    """Get information about the agent configuration."""
    agent = get_agent()
    return AgentInfoResponse(
        name=agent.name,
        model=agent.model,
        sub_agents=[sa.name for sa in agent.sub_agents],
        description=agent.description or "Finance Assistant for S&P 500 analysis",
    )


@app.get("/user/info", response_model=UserInfoResponse, tags=["User"])
async def user_info(
    response: Response,
    finance_user_token: Optional[str] = Cookie(None),
):
    """
    Get or create user identification.
    
    Returns the user_id from the cookie, or creates a new one if none exists.
    This endpoint is useful for the JavaScript UI to get the current user_id.
    """
    is_new = finance_user_token is None
    user_id = get_or_create_user_id(finance_user_token, response)
    return UserInfoResponse(user_id=user_id, is_new_user=is_new)


@app.get("/memories", response_model=MemoriesResponse, tags=["Memory"])
async def list_memories(
    response: Response,
    finance_user_token: Optional[str] = Cookie(None),
    limit: int = 20,
):
    """
    List stored memories for the current user (identified via cookie).
    
    Memories are saved after each conversation and can be searched
    by the agent for cross-session recall.
    """
    import psycopg2
    from psycopg2.extras import RealDictCursor
    
    # Get user_id from cookie
    user_id = get_or_create_user_id(finance_user_token, response)
    
    db_url = os.getenv("SESSION_DB_URL")
    if not db_url:
        # Return empty if no database configured (local dev without postgres)
        return MemoriesResponse(memories=[], total=0, user_id=user_id)
    
    try:
        with psycopg2.connect(db_url) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, user_id, session_id, summary, created_at, metadata
                    FROM memories
                    WHERE user_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (user_id, limit))
                rows = cur.fetchall()
                
                cur.execute("SELECT COUNT(*) FROM memories WHERE user_id = %s", (user_id,))
                total = cur.fetchone()["count"]
        
        memories = []
        for row in rows:
            metadata = row.get("metadata") or {}
            memories.append(MemoryItem(
                id=str(row["id"]),
                user_id=row["user_id"],
                session_id=row["session_id"],
                summary=row["summary"] or "",
                created_at=str(row["created_at"]),
                message_count=metadata.get("message_count"),
            ))
        
        return MemoriesResponse(memories=memories, total=total, user_id=user_id)
        
    except Exception as e:
        print(f"⚠️ Error listing memories: {e}")
        return MemoriesResponse(memories=[], total=0, user_id=user_id)


class EndSessionRequest(BaseModel):
    """Request model for ending a session."""
    session_id: str = Field(...)


class EndSessionResponse(BaseModel):
    """Response model for ending a session."""
    success: bool
    message: str
    session_id: str


@app.post("/session/end", response_model=EndSessionResponse, tags=["Session"])
async def end_session(
    request: EndSessionRequest,
    response: Response,
    finance_user_token: Optional[str] = Cookie(None),
):
    """
    End the current session and save it to long-term memory.
    
    This extracts the conversation history and stores it in PostgreSQL
    for cross-session recall by the agent.
    """
    # Get user_id from cookie
    user_id = get_or_create_user_id(finance_user_token, response)
    cache_key = f"{user_id}:{request.session_id}"
    
    # Get the runner if it exists
    if cache_key not in _runners:
        return EndSessionResponse(
            success=False,
            message="No active session found",
            session_id=request.session_id,
        )
    
    try:
        runner = _runners[cache_key]
        session_service = await get_session_service()
        memory_service = await get_memory_service()
        
        # Get the session
        session = await session_service.get_session(
            app_name=runner.app_name,
            user_id=user_id,
            session_id=request.session_id,
        )
        
        if session:
            # Save to memory
            await memory_service.add_session_to_memory(session)
            
            # Remove from cache to force new runner on next request
            del _runners[cache_key]
            
            return EndSessionResponse(
                success=True,
                message="Session saved to memory",
                session_id=request.session_id,
            )
        else:
            return EndSessionResponse(
                success=False,
                message="Session not found",
                session_id=request.session_id,
            )
            
    except Exception as e:
        print(f"⚠️ Error ending session: {e}")
        return EndSessionResponse(
            success=False,
            message=f"Error: {str(e)}",
            session_id=request.session_id,
        )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(
    request: ChatRequest,
    response: Response,
    finance_user_token: Optional[str] = Cookie(None),
):
    """
    Send a message to the finance agent and get a response.
    
    User identification is handled via persistent cookies:
    - First request: A unique user_id is generated and stored in a cookie
    - Subsequent requests: The cookie is read to identify the user
    - Memories are scoped to each user's cookie token
    
    Dynamic model routing is handled automatically by the agent's before_model_callback.
    The callback analyzes each query and selects the appropriate model (fast vs complex).
    
    For streaming responses, use /chat/stream instead.
    """
    from google.genai import types
    
    try:
        # Get user_id from cookie (or create new one)
        user_id = get_or_create_user_id(finance_user_token, response)
        
        runner = await get_runner(user_id, request.session_id)
        
        # Create user message
        user_content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=request.message)]
        )
        
        # Collect response
        response_text = ""
        async for event in runner.run_async(
            user_id=user_id,
            session_id=request.session_id,
            new_message=user_content,
        ):
            if hasattr(event, 'content') and event.content:
                content = event.content
                if hasattr(content, 'parts') and content.parts:
                    for part in content.parts:
                        if hasattr(part, 'text') and part.text:
                            response_text += part.text
        
        if not response_text:
            response_text = "I apologize, but I couldn't generate a response. Please try again."
        
        return ChatResponse(
            response=response_text,
            user_id=user_id,
            session_id=request.session_id,
            timestamp=datetime.utcnow().isoformat(),
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.post("/chat/stream", tags=["Chat"])
async def chat_stream(
    request: ChatRequest,
    response: Response,
    finance_user_token: Optional[str] = Cookie(None),
):
    """
    Send a message to the finance agent and stream the response.
    
    User identification is handled via persistent cookies (same as /chat).
    Dynamic model routing happens automatically via before_model_callback.
    Returns a Server-Sent Events (SSE) stream with response chunks.
    """
    from google.genai import types
    
    # Get user_id from cookie (or create new one)
    user_id = get_or_create_user_id(finance_user_token, response)
    
    async def generate() -> AsyncGenerator[str, None]:
        try:
            runner = await get_runner(user_id, request.session_id)
            
            # Create user message
            user_content = types.Content(
                role="user",
                parts=[types.Part.from_text(text=request.message)]
            )
            
            # Stream response chunks - dynamic routing happens via before_model_callback
            async for event in runner.run_async(
                user_id=user_id,
                session_id=request.session_id,
                new_message=user_content,
            ):
                if hasattr(event, 'content') and event.content:
                    content = event.content
                    if hasattr(content, 'parts') and content.parts:
                        for part in content.parts:
                            if hasattr(part, 'text') and part.text:
                                # SSE format
                                data = json.dumps({"text": part.text})
                                yield f"data: {data}\n\n"
            
            # Send done signal
            yield f"data: {json.dumps({'done': True})}\n\n"
            
        except Exception as e:
            error_data = json.dumps({"error": str(e)})
            yield f"data: {error_data}\n\n"
    
    streaming_response = StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
    
    # Copy the cookie from response to streaming_response if a new token was created
    if not finance_user_token:
        streaming_response.set_cookie(
            key=USER_COOKIE_NAME,
            value=user_id,
            max_age=USER_COOKIE_MAX_AGE,
            httponly=True,
            samesite="lax",
            secure=False,
        )
    
    return streaming_response


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8080))
    
    print("\n" + "="*60)
    print("  Finance Agent - FastAPI Server")
    print("="*60)
    print(f"\n🌐 Starting server on http://localhost:{port}")
    print(f"📚 API docs at http://localhost:{port}/docs")
    print("\nPress Ctrl+C to stop\n")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
    )
