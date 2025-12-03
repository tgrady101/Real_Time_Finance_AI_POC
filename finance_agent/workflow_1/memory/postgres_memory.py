"""PostgreSQL-backed memory service for persistent cross-session recall.

Stores conversation memories in PostgreSQL for agent recall across sessions.
Sessions themselves are NOT stored here - Arize handles logging/monitoring.
Only searchable memories are persisted for the load_memory tool.
"""

import json
import os
from datetime import datetime
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor

from google.adk.memory.base_memory_service import (
    BaseMemoryService, 
    SearchMemoryResponse, 
    MemoryEntry
)
from google.adk.sessions import Session
from google.genai import types


class PostgresMemoryService(BaseMemoryService):
    """Memory service backed by PostgreSQL for persistent storage.
    
    Inherits from ADK's BaseMemoryService for compatibility with
    InvocationContext and Runner. Stores searchable memories that 
    the agent can recall using the load_memory tool.
    
    Usage:
        memory_service = PostgresMemoryService()
        await memory_service.add_session_to_memory(session)
        results = await memory_service.search_memory(app_name, user_id, query)
    """
    
    def __init__(self, db_url: Optional[str] = None):
        """Initialize PostgreSQL memory service.
        
        Args:
            db_url: PostgreSQL connection URL. Defaults to SESSION_DB_URL env var.
        """
        self.db_url = db_url or os.getenv("SESSION_DB_URL")
        if not self.db_url:
            raise ValueError("db_url or SESSION_DB_URL environment variable required")
        
        self._initialized = False
        self._ensure_table()
    
    def _get_connection(self):
        """Get a database connection."""
        return psycopg2.connect(self.db_url)
    
    def _ensure_table(self):
        """Create memories table if it doesn't exist."""
        if self._initialized:
            return
        
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS memories (
            id SERIAL PRIMARY KEY,
            app_name VARCHAR(255) NOT NULL,
            user_id VARCHAR(255) NOT NULL,
            session_id VARCHAR(255),
            content TEXT NOT NULL,
            summary TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata JSONB
        );
        """
        
        create_index_sql = """
        CREATE INDEX IF NOT EXISTS idx_memories_app_user 
        ON memories(app_name, user_id);
        """
        
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(create_table_sql)
                    cur.execute(create_index_sql)
                conn.commit()
            self._initialized = True
            print("🧠 PostgresMemoryService initialized (memories table ready)")
        except Exception as e:
            print(f"⚠️ PostgresMemoryService table setup warning: {e}")
            self._initialized = True  # Continue anyway
    
    async def add_session_to_memory(self, session: Session) -> None:
        """Add a session's conversation to memory.
        
        Extracts conversation content and stores it as a searchable memory.
        
        Args:
            session: The completed session to add to memory
        """
        # Extract conversation content from session
        conversation_parts = []
        
        # Try different session formats (ADK session structure varies)
        events = getattr(session, 'events', None) or getattr(session, 'history', [])
        
        for event in events:
            content = getattr(event, 'content', None)
            if content:
                role = getattr(content, 'role', 'unknown')
                parts = getattr(content, 'parts', [])
                for part in parts:
                    text = getattr(part, 'text', None)
                    if text:
                        conversation_parts.append(f"{role}: {text}")
        
        if not conversation_parts:
            return  # No content to store
        
        content = "\n".join(conversation_parts)
        summary = content[:500] + "..." if len(content) > 500 else content
        
        app_name = getattr(session, 'app_name', 'finance_assistant')
        user_id = getattr(session, 'user_id', 'default_user')
        session_id = getattr(session, 'id', None) or getattr(session, 'session_id', None)
        
        insert_sql = """
            INSERT INTO memories (app_name, user_id, session_id, content, summary, metadata)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(insert_sql, (
                        app_name,
                        user_id,
                        session_id,
                        content,
                        summary,
                        json.dumps({
                            "created_at": datetime.utcnow().isoformat(),
                            "message_count": len(conversation_parts),
                        }),
                    ))
                conn.commit()
            print(f"🧠 Memory saved: {len(conversation_parts)} messages from user {user_id}")
        except Exception as e:
            print(f"⚠️ Failed to save memory: {e}")
    
    async def search_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        query: str,
    ) -> SearchMemoryResponse:
        """Search memories for relevant content.
        
        Uses PostgreSQL ILIKE search to find matching memories.
        For generic recall queries (past questions, history, etc.), returns recent memories.
        
        Args:
            app_name: Application name
            user_id: User ID to search memories for
            query: Search query
            
        Returns:
            SearchMemoryResponse with matching MemoryEntry objects
        """
        # Check if this is a generic recall query (user asking about past conversations)
        recall_patterns = [
            "past", "history", "previous", "before", "asked", "questions",
            "discussed", "conversations", "remember", "recall", "what did"
        ]
        is_recall_query = any(pattern in query.lower() for pattern in recall_patterns)
        
        if is_recall_query:
            # For recall queries, just return recent memories for this user
            search_sql = """
                SELECT id, content, summary, session_id, created_at
                FROM memories
                WHERE app_name = %s 
                  AND user_id = %s
                ORDER BY created_at DESC
                LIMIT 5
            """
            params = (app_name, user_id)
            print(f"🧠 Memory recall query detected, returning recent memories for user {user_id}")
        else:
            # For specific queries, use ILIKE search
            search_sql = """
                SELECT id, content, summary, session_id, created_at
                FROM memories
                WHERE app_name = %s 
                  AND user_id = %s
                  AND (content ILIKE %s OR summary ILIKE %s)
                ORDER BY created_at DESC
                LIMIT 5
            """
            search_pattern = f"%{query}%"
            params = (app_name, user_id, search_pattern, search_pattern)
        
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(search_sql, params)
                    rows = cur.fetchall()
            print(f"🧠 Memory search found {len(rows)} results for query: {query[:50]}...")
        except Exception as e:
            print(f"⚠️ Memory search error: {e}")
            rows = []
        
        # Return in ADK-compatible format using proper MemoryEntry objects
        memories = []
        for row in rows:
            # Create types.Content with the memory text
            memory_content = types.Content(
                role="user",  # Memory content is treated as user context
                parts=[types.Part.from_text(text=row["content"])]
            )
            
            memories.append(MemoryEntry(
                content=memory_content,
                id=str(row["id"]),
                timestamp=str(row["created_at"]) if row["created_at"] else None,
                custom_metadata={
                    "session_id": row["session_id"],
                    "summary": row["summary"],
                },
            ))
        
        return SearchMemoryResponse(memories=memories)
