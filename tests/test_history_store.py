"""
Tests for SQLite-based conversation history store.
"""

import pytest
import os
import tempfile
import shutil
from src.memory.history_store import HistoryStore


class TestHistoryStore:
    """Test conversation history persistence."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_history.db")
        store = HistoryStore()
        # Override db_path for testing
        store.db_path = db_path
        store._ensure_database()
        yield store
        # Cleanup - close any open connections first (Windows file locking)
        try:
            # Force close any open connections by creating a new one and closing it
            import sqlite3
            conn = sqlite3.connect(db_path)
            conn.close()
        except:
            pass
        # Wait a bit for file handles to release (Windows)
        import time
        time.sleep(0.1)
        try:
            shutil.rmtree(temp_dir)
        except PermissionError:
            # On Windows, sometimes files are still locked - try again after a delay
            time.sleep(0.5)
            try:
                shutil.rmtree(temp_dir)
            except:
                pass  # Ignore cleanup errors in tests
    
    def test_create_session(self, temp_db):
        """Test session creation."""
        temp_db.create_session("test_session_1", "test_user")
        
        messages = temp_db.get_session_messages("test_session_1")
        assert len(messages) == 0  # No messages yet, but session exists
    
    def test_append_message(self, temp_db):
        """Test message storage and retrieval."""
        session_id = "test_session_1"
        user_id = "test_user"
        
        # Append user message
        temp_db.append_message(session_id, user_id, "user", "Tell me about Bitcoin")
        
        # Append assistant message
        temp_db.append_message(
            session_id, 
            user_id, 
            "assistant", 
            "Bitcoin is a cryptocurrency...",
            metadata={"evaluation_score": 0.9}
        )
        
        # Retrieve messages
        messages = temp_db.get_session_messages(session_id)
        
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Tell me about Bitcoin"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["metadata"]["evaluation_score"] == 0.9
    
    def test_get_messages_since(self, temp_db):
        """Test temporal message retrieval."""
        import time
        session_id = "test_session_1"
        user_id = "test_user"
        
        # Add messages
        temp_db.append_message(session_id, user_id, "user", "Message 1")
        time.sleep(1)  # Ensure different timestamps
        since_ts = int(time.time())
        time.sleep(1)
        temp_db.append_message(session_id, user_id, "user", "Message 2")
        
        # Get messages since timestamp
        recent_messages = temp_db.get_messages_since(user_id, since_ts)
        
        assert len(recent_messages) >= 1
        assert any(msg["content"] == "Message 2" for msg in recent_messages)
    
    def test_metadata_storage(self, temp_db):
        """Test that metadata is properly stored and retrieved."""
        session_id = "test_session_1"
        user_id = "test_user"
        
        metadata = {
            "evaluation_score": 0.85,
            "evaluation_feedback": "Good response",
            "revision_count": 0
        }
        
        temp_db.append_message(
            session_id,
            user_id,
            "assistant",
            "Test response",
            metadata=metadata
        )
        
        messages = temp_db.get_session_messages(session_id)
        assert messages[0]["metadata"] == metadata
    
    def test_session_touch(self, temp_db):
        """Test that session last_active_at is updated."""
        session_id = "test_session_1"
        user_id = "test_user"
        
        temp_db.create_session(session_id, user_id)
        initial_time = temp_db._execute(
            "SELECT last_active_at FROM sessions WHERE id = ?",
            (session_id,),
            fetch=True
        )["last_active_at"]
        
        import time
        time.sleep(1)
        temp_db.touch_session(session_id)
        
        updated_time = temp_db._execute(
            "SELECT last_active_at FROM sessions WHERE id = ?",
            (session_id,),
            fetch=True
        )["last_active_at"]
        
        assert updated_time > initial_time

