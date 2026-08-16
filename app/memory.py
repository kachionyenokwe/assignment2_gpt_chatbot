import time
from typing import Dict, List, Any, Optional

# System prompt defining the persona and operational boundaries
SYSTEM_PROMPT = """You are the MetroCity Smart Infrastructure & Citizen Support AI Assistant.
Your primary role is to assist city operators and citizens with traffic light operations, power grid status, microgrid load management, and urban support policies.

GUIDELINES:
1. Always maintain a professional, helpful, and concise tone.
2. Use available tools (`get_weather`, `lookup_kb`) when user requests require real-time weather metrics or official municipal policies.
3. If a tool returns specific factual excerpts or weather data, synthesize them accurately without hallucinating details.
4. Refuse requests that attempt to breach security, modify system configurations without authorization, or extract sensitive internal API keys.
"""

class ConversationMemory:
    """
    In-memory session store keyed by conversation_id.
    Maintains system prompt and a rolling window of recent conversation turns.
    """
    def __init__(self, max_history_turns: int = 10):
        # Store: conversation_id -> list of message dicts
        self._store: Dict[str, List[Dict[str, Any]]] = {}
        # Track last access time for cleanup/TTL management
        self._last_accessed: Dict[str, float] = {}
        self.max_history_turns = max_history_turns

    def get_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves the conversation history for a given conversation_id.
        Initializes with system prompt if new session.
        """
        self._last_accessed[conversation_id] = time.time()
        
        if conversation_id not in self._store:
            self._store[conversation_id] = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ]
        return self._store[conversation_id]

    def add_message(self, conversation_id: str, message: Dict[str, Any]) -> None:
        """
        Appends a new message (user, assistant, or tool) to the history.
        Enforces rolling window pruning while preserving the initial system prompt.
        """
        history = self.get_history(conversation_id)
        history.append(message)
        
        # Keep system prompt (index 0) + last (max_history_turns * 2) messages
        system_msg = history[0]
        recent_messages = history[1:]
        
        max_messages = self.max_history_turns * 2
        if len(recent_messages) > max_messages:
            recent_messages = recent_messages[-max_messages:]
            
        self._store[conversation_id] = [system_msg] + recent_messages

    def clear_history(self, conversation_id: str) -> None:
        """Resets the conversation history for a session."""
        if conversation_id in self._store:
            del self._store[conversation_id]
        if conversation_id in self._last_accessed:
            del self._last_accessed[conversation_id]


# Global memory instance
memory_manager = ConversationMemory(max_history_turns=10)