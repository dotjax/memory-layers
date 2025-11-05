from __future__ import annotations

import time
import traceback

from typing import Callable, Dict, MutableSequence, Optional, TypeVar
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

MessageList = MutableSequence[Dict[str, str]]
T = TypeVar("T")

_MODEL_CACHE: Dict[str, SentenceTransformer] = {}
_QDRANT_CACHE: Dict[str, QdrantClient] = {}


def _generate_memory_id(collection: str, existing_id: Optional[str] = None) -> str:
    """Generate a unique memory ID with collection prefix."""
    import uuid
    if existing_id:
        return f"{collection[:2]}_{existing_id[:8]}"
    return f"{collection[:2]}_{uuid.uuid4().hex[:8]}"


def _format_memories_json(memories: list[dict[str, any]]) -> str:
    """Format multiple memories as JSON array string."""
    import json
    from datetime import datetime, timezone
    if not memories:
        return "[]"
    
    memory_objects = []
    for memory in memories:
        memory_id = _generate_memory_id(
            memory["collection"], 
            memory.get("existing_id")
        )
        
        memory_obj = {
            "memory_id": memory_id,
            "collection": memory["collection"],
            "timestamp": memory.get("timestamp", datetime.now(timezone.utc).isoformat()),
            "content": memory["content"]
        }
        memory_objects.append(memory_obj)
    
    return json.dumps(memory_objects, indent=2)


def append_system_context(messages: MessageList, context: str) -> None:
    if not context:
        return
    if not messages:
        messages.insert(0, {"role": "system", "content": context})
        return
    first_message = messages[0]
    if first_message.get("role") == "system":
        first_message["content"] = f"{first_message.get('content', '')}{context}"
    else:
        messages.insert(0, {"role": "system", "content": context})

def run_qdrant_operation(
    operation: Callable[[], T],
    log: Callable[[str, str], None],
    *,
    description: str,
    retries: int = 1,
    backoff_seconds: float = 0.5,
) -> T:
    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            return operation()
        except Exception as exc:  
            last_error = exc
            log(f"{description} failed (attempt {attempt + 1}/{retries + 1}): {exc}", "ERROR")
            if attempt == retries:
                raise
            time.sleep(backoff_seconds)
    raise RuntimeError(f"{description} failed after {retries + 1} attempts") from last_error


class Filter:
    class Valves(BaseModel):
        qdrant_host: str = Field(
            default="localhost",
            description="Qdrant server host"
        )
        qdrant_port: int = Field(
            default=6333,
            description="Qdrant server port"
        )
        collection_name: str = Field(
            default="symbolic",
            description="Qdrant collection name for symbolic memories"
        )
        embedding_model: str = Field(
            default="/home/ubuntu/dev/models/embedding/mxbai-embed-large-v1",
            description="Sentence transformer model for embeddings"
        )
        embedding_device: str = Field(
            default="cpu",
            description="Device for embedding model (cpu/cuda)"
        )
        top_k_symbolic: int = Field(
            default=5,
            description="Number of symbolic memories to retrieve"
        )
        enabled: bool = Field(
            default=True,
            description="Enable/disable symbolic memory system"
        )
        debug_logging: bool = Field(
            default=True,
            description="Enable detailed debug logging"
        )

    def __init__(self) -> None:
        self.valves = self.Valves()
        self._qdrant: Optional[QdrantClient] = None
        self._embedding_model: Optional[SentenceTransformer] = None

    def _log(self, message: str, level: str = "INFO"):
        if level == "DEBUG" and not self.valves.debug_logging:
            return
        print(f"[Symbolic Memory] {level}: {message}")

    @property
    def qdrant(self) -> QdrantClient:
        cache_key = f"{self.valves.qdrant_host}:{self.valves.qdrant_port}"
        if cache_key not in _QDRANT_CACHE:
            try:
                self._log(f"Connecting to Qdrant at {self.valves.qdrant_host}:{self.valves.qdrant_port} (FIRST CONNECTION - caching in memory)", "INFO")
                _QDRANT_CACHE[cache_key] = QdrantClient(
                    host=self.valves.qdrant_host,
                    port=self.valves.qdrant_port,
                    timeout=5.0
                )
                run_qdrant_operation(
                    _QDRANT_CACHE[cache_key].get_collections,
                    self._log,
                    description="Qdrant get_collections",
                )
                self._log("Qdrant client cached in memory permanently", "DEBUG")
            except Exception as exc:
                self._log(f"Failed to connect to Qdrant: {exc}", "ERROR")
                raise
        else:
            self._log("Using cached Qdrant client", "DEBUG")
        return _QDRANT_CACHE[cache_key]

    @property
    def embedding_model(self) -> SentenceTransformer:
        cache_key = f"{self.valves.embedding_model}_{self.valves.embedding_device}"
        if cache_key not in _MODEL_CACHE:
            try:
                self._log(f"Loading embedding model: {self.valves.embedding_model} (FIRST LOAD - caching in memory)", "INFO")
                _MODEL_CACHE[cache_key] = SentenceTransformer(
                    self.valves.embedding_model,
                    device=self.valves.embedding_device
                )
                self._log("Embedding model cached in memory permanently", "DEBUG")
            except Exception as exc:
                self._log(f"Failed to load embedding model: {exc}", "ERROR")
                raise
        else:
            self._log("Using cached embedding model", "DEBUG")
        return _MODEL_CACHE[cache_key]

    def inlet(self, body: dict, __user__: dict) -> dict:
        if not self.valves.enabled:
            self._log("Symbolic memory retrieval disabled via configuration", "DEBUG")
            return body

        try:
            messages = body.get("messages", [])
            if not messages:
                self._log("No messages in body", "DEBUG")
                return body

            last_message = messages[-1]
            if last_message.get("role") != "user":
                self._log("Last message is not from user", "DEBUG")
                return body

            user_message = last_message.get("content", "")
            if not user_message or not user_message.strip():
                self._log("User message is empty", "DEBUG")
                return body

            self._log(f"Processing user message: {user_message[:100]}...", "DEBUG")
            assistant_messages = []
            for msg in reversed(messages[:-1]):
                role = msg.get("role", "")
                if role != "system" and role != "user":
                    content = msg.get("content", "")
                    if content:
                        assistant_messages.append(content)
                    if len(assistant_messages) == 2:
                        break

            assistant_message_1 = assistant_messages[1] if len(assistant_messages) > 1 else ""
            assistant_message_2 = assistant_messages[0] if assistant_messages else ""

            combined_text = f"{assistant_message_1}\n{assistant_message_2}\n{user_message}"
            self._log("Generating embedding for combined messages", "DEBUG")
            embedding = self.embedding_model.encode(combined_text)

            self._log("Searching symbolic collection", "DEBUG")
            results = run_qdrant_operation(
                lambda: self.qdrant.search(
                    collection_name=self.valves.collection_name,
                    query_vector=embedding,
                    limit=self.valves.top_k_symbolic,
                    with_payload=True
                ),
                self._log,
                description="Qdrant symbolic search",
            )

            if not results:
                self._log("No symbolic knowledge found", "DEBUG")
                return body
            
            # Transform symbolic summaries to JSON format
            json_memories = []
            for result in results:
                summary_text = result.payload.get("symbolic_summary", "")
                if not summary_text:
                    summary_text = result.payload.get("content", "")
                
                if summary_text:
                    # Create structured content
                    memory_content = {
                        "symbolic_summary": summary_text,
                        "relevance_score": result.score,
                        "knowledge_type": "symbolic",
                        "abstraction_level": "high",
                        "source": "knowledge_base"
                    }
                    
                    json_memories.append({
                        "collection": "symbolic",
                        "content": memory_content,
                        "timestamp": None,  # Symbolic knowledge doesn't have timestamps
                        "existing_id": str(result.id)
                    })

            if not json_memories:
                self._log("No valid symbolic summaries found", "DEBUG")
                return body
            
            context = _format_memories_json(json_memories)
            append_system_context(body["messages"], context)
            self._log(f"Injected {len(json_memories)} symbolic summaries into context", "DEBUG")

        except Exception as exc:
            self._log(f"Error in inlet(): {exc}", "ERROR")
            if self.valves.debug_logging:
                self._log(traceback.format_exc(), "ERROR")
        return body