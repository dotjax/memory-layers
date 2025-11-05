from __future__ import annotations

import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, MutableSequence, Optional, Sequence, TypeVar

from pydantic import BaseModel, Field, validator
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter as QdrantFilter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer
from zoneinfo import ZoneInfo

MODEL_VECTOR_SIZE = 1024
KAELAN_ID = "kaelan"
USER_METADATA_KEY = "_episodic_user_id"
USER_MESSAGE_KEY = "_episodic_user_message"
USER_CONVERSATION_KEY = "_episodic_user_conversation_id"
LAST_MESSAGE_ID_KEY = "_episodic_last_message_id"
LAST_MESSAGE_ROLE_KEY = "_episodic_last_role"
MEMORY_SECTION_HEADER = "## Relevant Memories"
MEMORY_SECTION_FOOTER = "---"
DEFAULT_MODEL_PATH = "/home/ubuntu/dev/models/embedding/mxbai-embed-large-v1"
LOCAL_TIMEZONE = ZoneInfo("America/Chicago")

_MODEL_CACHE: Dict[str, SentenceTransformer] = {}
_QDRANT_CACHE: Dict[str, QdrantClient] = {}

MessageList = MutableSequence[Dict[str, str]]
T = TypeVar("T")


def _generate_memory_id(collection: str, existing_id: Optional[str] = None) -> str:
    """Generate a unique memory ID with collection prefix."""
    if existing_id:
        return f"{collection[:2]}_{existing_id[:8]}"
    return f"{collection[:2]}_{uuid.uuid4().hex[:8]}"


def _format_memories_json(memories: list[dict[str, Any]]) -> str:
    """Format multiple memories as JSON array string."""
    import json
    from datetime import timezone
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
        existing = first_message.get("content", "")
        first_message["content"] = f"{existing}{context}"
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
                break
            time.sleep(backoff_seconds)
    raise RuntimeError(f"{description} failed after {retries + 1} attempts") from last_error


@dataclass(frozen=True)
class RetrievalQuery:
    label: str
    text: str
    query_filter: QdrantFilter


class Filter:
    class Valves(BaseModel):
        qdrant_host: str = Field(default="localhost", description="Qdrant server host")
        qdrant_port: int = Field(default=6333, description="Qdrant server port")
        collection_name: str = Field(default="episodic", description="Qdrant server port")
        embedding_model: str = Field(
            default=DEFAULT_MODEL_PATH,
            description="SentenceTransformer model path or name",
        )
        embedding_device: str = Field(default="cpu", description="Embedding device (cpu/cuda)")
        top_k: int = Field(default=30, description="Total memories to retrieve across strategies")
        similarity_threshold: float = Field(
            default=0.4,
            ge=0.0,
            le=1.0,
            description="Minimum similarity score for a memory to be considered relevant",
        )
        user_display_name: str = Field(default="USER", description="Display label for human messages")
        ai_display_name: str = Field(default="ASSISTANT", description="Display label for assistant messages")
        enabled: bool = Field(default=True, description="Enable episodic memory system")
        inject_memories: bool = Field(default=True, description="Inject relevant memories into context")
        debug_logging: bool = Field(default=True, description="Emit verbose debug logs")

        @validator("top_k")
        def _validate_top_k(cls, value: int) -> int:  # noqa: N805
            if value < 0:
                raise ValueError("top_k must be non-negative")
            return value

    def __init__(self) -> None:
        self.valves = self.Valves()
        self._collection_initialized = False

    def _log(self, message: str, level: str = "INFO") -> None:
        if level == "DEBUG" and not self.valves.debug_logging:
            return
        print(f"[Episodic Memory] {level}: {message}")

    def _log_exception(self, message: str, exc: Exception) -> None:
        self._log(f"{message}: {exc}", "ERROR")
        if self.valves.debug_logging:
            self._log(traceback.format_exc(), "ERROR")

    @property
    def qdrant(self) -> QdrantClient:
        cache_key = f"{self.valves.qdrant_host}:{self.valves.qdrant_port}"
        if cache_key not in _QDRANT_CACHE:
            try:
                self._log(f"Connecting to Qdrant at {self.valves.qdrant_host}:{self.valves.qdrant_port} (FIRST CONNECTION - caching in memory)", "INFO")
                _QDRANT_CACHE[cache_key] = QdrantClient(
                    host=self.valves.qdrant_host,
                    port=self.valves.qdrant_port,
                    timeout=5.0,
                )
                run_qdrant_operation(
                    _QDRANT_CACHE[cache_key].get_collections,
                    self._log,
                    description="qdrant.get_collections",
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
                    device=self.valves.embedding_device,
                )
                self._log("Embedding model cached in memory permanently", "DEBUG")
            except Exception as exc:
                self._log(f"Failed to load embedding model: {exc}", "ERROR")
                raise
        else:
            self._log("Using cached embedding model", "DEBUG")
        return _MODEL_CACHE[cache_key]

    def _ensure_collection(self) -> None:
        if self._collection_initialized:
            return
        collection_name = self.valves.collection_name
        collections = run_qdrant_operation(
            self.qdrant.get_collections,
            self._log,
            description="qdrant.get_collections",
        ).collections
        exists = any(col.name == collection_name for col in collections)
        if not exists:
            self._log(f"Creating collection '{collection_name}' with vector size {MODEL_VECTOR_SIZE}", "INFO")
            run_qdrant_operation(
                lambda: self.qdrant.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=MODEL_VECTOR_SIZE, distance=Distance.COSINE),
                ),
                self._log,
                description=f"qdrant.create_collection[{collection_name}]",
            )
            self._log(f"Collection '{collection_name}' created", "DEBUG")
        else:
            info = run_qdrant_operation(
                lambda: self.qdrant.get_collection(collection_name=collection_name),
                self._log,
                description=f"qdrant.get_collection[{collection_name}]",
            )
            vector_size = self._extract_vector_size(info.config.params.vectors)  # type: ignore[attr-defined]
            if vector_size and vector_size != MODEL_VECTOR_SIZE:
                raise ValueError(
                    f"Collection '{collection_name}' vector size {vector_size} does not match required size {MODEL_VECTOR_SIZE}",
                )
            self._log(f"Collection '{collection_name}' already exists", "DEBUG")
        self._collection_initialized = True

    @staticmethod
    def _extract_vector_size(vectors_config: Any) -> Optional[int]:
        try:
            if hasattr(vectors_config, "size"):
                return int(vectors_config.size)
            if isinstance(vectors_config, dict):
                for vector in vectors_config.values():
                    if hasattr(vector, "size"):
                        return int(vector.size)
        except (TypeError, ValueError):
            return None
        return None

    def inlet(self, body: Dict[str, Any], __user__: Dict[str, Any]) -> Dict[str, Any]:
        if not self.valves.enabled:
            self._log("Episodic memory disabled via configuration", "DEBUG")
            return body
        try:
            messages = body.get("messages") or []
            if not messages:
                self._log("No messages found in request body", "DEBUG")
                return body
            last_message = messages[-1]
            role = last_message.get("role", "unknown")
            message_content = (last_message.get("content") or "").strip()
            if not message_content:
                self._log(f"Latest message with role '{role}' is empty; skipping inlet processing", "DEBUG")
                return body
            user_id = (__user__ or {}).get("id", "unknown")
            metadata = body.setdefault("metadata", {})
            metadata.setdefault(USER_METADATA_KEY, user_id)
            message_conversation_id = str(uuid.uuid4())
            metadata[LAST_MESSAGE_ID_KEY] = message_conversation_id
            metadata[LAST_MESSAGE_ROLE_KEY] = role
            if role == "user":
                metadata[USER_CONVERSATION_KEY] = message_conversation_id
                metadata[USER_MESSAGE_KEY] = message_content
            previous_assistant_message = self._find_latest_assistant_message(messages[:-1])
            self._log(
                (
                    "Inlet captured message "
                    f"role='{role}' conversation_id='{message_conversation_id}' for user_id='{metadata.get(USER_METADATA_KEY)}'"
                ),
                "DEBUG",
            )
            self._ensure_collection()
            if self.valves.inject_memories and role == "user":
                if previous_assistant_message:
                    memories = self._retrieve_memories(
                        message_content,
                        previous_assistant_message,
                        metadata[USER_METADATA_KEY],
                    )
                else:
                    memories = []
                    self._log("No prior assistant message located for memory retrieval", "DEBUG")
                if memories:
                    append_system_context(messages, self._format_memories(memories))
                    self._log(f"Injected {len(memories)} episodic memories into context", "DEBUG")
            else:
                self._log(
                    "Skipping hybrid retrieval "
                    f"(inject_memories={self.valves.inject_memories}, role='{role}')",
                    "DEBUG",
                )
            self._store_memory(
                content=message_content,
                role=role,
                conversation_id=message_conversation_id,
                user_id=metadata.get(USER_METADATA_KEY, "unknown"),
                message_type="individual",
                linked_ids=None,
            )
        except Exception as exc:  # noqa: BLE001
            self._log_exception("Error during inlet processing", exc)
        return body

    def outlet(self, body: Dict[str, Any], __user__: Dict[str, Any]) -> Dict[str, Any]:
        if not self.valves.enabled:
            return body
        try:
            messages = body.get("messages") or []
            if not messages:
                self._log("No messages available during outlet processing", "DEBUG")
                return body
            last_message = messages[-1]
            if last_message.get("role") != "assistant":
                self._log("Latest message is not from the assistant; skipping outlet processing", "DEBUG")
                return body
            assistant_message = (last_message.get("content") or "").strip()
            if not assistant_message:
                self._log("Assistant message is empty; skipping outlet processing", "DEBUG")
                return body
            metadata = body.get("metadata") or {}
            user_id = metadata.get(USER_METADATA_KEY, "unknown")
            user_conversation_id = metadata.get(LAST_MESSAGE_ID_KEY)
            user_message_content = metadata.get(USER_MESSAGE_KEY, "")
            self._ensure_collection()
            ai_conversation_id = str(uuid.uuid4())
            self._store_memory(
                content=assistant_message,
                role="assistant",
                conversation_id=ai_conversation_id,
                user_id=user_id,
                message_type="individual",
                linked_ids=None,
            )
            if user_conversation_id and user_message_content:
                pair_content = f"User: {user_message_content}\nMe: {assistant_message}"
                pair_conversation_id = str(uuid.uuid4())
                self._store_memory(
                    content=pair_content,
                    role="pair",
                    conversation_id=pair_conversation_id,
                    user_id=user_id,
                    message_type="pair",
                    linked_ids=[user_conversation_id, ai_conversation_id],
                )
        except Exception as exc:  # noqa: BLE001
            self._log_exception("Error during outlet processing", exc)
        return body

    def _store_memory(
        self,
        *,
        content: str,
        role: str,
        conversation_id: str,
        user_id: str,
        message_type: str,
        linked_ids: Optional[Sequence[str]],
    ) -> None:
        if not content.strip():
            self._log("Skipped storing blank content", "DEBUG")
            return
        embedding = self.embedding_model.encode(content)
        point_id = str(uuid.uuid4())
        timestamp = datetime.now(LOCAL_TIMEZONE).isoformat()
        payload: Dict[str, Any] = {
            "content": content,
            "role": role,
            "timestamp": timestamp,
            "conversation_id": conversation_id,
            "user_id": user_id,
            "kaelan_id": KAELAN_ID,
            "message_type": message_type,
        }
        if linked_ids:
            payload["linked_ids"] = list(linked_ids)
        point = PointStruct(id=point_id, vector=embedding, payload=payload)
        run_qdrant_operation(
            lambda: self.qdrant.upsert(
                collection_name=self.valves.collection_name,
                points=[point],
            ),
            self._log,
            description=f"qdrant.upsert[{role}:{message_type}]",
        )
        self._log(f"Persisted episodic memory point_id={point_id}", "DEBUG")

    def _retrieve_memories(self, user_message: str, ai_message: str, user_id: str) -> list[Dict[str, Any]]:
        if self.valves.top_k <= 0:
            self._log("top_k <= 0; retrieval skipped", "DEBUG")
            return []
        queries = [
            RetrievalQuery(
                label="pair",
                text=f"{user_message}\n{ai_message}",
                query_filter=QdrantFilter(
                    must=[
                        FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                        FieldCondition(key="message_type", match=MatchValue(value="pair")),
                    ],
                ),
            ),
            RetrievalQuery(
                label="assistant",
                text=ai_message,
                query_filter=QdrantFilter(
                    must=[
                        FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                        FieldCondition(key="role", match=MatchValue(value="assistant")),
                        FieldCondition(key="message_type", match=MatchValue(value="individual")),
                    ],
                ),
            ),
            RetrievalQuery(
                label="user",
                text=user_message,
                query_filter=QdrantFilter(
                    must=[
                        FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                        FieldCondition(key="role", match=MatchValue(value="user")),
                        FieldCondition(key="message_type", match=MatchValue(value="individual")),
                    ],
                ),
            ),
        ]
        limits = self._distribute_limits(self.valves.top_k, len(queries))
        memories: list[Dict[str, Any]] = []
        seen_ids: set[str] = set()
        for limit, query in zip(limits, queries):
            if limit <= 0:
                self._log(f"Skipping '{query.label}' retrieval due to zero limit", "DEBUG")
                continue
            embedding = self.embedding_model.encode(query.text)
            search_results = run_qdrant_operation(
                lambda: self.qdrant.search(
                    collection_name=self.valves.collection_name,
                    query_vector=embedding,
                    query_filter=query.query_filter,
                    limit=limit,
                    score_threshold=self.valves.similarity_threshold,
                ),
                self._log,
                description=f"qdrant.search[{query.label}]",
            )
            for result in search_results:
                point_id = str(result.id)
                if point_id in seen_ids:
                    continue
                seen_ids.add(point_id)
                payload = result.payload or {}
                memories.append(
                    {
                        "id": point_id,
                        "content": payload.get("content", ""),
                        "role": payload.get("role", "unknown"),
                        "timestamp": payload.get("timestamp", ""),
                        "score": result.score,
                    }
                )
        self._log(f"Hybrid retrieval produced {len(memories)} memories", "DEBUG")
        return memories

    @staticmethod
    def _distribute_limits(total: int, buckets: int) -> list[int]:
        if buckets <= 0:
            return []
        base = total // buckets
        remainder = total % buckets
        return [base + (1 if idx < remainder else 0) for idx in range(buckets)]

    def _format_memories(self, memories: Sequence[Dict[str, Any]]) -> str:
        """Format episodic memories as JSON according to Kaelan's cognitive architecture."""
        if not memories:
            return "[]"
        
        json_memories = []
        for memory in memories:
            role = memory.get("role", "conversation")
            content = memory.get("content", "")
            timestamp = memory.get("timestamp", "")
            
            # Determine participants based on role
            if role == "assistant":
                participants = ["Kaelan", "User"]
                speaker = "Kaelan"
            elif role == "user":
                participants = ["User", "Kaelan"]
                speaker = "User"
            else:  # pair or conversation
                participants = ["Kaelan", "User"]
                speaker = "Conversation"
            
            # Create structured content
            memory_content = {
                "narrative": content,
                "role": role,
                "speaker": speaker,
                "participants": participants,
                "relevance_score": memory.get("score", 0.0)
            }
            
            json_memories.append({
                "collection": "episodic",
                "content": memory_content,
                "timestamp": timestamp,
                "existing_id": memory.get("id")
            })
        
        return _format_memories_json(json_memories)

    def _format_timestamp(self, timestamp_str: str) -> str:
        if not timestamp_str:
            return "unknown time"
        try:
            normalized = timestamp_str.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(normalized)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=LOCAL_TIMEZONE)
            local_timestamp = parsed.astimezone(LOCAL_TIMEZONE)
            now_local = datetime.now(LOCAL_TIMEZONE)
            delta = now_local - local_timestamp
            if delta.total_seconds() < 0:
                delta = -delta
            if delta.days > 365:
                years = delta.days // 365
                relative = f"{years} year{'s' if years != 1 else ''} ago"
            elif delta.days > 30:
                months = delta.days // 30
                relative = f"{months} month{'s' if months != 1 else ''} ago"
            elif delta.days > 0:
                relative = f"{delta.days} day{'s' if delta.days != 1 else ''} ago"
            elif delta.seconds > 3600:
                hours = delta.seconds // 3600
                relative = f"{hours} hour{'s' if hours != 1 else ''} ago"
            else:
                minutes = max(delta.seconds // 60, 1)
                relative = f"{minutes} minute{'s' if minutes != 1 else ''} ago"
            absolute = local_timestamp.strftime("%Y-%m-%d %H:%M %Z")
            return f"{absolute} | {relative}"
        except Exception:
            return timestamp_str[:16].replace("T", " ") or timestamp_str

    def _display_info_for_role(self, role: str) -> tuple[str, str]:
        if role == "user":
            return self.valves.user_display_name, "user"
        if role == "assistant":
            return self.valves.ai_display_name, "assistant"
        return "CONVERSATION", "pair"

    @staticmethod
    def _find_latest_assistant_message(messages: Sequence[Dict[str, Any]]) -> str:
        for message in reversed(messages):
            if message.get("role") == "assistant":
                return (message.get("content") or "").strip()
        return ""