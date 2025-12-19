"""
Episodic Memory Filter for Open WebUI

This filter stores per-user conversation snippets in Qdrant and retrieves
relevant memories before each assistant response.

Flow:
    1. Inlet: capture the latest user message, retrieve similar memories,
       and inject them into the system context.
    2. Outlet: store the assistant response and an optional user+assistant pair.

Memory Injection Format:
    [
        {
            "memory_id": "ep_a1b2c3d4",
            "collection": "episodic",
            "timestamp": "2025-11-04T20:30:00+00:00",
            "content": {
                "narrative": "...",
                "role": "user",
                "speaker": "USER",
                "participants": ["ASSISTANT", "USER"],
                "relevance_score": 0.78
            }
        }
    ]

Technical Details:
    - Embedding Model: mixedbread-ai/mxbai-embed-large-v1 (1024 dims)
    - Similarity: cosine distance
    - Storage: Qdrant vector database
    - Lazy Loading: models load on first use

Author: dotjax
License: GPL-3.0
Repository: https://github.com/dotjax/memory-layers
"""

from __future__ import annotations

import json
import time
import traceback
import uuid
from collections import OrderedDict
from collections.abc import Callable, MutableSequence, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, TypeVar

from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter as QdrantFilter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    SearchRequest,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

ASSISTANT_ID = "assistant"  # Generic assistant identifier
USER_METADATA_KEY = "_episodic_user_id"
USER_MESSAGE_KEY = "_episodic_user_message"
USER_CONVERSATION_KEY = "_episodic_user_conversation_id"
LAST_MESSAGE_ID_KEY = "_episodic_last_message_id"
DEFAULT_EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"


@dataclass
class _CacheEntry:
    value: Any
    last_used_monotonic: float


_MODEL_CACHE: "OrderedDict[str, _CacheEntry]" = OrderedDict()
_QDRANT_CACHE: "OrderedDict[str, _CacheEntry]" = OrderedDict()

MessageList = MutableSequence[dict[str, str]]
T = TypeVar("T")


def _generate_memory_id(collection: str, existing_id: Optional[str] = None) -> str:
    """Generate a unique memory ID with collection prefix."""
    if existing_id:
        return f"{collection[:2]}_{existing_id[:8]}"
    return f"{collection[:2]}_{uuid.uuid4().hex[:8]}"


def _format_memories_json(memories: list[dict[str, Any]]) -> str:
    """Format multiple memories as JSON array string."""
    if not memories:
        return "[]"

    memory_objects = []
    for memory in memories:
        memory_id = _generate_memory_id(
            memory["collection"],
            memory.get("existing_id"),
        )

        memory_obj = {
            "memory_id": memory_id,
            "collection": memory["collection"],
            "timestamp": memory.get("timestamp", datetime.now(timezone.utc).isoformat()),
            "content": memory["content"],
        }
        memory_objects.append(memory_obj)

    return json.dumps(memory_objects, indent=2)


def append_system_context(messages: MessageList, context: str) -> None:
    if not context:
        return
    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": context})
        return
    first_message = messages[0]
    existing = first_message.get("content", "")
    separator = "\n" if existing and not existing.endswith("\n") else ""
    first_message["content"] = f"{existing}{separator}{context}"


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


def _prune_lru_cache(
    cache: "OrderedDict[str, _CacheEntry]",
    *,
    max_items: int,
    ttl_seconds: int,
    on_evict: Optional[Callable[[Any], None]] = None,
) -> None:
    if not cache:
        return
    now = time.monotonic()

    if ttl_seconds > 0:
        expired_keys = [
            key
            for key, entry in cache.items()
            if now - entry.last_used_monotonic > ttl_seconds
        ]
        for key in expired_keys:
            entry = cache.pop(key, None)
            if entry is not None and on_evict is not None:
                try:
                    on_evict(entry.value)
                except Exception:
                    pass

    if max_items <= 0:
        return
    while len(cache) > max_items:
        _, entry = cache.popitem(last=False)
        if on_evict is not None:
            try:
                on_evict(entry.value)
            except Exception:
                pass


def _cache_get(
    cache: "OrderedDict[str, _CacheEntry]",
    key: str,
    *,
    ttl_seconds: int,
) -> Optional[Any]:
    entry = cache.get(key)
    if entry is None:
        return None
    entry.last_used_monotonic = time.monotonic()
    cache.move_to_end(key)
    return entry.value


def _to_float_vector(vector: Any) -> list[float]:
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return [float(value) for value in vector]


def _cache_put(
    cache: "OrderedDict[str, _CacheEntry]",
    key: str,
    value: Any,
) -> None:
    cache[key] = _CacheEntry(value=value, last_used_monotonic=time.monotonic())
    cache.move_to_end(key)


@dataclass(frozen=True)
class RetrievalQuery:
    label: str
    text: str
    query_filter: QdrantFilter


class Filter:
    class Valves(BaseModel):
        qdrant_host: str = Field(default="localhost", description="Qdrant server host (if server is running)")
        qdrant_port: int = Field(default=6333, description="Qdrant server port (if server is running)")
        storage_path: str = Field(default="./qdrant_storage", description="Local storage path for embedded mode")
        collection_name: str = Field(default="episodic", description="Qdrant collection name")
        embedding_model: str = Field(
            default=DEFAULT_EMBEDDING_MODEL,
            description="SentenceTransformer model name or path",
        )
        embedding_device: str = Field(default="cpu", description="Embedding device (cpu/cuda)")
        top_k: int = Field(
            default=30,
            ge=0,
            description="Total memories to retrieve across strategies",
        )
        retrieval_overfetch_factor: float = Field(
            default=1.2,
            ge=1.0,
            le=3.0,
            description="Multiplier for per-strategy retrieval limits to offset dedupe/threshold filtering",
        )
        similarity_threshold: float = Field(
            default=0.4,
            ge=0.0,
            le=1.0,
            description="Minimum similarity score for a memory to be considered relevant",
        )
        max_cached_models: int = Field(
            default=2,
            ge=0,
            description="Maximum cached embedding models (0 = unlimited)",
        )
        max_cached_qdrant_clients: int = Field(
            default=4,
            ge=0,
            description="Maximum cached Qdrant clients (0 = unlimited)",
        )
        cache_ttl_seconds: int = Field(
            default=0,
            ge=0,
            description="Cache TTL in seconds for models/clients (0 = no TTL)",
        )
        user_display_name: str = Field(default="USER", description="Display label for human messages")
        ai_display_name: str = Field(default="ASSISTANT", description="Display label for assistant messages")
        enabled: bool = Field(default=True, description="Enable episodic memory system")
        inject_memories: bool = Field(default=True, description="Inject relevant memories into context")
        debug_logging: bool = Field(default=True, description="Emit verbose debug logs")

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
        cache_key = f"{self.valves.qdrant_host}:{self.valves.qdrant_port}:{self.valves.storage_path}"
        cached = _cache_get(_QDRANT_CACHE, cache_key, ttl_seconds=self.valves.cache_ttl_seconds)
        if cached is None:
            client = None
            try:
                self._log(
                    f"Attempting to connect to Qdrant server at {self.valves.qdrant_host}:{self.valves.qdrant_port}",
                    "INFO",
                )
                client = QdrantClient(
                    host=self.valves.qdrant_host,
                    port=self.valves.qdrant_port,
                    timeout=2.0,
                )
                run_qdrant_operation(
                    client.get_collections,
                    self._log,
                    description="qdrant.get_collections",
                    retries=0,
                )
                self._log(f"Connected to Qdrant server (dashboard available at http://{self.valves.qdrant_host}:{self.valves.qdrant_port}/dashboard)", "INFO")
            except Exception:
                self._log(f"Qdrant server not available, using embedded mode at {self.valves.storage_path}", "INFO")
                client = QdrantClient(path=self.valves.storage_path)
            _cache_put(_QDRANT_CACHE, cache_key, client)
            _prune_lru_cache(
                _QDRANT_CACHE,
                max_items=self.valves.max_cached_qdrant_clients,
                ttl_seconds=self.valves.cache_ttl_seconds,
                on_evict=lambda c: getattr(c, "close", lambda: None)(),
            )
            return client
        return cached

    @property
    def embedding_model(self) -> SentenceTransformer:
        cache_key = f"{self.valves.embedding_model}_{self.valves.embedding_device}"
        cached = _cache_get(_MODEL_CACHE, cache_key, ttl_seconds=self.valves.cache_ttl_seconds)
        if cached is None:
            try:
                self._log(f"Loading embedding model: {self.valves.embedding_model} (FIRST LOAD - caching in memory)", "INFO")
                model = SentenceTransformer(
                    self.valves.embedding_model,
                    device=self.valves.embedding_device,
                )
                _cache_put(_MODEL_CACHE, cache_key, model)
                _prune_lru_cache(
                    _MODEL_CACHE,
                    max_items=self.valves.max_cached_models,
                    ttl_seconds=self.valves.cache_ttl_seconds,
                )
                return model
            except Exception as exc:
                self._log(f"Failed to load embedding model: {exc}", "ERROR")
                raise
        return cached

    @property
    def vector_size(self) -> int:
        return self.embedding_model.get_sentence_embedding_dimension()

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
            vector_size = self.vector_size
            self._log(f"Creating collection '{collection_name}' with vector size {vector_size}", "INFO")
            run_qdrant_operation(
                lambda: self.qdrant.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                ),
                self._log,
                description=f"qdrant.create_collection[{collection_name}]",
            )
        else:
            info = run_qdrant_operation(
                lambda: self.qdrant.get_collection(collection_name=collection_name),
                self._log,
                description=f"qdrant.get_collection[{collection_name}]",
            )
            existing_size = self._extract_vector_size(info.config.params.vectors)  # type: ignore[attr-defined]
            if existing_size:
                expected_size = self.vector_size
                if existing_size != expected_size:
                    raise ValueError(
                        f"Collection '{collection_name}' vector size {existing_size} does not match model dimension {expected_size}",
                    )

        self._collection_initialized = True

    def _ensure_payload_indexes(self, collection_name: str) -> None:
        """Create payload indexes for filtered fields to keep searches fast as the collection grows."""

        def ensure_index(field_name: str, schema: PayloadSchemaType) -> None:
            try:
                run_qdrant_operation(
                    lambda: self.qdrant.create_payload_index(
                        collection_name=collection_name,
                        field_name=field_name,
                        field_schema=schema,
                    ),
                    self._log,
                    description=f"qdrant.create_payload_index[{field_name}]",
                )
            except UnexpectedResponse as exc:
                if getattr(exc, "status_code", None) == 409:
                    self._log(f"Payload index '{field_name}' already exists", "DEBUG")
                    return
                self._log_exception(f"Failed creating payload index '{field_name}'", exc)
            except Exception as exc:  # noqa: BLE001
                message = str(exc).lower()
                if "already exists" in message or "conflict" in message:
                    self._log(f"Payload index '{field_name}' already exists", "DEBUG")
                    return
                self._log_exception(f"Failed creating payload index '{field_name}'", exc)

        fields: dict[str, PayloadSchemaType] = {
            "user_id": PayloadSchemaType.KEYWORD,
            "role": PayloadSchemaType.KEYWORD,
            "message_type": PayloadSchemaType.KEYWORD,
            "conversation_id": PayloadSchemaType.UUID,
        }
        for field_name, schema in fields.items():
            ensure_index(field_name, schema)

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

    def inlet(self, body: dict[str, Any], __user__: dict[str, Any]) -> dict[str, Any]:
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
            if role != "user":
                self._log(
                    f"Latest message role '{role}' not stored in episodic memory; skipping inlet storage",
                    "DEBUG",
                )
                return body
            metadata[USER_CONVERSATION_KEY] = message_conversation_id
            metadata[USER_MESSAGE_KEY] = message_content
            previous_assistant_message = self._find_latest_assistant_message(messages[:-1])
            self._log(
                (
                    "Inlet captured user message "
                    f"conversation_id='{message_conversation_id}' for user_id='{metadata.get(USER_METADATA_KEY)}'"
                ),
                "DEBUG",
            )
            self._ensure_collection()
            if self.valves.inject_memories:
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
                self._log("Skipping memory injection (inject_memories=False)", "DEBUG")
            self._store_memory(
                content=message_content,
                role="user",
                conversation_id=message_conversation_id,
                user_id=metadata.get(USER_METADATA_KEY, "unknown"),
                message_type="individual",
                linked_ids=None,
            )
        except Exception as exc:  # noqa: BLE001
            self._log_exception("Error during inlet processing", exc)
        return body

    def outlet(self, body: dict[str, Any], __user__: dict[str, Any]) -> dict[str, Any]:
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
            user_id = metadata.get(USER_METADATA_KEY) or (__user__ or {}).get("id", "unknown")
            user_conversation_id = metadata.get(USER_CONVERSATION_KEY) or metadata.get(
                LAST_MESSAGE_ID_KEY
            )
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
                pair_content = (
                    f"{self.valves.user_display_name}: {user_message_content}\n"
                    f"{self.valves.ai_display_name}: {assistant_message}"
                )
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
        vector = _to_float_vector(self.embedding_model.encode(content))
        point_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        payload: dict[str, Any] = {
            "content": content,
            "role": role,
            "timestamp": timestamp,
            "conversation_id": conversation_id,
            "user_id": user_id,
            "assistant_id": ASSISTANT_ID,
            "message_type": message_type,
        }
        if linked_ids:
            payload["linked_ids"] = list(linked_ids)
        point = PointStruct(id=point_id, vector=vector, payload=payload)
        run_qdrant_operation(
            lambda: self.qdrant.upsert(
                collection_name=self.valves.collection_name,
                points=[point],
            ),
            self._log,
            description=f"qdrant.upsert[{role}:{message_type}]",
        )
        self._log(f"Persisted episodic memory point_id={point_id}", "DEBUG")

    def _retrieve_memories(self, user_message: str, ai_message: str, user_id: str) -> list[dict[str, Any]]:
        if self.valves.top_k <= 0:
            self._log("top_k <= 0; retrieval skipped", "DEBUG")
            return []

        k_total = self.valves.top_k
        if k_total == 1:
            budgets = {"pair": 1, "assistant": 0, "user": 0}
        elif k_total == 2:
            budgets = {"pair": 1, "assistant": 0, "user": 1}
        else:
            remaining = k_total - 3
            add_pair = int(remaining * 0.5 + 0.5)
            add_assistant = int(remaining * 0.25 + 0.5)
            add_user = remaining - add_pair - add_assistant
            budgets = {
                "pair": 1 + add_pair,
                "assistant": 1 + add_assistant,
                "user": 1 + add_user,
            }

        overfetch = self.valves.retrieval_overfetch_factor
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
        active_queries = [(q, budgets.get(q.label, 0)) for q in queries if budgets.get(q.label, 0) > 0]
        if not active_queries:
            return []
        embeddings = self.embedding_model.encode([q.text for q, _ in active_queries])
        vector_size = self.vector_size
        search_requests = []
        for (query, budget), embedding in zip(active_queries, embeddings):
            limit = max(1, int(budget * overfetch))
            vector = _to_float_vector(embedding)
            if len(vector) != vector_size:
                self._log(
                    f"Query embedding size {len(vector)} does not match model dimension {vector_size}; retrieval skipped",
                    "ERROR",
                )
                return []
            search_requests.append(
                SearchRequest(
                    vector=vector,
                    filter=query.query_filter,
                    limit=limit,
                    score_threshold=self.valves.similarity_threshold,
                )
            )
        memories: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        batch_results = run_qdrant_operation(
            lambda: self.qdrant.search_batch(
                collection_name=self.valves.collection_name,
                requests=search_requests,
            ),
            self._log,
            description="qdrant.search_batch",
        )
        for results in batch_results:
            for result in results:
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
        memories.sort(key=lambda memory: memory.get("score", 0.0), reverse=True)
        memories = memories[: self.valves.top_k]
        self._log(f"Hybrid retrieval produced {len(memories)} memories", "DEBUG")
        return memories

    def _format_memories(self, memories: Sequence[dict[str, Any]]) -> str:
        """Format episodic memories as JSON for the AI assistant's cognitive architecture."""
        if not memories:
            return "[]"

        user_label = self.valves.user_display_name
        assistant_label = self.valves.ai_display_name
        json_memories = []
        for memory in memories:
            role = memory.get("role", "conversation")
            content = memory.get("content", "")
            timestamp = memory.get("timestamp", "")

            # Determine participants based on role
            if role == "assistant":
                participants = [assistant_label, user_label]
                speaker = assistant_label
            elif role == "user":
                participants = [user_label, assistant_label]
                speaker = user_label
            else:  # pair or conversation
                participants = [assistant_label, user_label]
                speaker = "Conversation"

            # Create structured content
            memory_content = {
                "narrative": content,
                "role": role,
                "speaker": speaker,
                "participants": participants,
                "relevance_score": memory.get("score", 0.0),
            }

            json_memories.append(
                {
                    "collection": "episodic",
                    "content": memory_content,
                    "timestamp": timestamp,
                    "existing_id": memory.get("id"),
                }
            )

        return _format_memories_json(json_memories)

    @staticmethod
    def _find_latest_assistant_message(messages: Sequence[dict[str, Any]]) -> str:
        for message in reversed(messages):
            if message.get("role") == "assistant":
                return (message.get("content") or "").strip()
        return ""
