from __future__ import annotations

import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, MutableSequence, Optional, Tuple, TypeVar
import traceback
import numpy as np
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter as QdrantFilter, MatchValue, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

MessageList = MutableSequence[Dict[str, str]]
T = TypeVar("T")

_MODEL_CACHE: Dict[str, SentenceTransformer] = {}
_QDRANT_CACHE: Dict[str, QdrantClient] = {}


def _generate_memory_id(collection: str, existing_id: Optional[str] = None) -> str:
    """Generate a unique memory ID with collection prefix."""
    if existing_id:
        return f"{collection[:2]}_{existing_id[:8]}"
    return f"{collection[:2]}_{uuid.uuid4().hex[:8]}"


def _format_memories_json(memories: List[Dict[str, Any]]) -> str:
    """Format multiple memories as JSON array string."""
    import json
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
            default="associative",
            description="Qdrant collection name for associative memories"
        )
        embedding_model: str = Field(
            default="/home/ubuntu/dev/models/embedding/mxbai-embed-large-v1",
            description="Sentence transformer model for embeddings"
        )
        embedding_device: str = Field(
            default="cpu",
            description="Device for embedding model (cpu/cuda)"
        )
        vector_size: int = Field(
            default=1024,
            description="Vector dimension size"
        )
        top_k: int = Field(
            default=5,
            description="Number of associations to retrieve"
        )
        similarity_threshold: float = Field(
            default=0.3,
            description="Minimum similarity score for retrieval"
        )
        trigger_interval: int = Field(
            default=300,
            description="Seconds between activations (default: 5 minutes)"
        )
        enabled: bool = Field(
            default=True,
            description="Enable associative memory filter"
        )
        debug_logging: bool = Field(
            default=True,
            description="Enable debug logging"
        )
        min_concept_length: int = Field(
            default=3,
            description="Minimum length for concept extraction"
        )
        max_concepts: int = Field(
            default=5,
            description="Maximum concepts to extract per conversation"
        )
        association_strength_threshold: float = Field(
            default=0.4,
            description="Minimum association strength to store"
        )

    def __init__(self) -> None:
        self.valves = self.Valves()
        self._last_activation = 0
        self._concept_embedding_cache: Dict[str, np.ndarray] = {}

    def _log(self, message: str, level: str = "INFO") -> None:
        if self.valves.debug_logging:
            print(f"[Associative Memory] {level}: {message}")

    @property
    def qdrant(self) -> QdrantClient:
        cache_key = f"{self.valves.qdrant_host}:{self.valves.qdrant_port}"
        if cache_key not in _QDRANT_CACHE:
            self._log(f"Connecting to Qdrant at {self.valves.qdrant_host}:{self.valves.qdrant_port} (FIRST CONNECTION - caching in memory)", "INFO")
            _QDRANT_CACHE[cache_key] = QdrantClient(
                host=self.valves.qdrant_host,
                port=self.valves.qdrant_port
            )
            self._log("Qdrant client cached in memory permanently", "DEBUG")
        else:
            self._log("Using cached Qdrant client", "DEBUG")
        return _QDRANT_CACHE[cache_key]

    @property
    def embedding_model(self) -> SentenceTransformer:
        cache_key = f"{self.valves.embedding_model}_{self.valves.embedding_device}"
        if cache_key not in _MODEL_CACHE:
            self._log(f"Loading embedding model: {self.valves.embedding_model} (FIRST LOAD - caching in memory)", "INFO")
            _MODEL_CACHE[cache_key] = SentenceTransformer(
                self.valves.embedding_model,
                device=self.valves.embedding_device
            )
            self._log("Embedding model cached in memory permanently", "DEBUG")
        else:
            self._log("Using cached embedding model", "DEBUG")
        return _MODEL_CACHE[cache_key]

    def _ensure_collection(self) -> None:
        try:
            qdrant = self.qdrant
            collections = run_qdrant_operation(
                lambda: qdrant.get_collections(),
                self._log,
                description="Get collections"
            )
            collection_names = [c.name for c in collections.collections]
            if self.valves.collection_name not in collection_names:
                self._log(f"Creating collection: {self.valves.collection_name}", "INFO")
                run_qdrant_operation(
                    lambda: qdrant.create_collection(
                        collection_name=self.valves.collection_name,
                        vectors_config=VectorParams(
                            size=self.valves.vector_size,
                            distance=Distance.COSINE
                        )
                    ),
                    self._log,
                    description=f"Create collection {self.valves.collection_name}"
                )
                run_qdrant_operation(
                    lambda: qdrant.create_payload_index(
                        collection_name=self.valves.collection_name,
                        field_name="timestamp",
                        field_schema="datetime"
                    ),
                    self._log,
                    description="Qdrant create timestamp range index"
                )
            else:
                self._log(f"Collection {self.valves.collection_name} already exists", "DEBUG")
        except Exception as exc:
            self._log(f"Error ensuring collection: {exc}", "ERROR")
            if self.valves.debug_logging:
                self._log(traceback.format_exc(), "ERROR")
            raise

    def _should_trigger(self) -> bool:
        current_time = time.time()
        if current_time - self._last_activation >= self.valves.trigger_interval:
            self._last_activation = current_time
            return True
        return False

    def _extract_concepts(self, text: str) -> List[str]:
        concepts = []
        quoted = re.findall(r'"([^"]+)"', text)
        concepts.extend([q for q in quoted if len(q) >= self.valves.min_concept_length])
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        concepts.extend([c for c in capitalized if len(c) >= self.valves.min_concept_length])
        technical = re.findall(r'\b[a-zA-Z]+[-_][a-zA-Z]+\b', text)
        concepts.extend([t for t in technical if len(t) >= self.valves.min_concept_length])
        unique_concepts = list(dict.fromkeys(concepts))
        return unique_concepts[:self.valves.max_concepts]

    def _calculate_association_strength(
        self,
        concept1_embedding: np.ndarray,
        concept2_embedding: np.ndarray,
        concept1: str,
        concept2: str,
        context: str
    ) -> float:
        try:
            semantic_sim = float(np.dot(concept1_embedding, concept2_embedding))
        except Exception:
            semantic_sim = 0.0
        emotional_keywords = [
            "love", "hate", "fear", "joy", "sadness", "anger", "excitement", "calm",
            "warm", "cold", "bright", "dark", "safe", "dangerous", "comfortable", "uncomfortable"
        ]
        emotional_score = 0.0
        for keyword in emotional_keywords:
            if keyword in concept1.lower() or keyword in concept2.lower():
                emotional_score += 0.1
        emotional_score = min(emotional_score, 1.0)
        context_words = set(context.lower().split())
        concept1_words = set(concept1.lower().split())
        concept2_words = set(concept2.lower().split())
        shared_context = len(concept1_words & context_words) + len(concept2_words & context_words)
        thematic_score = min(shared_context / max(len(context_words), 1), 1.0)
        association_strength = (
            semantic_sim * 0.3 +
            emotional_score * 0.4 +
            thematic_score * 0.3
        )
        return association_strength

    def _extract_associations(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        if not messages:
            return []
        recent_text = " ".join([msg.get("content", "") for msg in messages[-5:]])
        concepts = self._extract_concepts(recent_text)
        if len(concepts) < 2:
            return []
        
        # Get embeddings from cache or compute
        concept_embeddings = []
        concepts_to_encode = []
        concepts_to_encode_indices = []
        
        for i, concept in enumerate(concepts):
            if concept in self._concept_embedding_cache:
                concept_embeddings.append(self._concept_embedding_cache[concept])
            else:
                concepts_to_encode.append(concept)
                concepts_to_encode_indices.append(i)
                concept_embeddings.append(None)
        
        # Batch encode only new concepts
        if concepts_to_encode:
            try:
                new_embeddings = self.embedding_model.encode(concepts_to_encode)
                for idx, embedding in zip(concepts_to_encode_indices, new_embeddings):
                    concept_embeddings[idx] = embedding
                    self._concept_embedding_cache[concepts[idx]] = embedding
                self._log(f"Encoded {len(concepts_to_encode)} new concepts, {len(concepts) - len(concepts_to_encode)} from cache", "DEBUG")
            except Exception as exc:
                self._log(f"Error encoding concepts: {exc}", "ERROR")
                return []
        else:
            self._log(f"All {len(concepts)} concepts retrieved from cache", "DEBUG")
        
        associations = []
        for i, concept1 in enumerate(concepts):
            for j, concept2 in enumerate(concepts[i+1:], start=i+1):
                strength = self._calculate_association_strength(
                    concept_embeddings[i],
                    concept_embeddings[j],
                    concept1,
                    concept2,
                    recent_text
                )
                if strength >= self.valves.association_strength_threshold:
                    associations.append({
                        "concept1": concept1,
                        "concept2": concept2,
                        "strength": strength,
                        "context": recent_text[:200] + "..." if len(recent_text) > 200 else recent_text,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
        return associations

    def _retrieve_associations(self, current_context: str, user_id: str) -> List[Dict[str, Any]]:
        try:
            model = self.embedding_model
            context_embedding = model.encode(current_context)
            qdrant = self.qdrant
            results = run_qdrant_operation(
                lambda: qdrant.search(
                    collection_name=self.valves.collection_name,
                    query_vector=context_embedding,
                    query_filter=QdrantFilter(
                        must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
                    ),
                    limit=self.valves.top_k,
                    score_threshold=self.valves.similarity_threshold,
                    with_payload=True
                ),
                self._log,
                description="Qdrant associative search"
            )
            associations = []
            for result in results:
                payload = result.payload
                associations.append({
                    "concept1": payload.get("concept1", ""),
                    "concept2": payload.get("concept2", ""),
                    "strength": payload.get("strength", 0.0),
                    "context": payload.get("context", ""),
                    "timestamp": payload.get("timestamp", ""),
                    "relevance_score": result.score
                })
            self._log(f"Retrieved {len(associations)} associations", "DEBUG")
            return associations
        except Exception as exc:
            self._log(f"Error retrieving associations: {exc}", "ERROR")
            return []

    def _format_associations(self, associations: List[Dict[str, Any]]) -> str:
        """Format associative memories as JSON according to Kaelan's cognitive architecture."""
        if not associations:
            return "[]"
        
        json_memories = []
        for assoc in associations:
            concept1 = assoc.get("concept1", "")
            concept2 = assoc.get("concept2", "")
            strength = assoc.get("strength", 0.0)
            context = assoc.get("context", "")
            timestamp = assoc.get("timestamp", "")
            relevance_score = assoc.get("relevance_score", 0.0)
            
            # Create structured content
            memory_content = {
                "concept1": concept1,
                "concept2": concept2,
                "association_strength": strength,
                "context": context,
                "relevance_score": relevance_score,
                "connection_type": "semantic_similarity"
            }
            
            json_memories.append({
                "collection": "associative",
                "content": memory_content,
                "timestamp": timestamp,
                "existing_id": None  # Associations don't have stored IDs
            })
        
        return _format_memories_json(json_memories)

    def _store_associations(
        self,
        body: dict,
        __user__: dict,
        associations: List[Dict[str, Any]]
    ) -> None:
        try:
            qdrant = self.qdrant
            model = self.embedding_model
            user_id = __user__.get("id", "unknown")

            points = []
            for assoc in associations:
                combined_text = f"{assoc['concept1']} {assoc['concept2']}"
                embedding = model.encode(combined_text)

                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "concept1": assoc["concept1"],
                        "concept2": assoc["concept2"],
                        "strength": assoc["strength"],
                        "context": assoc["context"],
                        "timestamp": assoc["timestamp"],
                        "user_id": user_id
                    }
                )
                points.append(point)

            if points:
                run_qdrant_operation(
                    lambda: qdrant.upsert(
                        collection_name=self.valves.collection_name,
                        points=points
                    ),
                    self._log,
                    description="Store associative memories"
                )

        except Exception as exc:
            self._log(f"Error storing associations: {exc}", "ERROR")
            if self.valves.debug_logging:
                self._log(traceback.format_exc(), "ERROR")

    def inlet(self, body: dict, __user__: dict) -> dict:
        if not self.valves.enabled:
            return body
        try:
            if not self._should_trigger():
                return body
            self._log("Associative memory filter triggered", "DEBUG")
            self._ensure_collection()
            messages = body.get("messages", [])
            if not messages:
                return body
            current_message = messages[-1].get("content", "") if messages else ""
            user_id = __user__.get("id", "unknown")
            associations = self._retrieve_associations(current_message, user_id)
            if associations:
                context = self._format_associations(associations)
                append_system_context(body["messages"], context)
                self._log(f"Injected {len(associations)} associations", "DEBUG")
            return body
        except Exception as exc:
            self._log(f"Error in inlet: {exc}", "ERROR")
            if self.valves.debug_logging:
                self._log(traceback.format_exc(), "ERROR")
            return body

    def outlet(self, body: dict, __user__: dict) -> dict:
        if not self.valves.enabled:
            return body
        try:
            self._ensure_collection()
            messages = body.get("messages", [])
            if not messages:
                return body
            associations = self._extract_associations(messages)
            if not associations:
                return body
            self._store_associations(body, __user__, associations)
            self._log(f"Stored {len(associations)} associations", "DEBUG")
            return body
        except Exception as exc:
            self._log(f"Error in outlet: {exc}", "ERROR")
            if self.valves.debug_logging:
                self._log(traceback.format_exc(), "ERROR")
            return body