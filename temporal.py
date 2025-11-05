from __future__ import annotations
import time
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, MutableSequence, Optional, TypeVar
import numpy as np
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter as QdrantFilter, MatchValue, OrderBy, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

MessageList = MutableSequence[Dict[str, str]]
T = TypeVar("T")
_MODEL_CACHE: Dict[str, SentenceTransformer] = {}
_QDRANT_CACHE: Dict[str, QdrantClient] = {}
MODEL_VECTOR_SIZE = 1024
def _generate_memory_id(collection: str, existing_id: Optional[str] = None) -> str:
    """Generate a unique memory ID with collection prefix."""
    if existing_id:
        return f"{collection[:2]}_{existing_id[:8]}"
    return f"{collection[:2]}_{uuid.uuid4().hex[:8]}"
def _format_memory_json(
    collection: str,
    content: Dict[str, Any],
    timestamp: Optional[str] = None,
    existing_id: Optional[str] = None
) -> str:
    """Format a single memory as JSON string."""
    import json
    memory_id = _generate_memory_id(collection, existing_id)
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()
    memory_obj = {
        "memory_id": memory_id,
        "collection": collection,
        "timestamp": timestamp,
        "content": content
    }
    return json.dumps(memory_obj, indent=2)
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
def prepend_system_context(messages: MessageList, context: str) -> None:
    if not context:
        return
    if not messages:
        messages.insert(0, {"role": "system", "content": context})
        return
    first_message = messages[0]
    if first_message.get("role") == "system":
        first_message["content"] = f"{context}{first_message.get('content', '')}"
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
            default="temporal",
            description="Qdrant collection name for temporal memories"
        )
        episodic_collection: str = Field(
            default="episodic",
            description="Episodic collection name"
        )
        emotional_collection: str = Field(
            default="emotional",
            description="Emotional collection name"
        )
        semantic_collection: str = Field(
            default="semantic",
            description="Semantic collection name"
        )
        embedding_model: str = Field(
            default="/home/ubuntu/dev/models/embedding/mxbai-embed-large-v1",
            description="Sentence transformer model for embeddings"
        )
        embedding_device: str = Field(
            default="cpu",
            description="Device for embedding model (cpu/cuda)"
        )
        rhythm_window: int = Field(
            default=5,
            description="Number of messages to analyze for rhythm"
        )
        momentum_threshold: float = Field(
            default=0.7,
            description="Threshold for detecting building momentum"
        )
        turning_point_threshold: float = Field(
            default=0.8,
            description="Threshold for detecting turning points"
        )
        top_k_temporal: int = Field(
            default=5,
            description="Number of temporal memories to retrieve"
        )
        similarity_threshold: float = Field(
            default=0.4,
            description="Minimum similarity score for retrieval (0.0-1.0)"
        )
        temporal_decay_rate: float = Field(
            default=0.005,
            description="Base exponential decay rate (λ)"
        )
        significance_decay_modifier: float = Field(
            default=0.5,
            description="How much significance slows decay (0.0-1.0)"
        )
        recency_weight_factor: float = Field(
            default=0.4,
            description="Weight factor for recency"
        )
        relevance_weight_factor: float = Field(
            default=0.4,
            description="Weight factor for relevance"
        )
        significance_weight_factor: float = Field(
            default=0.2,
            description="Weight factor for significance"
        )
        max_temporal_memories: int = Field(
            default=10,
            description="Maximum temporal memories to retrieve"
        )
        min_memory_weight: float = Field(
            default=0.3,
            description="Minimum weight for memory inclusion"
        )
        enabled: bool = Field(
            default=True,
            description="Enable/disable temporal memory system"
        )
        track_rhythm: bool = Field(
            default=True,
            description="Track conversation rhythm"
        )
        track_emotional_evolution: bool = Field(
            default=True,
            description="Track emotional evolution"
        )
        track_themes: bool = Field(
            default=True,
            description="Track thematic progression"
        )
        inject_narrative_context: bool = Field(
            default=True,
            description="Inject narrative context into conversations"
        )
        debug_logging: bool = Field(
            default=True,
            description="Enable detailed debug logging"
        )
    
    def __init__(self) -> None:
        self.valves = self.Valves()
        self._collection_initialized = False
        self._session_state = {}
    def _log(self, message: str, level: str = "INFO"):
        if level == "DEBUG" and not self.valves.debug_logging:
            return
        print(f"[Temporal Memory] {level}: {message}")
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
    def _ensure_collection(self) -> None:
        if self._collection_initialized:
            return
        try:
            collections = run_qdrant_operation(
                self.qdrant.get_collections,
                self._log,
                description="Qdrant get_collections",
            ).collections
            collection_exists = any(c.name == self.valves.collection_name for c in collections)
            if not collection_exists:
                self._log(f"Creating collection: {self.valves.collection_name}")
                run_qdrant_operation(
                    lambda: self.qdrant.create_collection(
                        collection_name=self.valves.collection_name,
                        vectors_config=VectorParams(
                            size=MODEL_VECTOR_SIZE,
                            distance=Distance.COSINE
                        )
                    ),
                    self._log,
                    description="Qdrant create temporal collection",
                )
                run_qdrant_operation(
                    lambda: self.qdrant.create_payload_index(
                        collection_name=self.valves.collection_name,
                        field_name="timestamp",
                        field_schema="datetime"
                    ),
                    self._log,
                    description="Qdrant create timestamp range index",
                )
                self._log(f"Collection '{self.valves.collection_name}' created successfully")
            else:
                self._log(f"Collection '{self.valves.collection_name}' already exists", "DEBUG")
            self._collection_initialized = True
        except Exception as exc:
            self._log(f"Error ensuring collection: {exc}", "ERROR")
            raise
    def _calculate_time_decay(self, timestamp: str, significance: float) -> float:
        try:
            memory_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            now = datetime.now(timezone.utc)
            age_hours = (now - memory_time).total_seconds() / 3600
            self._log(f"Time decay calc: age_hours={age_hours:.2f}, significance={significance:.3f}", "DEBUG")
            base_alpha = self.valves.temporal_decay_rate * 100
            adaptive_alpha = base_alpha * (1.0 - significance * self.valves.significance_decay_modifier)
            decay_weight = 1.0 / (1.0 + adaptive_alpha * age_hours)
            self._log(f"Time decay: base_alpha={base_alpha:.3f}, adaptive_alpha={adaptive_alpha:.3f}, decay_weight={decay_weight:.3f}", "DEBUG")
            return decay_weight
        except Exception as exc:
            self._log(f"Error in time decay calculation: {exc}", "ERROR")
            return 0.5
    def _calculate_significance_score(self, memory_payload: Dict[str, Any]) -> float:
        momentum = float(memory_payload.get("momentum", 0.5))
        turning_points = np.asarray(memory_payload.get("turning_points", []), dtype=object)
        emotional_arc = np.asarray(memory_payload.get("emotional_arc", []), dtype=object)
        episodic_refs = np.asarray(memory_payload.get("episodic_refs", []), dtype=object)
        self._log(
            f"Significance calc - momentum: {momentum:.3f}, turning_points: {turning_points.size}, emotional_arc: {emotional_arc.size}, episodic_refs: {episodic_refs.size}",
            "DEBUG",
        )
        momentum_score = momentum * 0.3
        turning_point_score = min(turning_points.size, 5) / 5 * 0.3
        if emotional_arc.size:
            emotional_confidences = np.array(
                [float(entry.get("confidence", 0.0)) for entry in emotional_arc if isinstance(entry, dict)],
                dtype=float,
            )
            avg_confidence = float(emotional_confidences.mean()) if emotional_confidences.size else 0.0
            emotional_intensity = avg_confidence * 0.2
            self._log(
                f"Emotional intensity: avg_confidence={avg_confidence:.3f} -> emotional_intensity={emotional_intensity:.3f}",
                "DEBUG",
            )
        else:
            emotional_intensity = 0.0
            self._log("No emotional_arc data available", "DEBUG")
        length_score = min(episodic_refs.size, 20) / 20 * 0.2
        significance = momentum_score + turning_point_score + emotional_intensity + length_score
        final_significance = min(significance, 1.0)
        self._log(
            f"Significance breakdown: momentum={momentum_score:.3f}, turning={turning_point_score:.3f}, emotional={emotional_intensity:.3f}, length={length_score:.3f} -> final={final_significance:.3f}",
            "DEBUG",
        )
        return final_significance
    def _calculate_memory_weight(
        self, 
        memory: Dict[str, Any], 
        relevance_score: float,
        query_context: str
    ) -> float:
        timestamp = memory.get("timestamp", "")
        significance = self._calculate_significance_score(memory)
        recency_weight = self._calculate_time_decay(timestamp, significance)
        relevance_weight = relevance_score
        significance_weight = significance
        self._log(f"Memory weight components: recency={recency_weight:.3f}, relevance={relevance_weight:.3f}, significance={significance_weight:.3f}", "DEBUG")
        recency_factor = self.valves.recency_weight_factor
        relevance_factor = self.valves.relevance_weight_factor
        significance_factor = self.valves.significance_weight_factor
        temporal_keywords = ["yesterday", "last time", "previously", "before", "earlier", "ago"]
        temporal_boost = any(kw in query_context.lower() for kw in temporal_keywords)
        if temporal_boost:
            recency_factor += 0.2
            relevance_factor -= 0.1
            self._log(f"Temporal keywords detected, adjusting factors", "DEBUG")
        total = recency_factor + relevance_factor + significance_factor
        recency_factor /= total
        relevance_factor /= total
        significance_factor /= total
        final_weight = (
            recency_weight * recency_factor +
            relevance_weight * relevance_factor +
            significance_weight * significance_factor
        )
        self._log(f"Final weight calculation: factors=({recency_factor:.3f}, {relevance_factor:.3f}, {significance_factor:.3f}) -> weight={final_weight:.3f}", "DEBUG")
        return final_weight
    def _retrieve_weighted_temporal_memories(
        self, 
        current_context: str, 
        user_id: str
    ) -> List[Dict[str, Any]]:
        self._log(f"Retrieving temporal memories for user_id={user_id}", "DEBUG")
        context_embedding = self.embedding_model.encode(current_context)
        results = run_qdrant_operation(
            lambda: self.qdrant.search(
                collection_name=self.valves.collection_name,
                query_vector=context_embedding,
                query_filter=QdrantFilter(
                    must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
                ),
                limit=self.valves.top_k_temporal * 2,
                score_threshold=self.valves.similarity_threshold,
                with_payload=True
            ),
            self._log,
            description="Qdrant weighted temporal search"
        )
        self._log(f"Retrieved {len(results)} temporal memories from Qdrant", "DEBUG")
        weighted_memories = []
        for i, result in enumerate(results):
            memory = result.payload
            relevance_score = result.score
            weight = self._calculate_memory_weight(memory, relevance_score, current_context)
            weighted_memories.append({
                "memory": memory,
                "weight": weight,
                "relevance_score": relevance_score,
                "recency_weight": self._calculate_time_decay(
                    memory.get("timestamp", ""),
                    self._calculate_significance_score(memory)
                ),
                "significance_weight": self._calculate_significance_score(memory)
            })
            self._log(f"Memory {i+1}: weight={weight:.3f}, relevance={relevance_score:.3f}, timestamp={memory.get('timestamp', 'N/A')[:10]}", "DEBUG")
        weighted_memories.sort(key=lambda x: x["weight"], reverse=True)
        filtered = [m for m in weighted_memories if m["weight"] >= self.valves.min_memory_weight]
        self._log(f"Quality filtering: {len(weighted_memories)} -> {len(filtered)} (threshold={self.valves.min_memory_weight})", "DEBUG")
        final_result = filtered[:self.valves.max_temporal_memories]
        self._log(f"Final result: {len(final_result)} memories selected", "DEBUG")
        return final_result
    def _format_temporal_memories(self, weighted_memories: List[Dict[str, Any]]) -> str:
        """Format temporal memories as JSON according to Kaelan's cognitive architecture."""
        if not weighted_memories:
            return "[]"
        json_memories = []
        for item in weighted_memories:
            memory = item["memory"]
            weight = item["weight"]
            narrative_summary = memory.get("narrative_summary", "")
            timestamp = memory.get("timestamp", "")
            rhythm = memory.get("conversation_rhythm", "steady")
            momentum = memory.get("momentum", 0.5)
            emotional_arc = memory.get("emotional_arc", [])
            relevance_score = item.get("relevance_score", 0.0)
            significance_weight = item.get("significance_weight", 0.0)
            # Create structured content
            memory_content = {
                "narrative_summary": narrative_summary,
                "conversation_rhythm": rhythm,
                "momentum": momentum,
                "emotional_arc": emotional_arc[:3] if emotional_arc else [],  # Limit to recent emotions
                "relevance_score": relevance_score,
                "significance_weight": significance_weight,
                "temporal_weight": weight,
                "time_relative": self._format_relative_time(timestamp)
            }
            
            json_memories.append({
                "collection": "temporal",
                "content": memory_content,
                "timestamp": timestamp,
                "existing_id": None  # Temporal memories don't have stored IDs
            })
        return _format_memories_json(json_memories)
    def _format_relative_time(self, timestamp: str) -> str:
        memory_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)
        delta = now - memory_time
        if delta.days == 0:
            hours = delta.seconds // 3600
            return "Earlier today" if hours == 0 else f"{hours} hours ago"
        elif delta.days == 1:
            return "Yesterday"
        elif delta.days < 7:
            return f"{delta.days} days ago"
        elif delta.days < 30:
            weeks = delta.days // 7
            return f"{weeks} week{'s' if weeks > 1 else ''} ago"
        else:
            months = delta.days // 30
            return f"{months} month{'s' if months > 1 else ''} ago"
    
    def _analyze_rhythm(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        if not self.valves.track_rhythm or len(messages) < 2:
            return {"rhythm": "steady", "momentum": 0.5, "pacing": "moderate"}
        recent_messages = messages[-self.valves.rhythm_window:]
        self._log(f"Analyzing rhythm for {len(recent_messages)} recent messages", "DEBUG")
        contents = np.array([msg.get("content", "") for msg in recent_messages], dtype=object)
        lengths = np.vectorize(len)(contents).astype(float)
        pacing = "moderate"
        if lengths.size >= 3:
            avg_length = float(lengths.mean())
            length_variance = float(lengths.var())
            if avg_length < 100 and length_variance < 1000:
                pacing = "rapid"
            elif avg_length > 300 and length_variance > 5000:
                pacing = "contemplative"
            self._log(f"Pacing analysis: avg_length={avg_length:.1f}, variance={length_variance:.1f} -> {pacing}", "DEBUG")
        momentum = 0.5
        if lengths.size >= 3:
            mid_point = lengths.size // 2
            first_half = lengths[:mid_point] if mid_point else lengths
            second_half = lengths[mid_point:]

            first_half_avg = float(first_half.mean()) if first_half.size else 0.0
            second_half_avg = float(second_half.mean()) if second_half.size else 0.0
            growth_rate = (second_half_avg - first_half_avg) / first_half_avg if first_half_avg else 0.0
            exchange_density = lengths.size / max(1, self.valves.rhythm_window)
            roles = np.array([msg.get("role", "") for msg in recent_messages], dtype=object)
            alternations = np.count_nonzero(roles[1:] != roles[:-1]) if roles.size > 1 else 0
            alternation_rate = alternations / max(1, roles.size - 1)
            momentum_score = (growth_rate * 0.4) + (exchange_density * 0.3) + (alternation_rate * 0.3)
            momentum = max(0.0, min(1.0, 0.5 + momentum_score))
            self._log(
                f"Momentum factors: growth={growth_rate:.3f}, density={exchange_density:.3f}, alternation={alternation_rate:.3f} -> momentum={momentum:.3f}",
                "DEBUG",
            )
        if momentum >= self.valves.momentum_threshold:
            rhythm = "building"
        elif momentum <= 0.4:
            rhythm = "winding_down"
        else:
            rhythm = "steady"
        result = {
            "rhythm": rhythm,
            "momentum": round(momentum, 3),
            "pacing": pacing,
            "message_count": len(recent_messages),
        }
        self._log(f"Rhythm analysis result: {result}", "DEBUG")
        return result
    def _query_emotional_evolution(self, session_id: str) -> List[Dict[str, Any]]:
        if not self.valves.track_emotional_evolution:
            self._log("Emotional evolution tracking disabled", "DEBUG")
            return []
        try:
            self._log(f"Querying emotional collection: {self.valves.emotional_collection}", "DEBUG")
            results = run_qdrant_operation(
                lambda: self.qdrant.scroll(
                    collection_name=self.valves.emotional_collection,
                    limit=10,
                    order_by=OrderBy(key="timestamp", direction="desc"),
                    with_payload=True
                ),
                self._log,
                description="Qdrant emotional evolution query",
            )
            emotional_arc = []
            if results and results[0]:
                for result in results[0]:
                    payload = result.payload
                    emotional_arc.append({
                        "emotion": payload.get("top_emotion", "neutral"),
                        "confidence": payload.get("confidence", 0.0),
                        "timestamp": payload.get("timestamp", ""),
                        "content": payload.get("content", "")[:100] + "..." if len(payload.get("content", "")) > 100 else payload.get("content", "")
                    })
                self._log(f"Retrieved {len(emotional_arc)} emotional states from Qdrant", "DEBUG")
            else:
                self._log("No emotional data found in Qdrant", "DEBUG")
            return emotional_arc
        except Exception as exc:
            self._log(f"Error querying emotional evolution: {exc}", "ERROR")
            return []
    def _query_episodic_timestamps(self, session_id: str) -> List[Dict[str, Any]]:
        try:
            self._log(f"Querying episodic collection: {self.valves.episodic_collection}", "DEBUG")
            results = run_qdrant_operation(
                lambda: self.qdrant.scroll(
                    collection_name=self.valves.episodic_collection,
                    limit=10,
                    order_by=OrderBy(key="timestamp", direction="desc"),
                    with_payload=True
                ),
                self._log,
                description="Qdrant episodic timestamps query",
            )
            episodic_data = []
            if results and results[0]:
                for result in results[0]:
                    payload = result.payload
                    episodic_data.append({
                        "timestamp": payload.get("timestamp", ""),
                        "role": payload.get("role", ""),
                        "content": payload.get("content", "")[:100] + "..." if len(payload.get("content", "")) > 100 else payload.get("content", ""),
                        "conversation_id": payload.get("conversation_id", "")
                    })
                self._log(f"Retrieved {len(episodic_data)} episodic messages from Qdrant", "DEBUG")
            else:
                self._log("No episodic data found in Qdrant", "DEBUG")
            return episodic_data
        except Exception as exc:
            self._log(f"Error querying episodic timestamps: {exc}", "ERROR")
            return []
    def _detect_turning_points(self, messages: List[Dict[str, str]], emotions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        turning_points = []
        if len(messages) < 3:
            return turning_points
        for i, message in enumerate(messages[-5:]):
            content = message.get("content", "")
            turning_indicators = [
                "aha", "eureka", "wait", "actually", "but", "however", 
                "interesting", "fascinating", "wow", "amazing"
            ]
            if any(indicator in content.lower() for indicator in turning_indicators):
                turning_points.append({
                    "position": len(messages) - 5 + i,
                    "type": "insight",
                    "description": f"Potential insight at position {len(messages) - 5 + i}",
                    "content": content[:100] + "..." if len(content) > 100 else content,
                    "significance": 0.8
                })
        return turning_points
    def _format_narrative_context(
        self, 
        rhythm_data: Dict[str, Any], 
        emotional_arc: List[Dict[str, Any]], 
        turning_points: List[Dict[str, Any]]
    ) -> str:
        """Format current narrative context as JSON according to Kaelan's cognitive architecture."""
        if not self.valves.inject_narrative_context:
            return "{}"
        rhythm = rhythm_data.get("rhythm", "steady")
        momentum = rhythm_data.get("momentum", 0.5)
        pacing = rhythm_data.get("pacing", "moderate")
        # Create structured emotional arc
        structured_emotional_arc = []
        if emotional_arc and self.valves.track_emotional_evolution:
            for emotion in emotional_arc[:3]:
                structured_emotional_arc.append({
                    "emotion": emotion.get("emotion", ""),
                    "confidence": emotion.get("confidence", 0.0),
                    "timestamp": emotion.get("timestamp", "")
                })
        # Create structured turning points
        structured_turning_points = []
        if turning_points:
            for point in turning_points[:2]:
                structured_turning_points.append({
                    "type": point.get("type", ""),
                    "description": point.get("description", ""),
                    "significance": point.get("significance", 0.0)
                })
        # Create structured content
        memory_content = {
            "conversation_rhythm": rhythm,
            "momentum": momentum,
            "pacing": pacing,
            "emotional_arc": structured_emotional_arc,
            "turning_points": structured_turning_points,
            "narrative_state": self._get_narrative_state_description(rhythm)
        }
        return _format_memory_json(
            collection="temporal",
            content=memory_content,
            existing_id=None
        )
    def _get_narrative_state_description(self, rhythm: str) -> str:
        """Get narrative state description based on rhythm."""
        if rhythm == "building":
            return "I'm building momentum, with ideas flowing rapidly and building on each other."
        elif rhythm == "winding_down":
            return "I'm winding down, becoming more contemplative and reflective."
        else:
            return "I'm steady, with a balanced exchange of ideas."
    def _store_temporal_memory(
        self, 
        session_id: str,
        user_id: str,                 
        rhythm_data: Dict[str, Any], 
        emotional_arc: List[Dict[str, Any]], 
        turning_points: List[Dict[str, Any]], 
        episodic_refs: List[str]
    ) -> None:
        try:
            narrative_summary = f"Rhythm: {rhythm_data.get('rhythm', 'steady')}, "
            narrative_summary += f"Momentum: {rhythm_data.get('momentum', 0.5):.2f}, "
            narrative_summary += f"Pacing: {rhythm_data.get('pacing', 'moderate')}"
            if emotional_arc:
                recent_emotions = [e['emotion'] for e in emotional_arc[:3]]
                narrative_summary += f", Emotional arc: {' → '.join(recent_emotions)}"
            embedding = self.embedding_model.encode(narrative_summary)
            point_id = str(uuid.uuid4())
            timestamp = datetime.now(timezone.utc).isoformat()
            payload = {
                "session_id": session_id,
                "user_id": user_id,                       
                "conversation_rhythm": rhythm_data.get("rhythm", "steady"),
                "momentum": rhythm_data.get("momentum", 0.5),
                "pacing": rhythm_data.get("pacing", "moderate"),
                "emotional_arc": emotional_arc,
                "turning_points": turning_points,
                "narrative_summary": narrative_summary,
                "episodic_refs": episodic_refs,
                "timestamp": timestamp
            }
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            )
            run_qdrant_operation(
                lambda: self.qdrant.upsert(
                    collection_name=self.valves.collection_name,
                    points=[point]
                ),
                self._log,
                description="Qdrant upsert temporal memory",
            )
            self._log(f"Stored temporal memory: {point_id}", "DEBUG")
        except Exception as exc:
            self._log(f"Error storing temporal memory: {exc}", "ERROR")
            raise
    def inlet(self, body: dict, __user__: dict) -> dict:
        if not self.valves.enabled:
            self._log("Temporal memory system disabled via Valves", "DEBUG")
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
            session_id = str(uuid.uuid4())
            user_id = __user__.get("id", "unknown")
            self._ensure_collection()
            rhythm_data = self._analyze_rhythm(messages)
            self._log(f"Rhythm analysis: {rhythm_data}", "DEBUG")
            current_context = f"{user_message}\n{rhythm_data.get('rhythm', 'steady')}"
            weighted_memories = self._retrieve_weighted_temporal_memories(current_context, user_id)
            if weighted_memories:
                temporal_memory_context = self._format_temporal_memories(weighted_memories)
                prepend_system_context(body["messages"], temporal_memory_context)
                self._log(f"Injected {len(weighted_memories)} weighted temporal memories", "DEBUG")
            emotional_arc = self._query_emotional_evolution(session_id)
            episodic_data = self._query_episodic_timestamps(session_id)
            episodic_refs = [item.get("conversation_id", "") for item in episodic_data]
            turning_points = self._detect_turning_points(messages, emotional_arc)
            narrative_context = self._format_narrative_context(rhythm_data, emotional_arc, turning_points)
            if narrative_context:
                prepend_system_context(body["messages"], narrative_context)
                self._log("Prepended current narrative context to system message", "DEBUG")
            if "metadata" not in body:
                body["metadata"] = {}
            body["metadata"]["_temporal_session_id"] = session_id
            body["metadata"]["_temporal_rhythm_data"] = rhythm_data
            body["metadata"]["_temporal_emotional_arc"] = emotional_arc
            body["metadata"]["_temporal_turning_points"] = turning_points
            body["metadata"]["_temporal_episodic_refs"] = episodic_refs
            self._log("inlet() completed successfully", "DEBUG")
        except Exception as exc:
            self._log(f"Error in inlet(): {exc}", "ERROR")
            if self.valves.debug_logging:
                self._log(traceback.format_exc(), "ERROR")
        return body
    def outlet(self, body: dict, __user__: dict) -> dict:
        if not self.valves.enabled:
            self._log("Temporal memory system disabled via Valves", "DEBUG")
            return body
        try:
            messages = body.get("messages", [])
            if not messages:
                self._log("No messages in body", "DEBUG")
                return body
            last_message = messages[-1]
            if last_message.get("role") != "assistant":
                self._log("Last message is not from assistant", "DEBUG")
                return body
            assistant_message = last_message.get("content", "")
            if not assistant_message or not assistant_message.strip():
                self._log("Assistant message is empty", "DEBUG")
                return body
            self._log(f"Processing assistant message: {assistant_message[:100]}...", "DEBUG")
            metadata = body.get("metadata", {})
            session_id = metadata.get("_temporal_session_id", str(uuid.uuid4()))
            rhythm_data = metadata.get("_temporal_rhythm_data", {})
            emotional_arc = metadata.get("_temporal_emotional_arc", [])
            turning_points = metadata.get("_temporal_turning_points", [])
            episodic_refs = metadata.get("_temporal_episodic_refs", [])
            user_id = __user__.get("id", "unknown")
            self._ensure_collection()
            self._log("Storing temporal memory", "DEBUG")
            self._store_temporal_memory(
                session_id=session_id,
                user_id=user_id,                 
                rhythm_data=rhythm_data,
                emotional_arc=emotional_arc,
                turning_points=turning_points,
                episodic_refs=episodic_refs
            )
            self._log("outlet() completed successfully", "DEBUG")
        except Exception as exc:
            self._log(f"Error in outlet(): {exc}", "ERROR")
            if self.valves.debug_logging:
                self._log(traceback.format_exc(), "ERROR")
        return body