from __future__ import annotations

import time
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, MutableSequence, Optional, Tuple, TypeVar

import numpy as np
import torch
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, PointStruct, VectorParams
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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

_MODEL_CACHE: Dict[str, AutoModelForSequenceClassification] = {}
_QDRANT_CACHE: Dict[str, QdrantClient] = {}
_TOKENIZER_CACHE: Dict[str, AutoTokenizer] = {}

MessageList = MutableSequence[Dict[str, str]]
T = TypeVar("T")

EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
]


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
            default="emotional",
            description="Qdrant collection name for emotional memories"
        )
        emotion_model: str = Field(
            default="/home/ubuntu/dev/models/embedding/roberta-base-go_emotions",
            description="HuggingFace model for emotion classification"
        )
        emotion_device: str = Field(
            default="cpu",
            description="Device for emotion model (cpu/cuda)"
        )
        top_k_emotional: int = Field(
            default=10,
            description="Number of emotional memories to retrieve"
        )
        confidence_threshold: float = Field(
            default=0.5,
            description="Confidence threshold for exact emotion matching (0.0-1.0)"
        )
        max_emotional_memories: int = Field(
            default=7,
            description="Maximum emotional memories to inject into context (top ~3 likely affectionate with boost)"
        )
        affection_boost: float = Field(
            default=1.3,
            description="Score multiplier for affectionate memories (1.0 = no boost, 1.5 = 50% boost)"
        )
        affectionate_emotions: str = Field(
            default="caring,love,desire,gratitude,admiration",
            description="Comma-separated emotions to boost during recall"
        )
        enabled: bool = Field(
            default=True,
            description="Enable/disable emotional context system"
        )
        inject_emotional_context: bool = Field(
            default=True,
            description="Inject emotional memories into context"
        )
        debug_logging: bool = Field(
            default=True,
            description="Enable detailed debug logging"
        )

    def __init__(self) -> None:
        self.valves = self.Valves()
        self._collection_initialized = False
        self._label_count: Optional[int] = len(EMOTION_LABELS)

    def _log(self, message: str, level: str = "INFO"):
        if level == "DEBUG" and not self.valves.debug_logging:
            return
        print(f"[Emotional Context] {level}: {message}")

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
    def tokenizer(self) -> Any:
        cache_key = f"{self.valves.emotion_model}_{self.valves.emotion_device}"
        if cache_key not in _TOKENIZER_CACHE:
            try:
                self._log(f"Loading tokenizer: {self.valves.emotion_model} (FIRST LOAD - caching in memory)", "INFO")
                _TOKENIZER_CACHE[cache_key] = AutoTokenizer.from_pretrained(self.valves.emotion_model, local_files_only=True)
                self._log("Tokenizer cached in memory permanently", "DEBUG")
            except Exception as exc:
                self._log(f"Failed to load tokenizer: {exc}", "ERROR")
                raise
        else:
            self._log("Using cached tokenizer", "DEBUG")
        return _TOKENIZER_CACHE[cache_key]

    @property
    def model(self) -> Any:
        cache_key = f"{self.valves.emotion_model}_{self.valves.emotion_device}"
        if cache_key not in _MODEL_CACHE:
            try:
                self._log(f"Loading emotion model: {self.valves.emotion_model} (FIRST LOAD - caching in memory)", "INFO")
                _MODEL_CACHE[cache_key] = AutoModelForSequenceClassification.from_pretrained(
                    self.valves.emotion_model,
                    use_safetensors=True,
                    local_files_only=True
                )
                _MODEL_CACHE[cache_key].to(self.valves.emotion_device)
                _MODEL_CACHE[cache_key].eval()
                num_labels = getattr(_MODEL_CACHE[cache_key].config, "num_labels", None)
                if isinstance(num_labels, int) and num_labels > 0:
                    self._label_count = num_labels
                self._log("Emotion model cached in memory permanently", "DEBUG")
            except Exception as exc:
                self._log(f"Failed to load emotion model: {exc}", "ERROR")
                raise
        else:
            self._log("Using cached emotion model", "DEBUG")
        return _MODEL_CACHE[cache_key]

    def _ensure_collection(self, force: bool = False) -> None:
        if self._collection_initialized and not force:
            return

        try:
            collections = self.qdrant.get_collections().collections
            collection_exists = any(c.name == self.valves.collection_name for c in collections)

            label_count = self._label_count
            if label_count is None:
                label_count = getattr(self.model.config, "num_labels", len(EMOTION_LABELS))
                self._label_count = label_count
            else:
                try:
                    label_count = getattr(self.model.config, "num_labels", label_count)
                    self._label_count = label_count
                except Exception as exc:
                    self._log(f"Could not update label_count from model config: {exc}", "DEBUG")

            def _apply_collection(operation: str) -> None:
                vectors_config = VectorParams(
                    size=28,
                    distance=Distance.COSINE
                )

                if operation == "create":
                    run_qdrant_operation(
                        lambda: self.qdrant.create_collection(
                            collection_name=self.valves.collection_name,
                            vectors_config=vectors_config,
                        ),
                        self._log,
                        description="Qdrant create emotional collection",
                    )
                else:
                    run_qdrant_operation(
                        lambda: self.qdrant.recreate_collection(
                            collection_name=self.valves.collection_name,
                            vectors_config=vectors_config,
                        ),
                        self._log,
                        description="Qdrant recreate emotional collection",
                    )

            if not collection_exists:
                self._log(f"Creating collection: {self.valves.collection_name}")
                _apply_collection("create")
            elif force:
                self._log(f"Recreating collection: {self.valves.collection_name}")
                _apply_collection("recreate")
            else:
                self._log(f"Collection '{self.valves.collection_name}' already exists", "DEBUG")

            if not collection_exists or force:
                run_qdrant_operation(
                    lambda: self.qdrant.create_payload_index(
                        collection_name=self.valves.collection_name,
                        field_name="top_emotion",
                        field_schema="keyword"
                    ),
                    self._log,
                    description="Qdrant create top_emotion index",
                )
                run_qdrant_operation(
                    lambda: self.qdrant.create_payload_index(
                        collection_name=self.valves.collection_name,
                        field_name="ref_id",
                        field_schema="keyword"
                    ),
                    self._log,
                    description="Qdrant create ref_id index",
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

                self._log(f"Collection '{self.valves.collection_name}' ready")

            self._collection_initialized = True

        except Exception as exc:
            self._log(f"Error ensuring collection: {exc}", "ERROR")
            raise

    def _analyze_emotion(self, text: str) -> Tuple[np.ndarray, str, float]:
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            inputs = {k: v.to(self.valves.emotion_device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0]
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            top_idx = int(probs.argmax())
            top_emotion = EMOTION_LABELS[top_idx]
            confidence = float(probs[top_idx])
            self._log(f"Emotion analysis: {top_emotion} (confidence: {confidence:.3f})", "DEBUG")
            return probs, top_emotion, confidence
        except Exception as exc:
            self._log(f"Error analyzing emotion: {exc}", "ERROR")
            label_count = self._label_count or len(EMOTION_LABELS)
            neutral_probs = np.zeros(label_count)
            neutral_probs[-1] = 1.0
            return neutral_probs, "neutral", 1.0

    def _retrieve_emotional_memories(
        self,
        emotion_probs: np.ndarray,
        top_emotion: str,
        confidence: float,
        _retry: bool = True,
    ) -> List[Dict[str, Any]]:

        try:
            if confidence >= self.valves.confidence_threshold:
                self._log(f"High confidence ({confidence:.3f}), using exact match for {top_emotion}", "DEBUG")
                from qdrant_client.models import Filter as QdrantFilter, FieldCondition, MatchValue
                results = run_qdrant_operation(
                    lambda: self.qdrant.search(
                        collection_name=self.valves.collection_name,
                        query_vector=emotion_probs,
                        query_filter=QdrantFilter(
                            must=[
                                FieldCondition(
                                    key="top_emotion",
                                    match=MatchValue(value=top_emotion)
                                )
                            ]
                        ),
                        limit=self.valves.top_k_emotional,
                        with_payload=True
                    ),
                    self._log,
                    description="Qdrant emotional exact search",
                )
            else:
                self._log(f"Low confidence ({confidence:.3f}), using fuzzy matching", "DEBUG")
                results = run_qdrant_operation(
                    lambda: self.qdrant.search(
                        collection_name=self.valves.collection_name,
                        query_vector=emotion_probs,
                        limit=self.valves.top_k_emotional,
                        with_payload=True
                    ),
                    self._log,
                    description="Qdrant emotional fuzzy search",
                )
            memories = []
            for result in results:
                memory = {
                    "ref_id": result.payload.get("ref_id"),
                    "content": result.payload.get("content"),
                    "role": result.payload.get("role"),
                    "top_emotion": result.payload.get("top_emotion"),
                    "timestamp": result.payload.get("timestamp"),
                    "emotion_score": float(result.score)
                }
                memories.append(memory)
            self._log(f"Retrieved {len(memories)} emotional memories", "DEBUG")
            memories = self._apply_affection_weighting(memories)
            return memories
        except Exception as exc:
            needs_recreate = (
                UnexpectedResponse is not None
                and isinstance(exc, UnexpectedResponse)
                and "Not existing vector name" in str(exc)
            )
            if needs_recreate and _retry:
                self._log(
                    "Qdrant returned missing vector name during search; forcing collection recreation",
                    "WARNING",
                )
                self._collection_initialized = False
                self._ensure_collection(force=True)
                return self._retrieve_emotional_memories(
                    emotion_probs,
                    top_emotion,
                    confidence,
                    _retry=False,
                )

            self._log(f"Error retrieving emotional memories: {exc}", "ERROR")
            return []

    def _apply_affection_weighting(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.valves.affection_boost <= 1.0:
            return memories
        
        affectionate_set = {
            emotion.strip().lower() 
            for emotion in self.valves.affectionate_emotions.split(',')
            if emotion.strip()
        }
        
        if not affectionate_set:
            return memories
        boosted_memories = []
        for memory in memories:
            memory_copy = memory.copy()
            top_emotion = memory.get("top_emotion", "").lower()
            if top_emotion in affectionate_set:
                original_score = memory_copy["emotion_score"]
                boosted_score = original_score * self.valves.affection_boost
                memory_copy["emotion_score"] = min(boosted_score, 1.0)
                memory_copy["affection_boosted"] = True
                memory_copy["original_score"] = original_score
                self._log(
                    f"Boosted {top_emotion} memory: {original_score:.3f} -> {boosted_score:.3f}",
                    "DEBUG"
                )
            boosted_memories.append(memory_copy)
        boosted_memories.sort(key=lambda m: m["emotion_score"], reverse=True)
        return boosted_memories

    def _format_kaelan_emotional_context(
        self,
        kaelan_emotion: str,
        kaelan_confidence: float,
        kaelan_memories: List[Dict[str, Any]],
        user_emotion: str,
        user_confidence: float
    ) -> str:
        """Format emotional context as JSON according to Kaelan's cognitive architecture."""
        # Create structured emotional memories
        similar_memories = []
        for memory in kaelan_memories[:self.valves.max_emotional_memories]:
            role_label = "Kaelan" if memory["role"] == "assistant" else "User"
            content = memory["content"]
            if len(content) > 200:
                content = content[:200] + "..."
            
            emotion_data = {
                "emotion": memory["top_emotion"],
                "confidence": memory["emotion_score"],
                "content": content,
                "role": role_label,
                "timestamp": memory["timestamp"][:10]
            }
            
            if memory.get("affection_boosted") and memory.get("original_score"):
                emotion_data["affection_boosted"] = True
                emotion_data["original_confidence"] = memory["original_score"]
            
            similar_memories.append(emotion_data)
        
        # Create structured content
        memory_content = {
            "kaelan_emotion": kaelan_emotion,
            "kaelan_confidence": kaelan_confidence,
            "user_emotion": user_emotion,
            "user_confidence": user_confidence,
            "similar_memories": similar_memories,
            "emotional_continuity": len(similar_memories) > 0
        }
        
        return _format_memory_json(
            collection="emotional",
            content=memory_content,
            existing_id=None
        )
    
    def _store_emotional_memory(
        self,
        content: str,
        role: str,
        conversation_id: str,
        episodic_point_id: Optional[str] = None,
        _retry: bool = True,
    ) -> None:
        try:
            probs, top_emotion, confidence = self._analyze_emotion(content)
            point_id = str(uuid.uuid4())
            timestamp = datetime.now(timezone.utc).isoformat()
            
            point = PointStruct(
                id=point_id,
                vector=probs,
                payload={
                    "ref_id": episodic_point_id or point_id,
                    "content": content,
                    "role": role,
                    "top_emotion": top_emotion,
                    "confidence": confidence,
                    "timestamp": timestamp,
                    "conversation_id": conversation_id
                }
            )
            
            run_qdrant_operation(
                lambda: self.qdrant.upsert(
                    collection_name=self.valves.collection_name,
                    points=[point]
                ),
                self._log,
                description="Qdrant upsert emotional memory",
            )
            
            self._log(f"Stored {role} emotional memory: {top_emotion} (confidence: {confidence:.3f})", "DEBUG")
            
        except Exception as exc:
            needs_recreate = (
                UnexpectedResponse is not None
                and isinstance(exc, UnexpectedResponse)
                and "Not existing vector name" in str(exc)
            )
            if needs_recreate and _retry:
                self._log(
                    "Qdrant returned missing vector name during upsert; forcing collection recreation",
                    "WARNING",
                )
                self._collection_initialized = False
                self._ensure_collection(force=True)
                return self._store_emotional_memory(
                    content,
                    role,
                    conversation_id,
                    episodic_point_id,
                    _retry=False,
                )

            self._log(f"Error storing emotional memory: {exc}", "ERROR")
            raise
    
    def inlet(self, body: dict, __user__: dict) -> dict:
        if not self.valves.enabled:
            self._log("Emotional context system disabled via Valves", "DEBUG")
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
            
            self._ensure_collection()
            
            from qdrant_client.models import Filter as QdrantFilter, FieldCondition, MatchValue, OrderBy
            
            kaelan_last_emotion = None
            kaelan_emotion_probs = None
            kaelan_confidence = 0.0
            
            try:
                recent_kaelan = self.qdrant.scroll(
                    collection_name=self.valves.collection_name,
                    scroll_filter=QdrantFilter(
                        must=[FieldCondition(key="role", match=MatchValue(value="assistant"))]
                    ),
                    limit=1,
                    order_by=OrderBy(key="timestamp", direction="desc"),
                    with_payload=True,
                    with_vectors=True
                )
                
                if recent_kaelan and recent_kaelan[0]:
                    last_point = recent_kaelan[0][0]
                    kaelan_last_emotion = last_point.payload.get("top_emotion", "neutral")
                    kaelan_confidence = last_point.payload.get("confidence", 0.0)
                    kaelan_emotion_probs = np.array(last_point.vector)
                    
                    self._log(f"Kaelan's last emotion: {kaelan_last_emotion} (confidence: {kaelan_confidence:.3f})", "DEBUG")
            except Exception as exc:
                self._log(f"Could not retrieve Kaelan's last emotion: {exc}", "DEBUG")
                label_count = self._label_count or len(EMOTION_LABELS)
                kaelan_emotion_probs = np.zeros(label_count)
                kaelan_emotion_probs[-1] = 1.0
                kaelan_last_emotion = "neutral"
                kaelan_confidence = 1.0
            
            emotional_memories = []
            if self.valves.inject_emotional_context and kaelan_emotion_probs is not None:
                self._log("Retrieving Kaelan's emotional memories for continuity", "DEBUG")
                emotional_memories = self._retrieve_emotional_memories(
                    kaelan_emotion_probs, 
                    kaelan_last_emotion, 
                    kaelan_confidence
                )
                self._log(f"Retrieved {len(emotional_memories)} of Kaelan's emotional memories", "DEBUG")
            
            user_probs, user_emotion, user_confidence = self._analyze_emotion(user_message)
            self._log(f"User's emotion: {user_emotion} (confidence: {user_confidence:.3f})", "DEBUG")
            
            conversation_id = body.get("metadata", {}).get("_memory_conversation_id", str(uuid.uuid4()))
            self._store_emotional_memory(
                content=user_message,
                role="user",
                conversation_id=conversation_id,
                episodic_point_id=None
            )
            
            if emotional_memories or kaelan_last_emotion:
                emotional_context = self._format_kaelan_emotional_context(
                    kaelan_last_emotion,
                    kaelan_confidence,
                    emotional_memories,
                    user_emotion,
                    user_confidence
                )
                
                append_system_context(messages, emotional_context)
                self._log("Injected Kaelan's emotional context into conversation", "DEBUG")
            
            if "metadata" not in body:
                body["metadata"] = {}
            body["metadata"]["_emotion_analyzed"] = True
            body["metadata"]["_kaelan_emotion"] = kaelan_last_emotion
            body["metadata"]["_kaelan_confidence"] = kaelan_confidence
            body["metadata"]["_user_emotion"] = user_emotion
            body["metadata"]["_user_confidence"] = user_confidence
            
            self._log("inlet() completed successfully", "DEBUG")
            
        except Exception as exc:
            self._log(f"Error in inlet(): {exc}", "ERROR")
            if self.valves.debug_logging:
                self._log(traceback.format_exc(), "ERROR")
        
        return body
    
    def outlet(self, body: dict, __user__: dict) -> dict:
        if not self.valves.enabled:
            self._log("Emotional context system disabled via Valves", "DEBUG")
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
            
            conversation_id = (
                body.get("metadata", {})
                .get("_memory_conversation_id", str(uuid.uuid4()))
            )
            
            episodic_point_id = (
                body.get("metadata", {})
                .get("_episodic_point_id")
            )
            
            self._ensure_collection()
            
            self._log("Storing assistant emotional memory", "DEBUG")
            self._store_emotional_memory(
                content=assistant_message,
                role="assistant",
                conversation_id=conversation_id,
                episodic_point_id=episodic_point_id
            )
            
            self._log("outlet() completed successfully", "DEBUG")
            
        except Exception as exc:
            self._log(f"Error in outlet(): {exc}", "ERROR")
            if self.valves.debug_logging:
                self._log(traceback.format_exc(), "ERROR")
        
        return body