# Open WebUI Episodic Memory Filter

Give your Open WebUI assistants persistent, automatic memory of past conversations. This repository contains a single filter: `filters/episodic.py`.

## What This Is

An episodic memory layer that:
- stores user and assistant messages as embeddings in Qdrant (embedded mode, no Docker required),
- retrieves relevant prior exchanges before each response,
- injects a compact JSON memory block into the system context.

## How It Works

- Embedding model: `mixedbread-ai/mxbai-embed-large-v1` (1024 dims)
- Similarity: cosine distance in Qdrant
- Retrieval: hybrid queries (user, assistant, pair), merged and globally ranked
- Storage: per-user isolation with conversation IDs
- Mode: embedded by default (file-based), auto-detects Qdrant server if available

## Quick Start

### Prerequisites

- Open WebUI (https://github.com/open-webui/open-webui)
- Python 3.9+
- (Optional) Qdrant server for web dashboard (https://qdrant.tech)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/dotjax/openwebui-episodic-memory-filter
   cd openwebui-episodic-memory-filter
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Add the filter to Open WebUI:
   - Admin Panel -> Settings -> Filters
   - "+ Add Filter"
   - Paste `filter/episodic.py` into the editor
   - Save and enable the filter

### Optional: Run Qdrant Server

The filter works out of the box with embedded mode. To use the Qdrant web dashboard:

```bash
docker run -d -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage:z \
  qdrant/qdrant
```

The filter will automatically detect and connect to the server. Access the dashboard at http://localhost:6333/dashboard

### Optional: pre-download the embedding model

The filter will download the model on first use. To pre-download:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1')"
```

## Configuration (Valves)

These are exposed in the Open WebUI filter settings:

| Valve | Default | Description |
|-------|---------|-------------|
| qdrant_host | localhost | Qdrant server host (if server is running) |
| qdrant_port | 6333 | Qdrant server port (if server is running) |
| storage_path | ./qdrant_storage | Local storage path for embedded mode |
| collection_name | episodic | Qdrant collection name |
| embedding_model | mixedbread-ai/mxbai-embed-large-v1 | SentenceTransformer model name or path |
| embedding_device | cpu | Embedding device (cpu/cuda) |
| top_k | 30 | Max memories to return after global ranking |
| similarity_threshold | 0.4 | Minimum similarity score |
| user_display_name | USER | Display label for human messages |
| ai_display_name | ASSISTANT | Display label for assistant messages |
| enabled | true | Enable episodic memory |
| inject_memories | true | Inject memories into context |
| debug_logging | true | Verbose logging |

## Memory Injection Format

The filter injects a JSON array into the system prompt:

```json
[
  {
    "memory_id": "ep_a1b2c3d4",
    "collection": "episodic",
    "timestamp": "2025-11-04T20:30:00+00:00",
    "content": {
      "narrative": "...",
      "role": "user",
      "speaker": "USER",
      "participants": [
        "ASSISTANT",
        "USER"
      ],
      "relevance_score": 0.78
    }
  }
]
```

## Contributing

Issues and pull requests are welcome.

## License

GPL-3.0
