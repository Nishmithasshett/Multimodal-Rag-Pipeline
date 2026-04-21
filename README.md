# Multi-Modal RAG Pipeline with Milvus Lite

A complete implementation of a Retrieval-Augmented Generation (RAG) pipeline.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run test
python test_pipeline.py

# Try example
python example_usage.py
```

## Features

- ✅ Multi-modal inputs (Documents, Text, Speech, Images)
- ✅ Vector search with Milvus Lite
- ✅ Intent classification
- ✅ Web search integration
- ✅ Citation-based output

## Usage

```python
from multimodal_rag_pipeline import MultiModalRAGPipeline, QueryInput, InputType

# Initialize
rag = MultiModalRAGPipeline()

# Ingest documents
rag.ingest_documents(["doc1.pdf", "doc2.txt"])

# Query
query = QueryInput(text="Your question?", input_type=InputType.TEXT)
result = rag.query(query, top_k=5)
rag.display_output(result)
```

## Project Structure

```
multimodal-rag-pipeline/
├── multimodal_rag_pipeline.py  # Main pipeline
├── example_usage.py            # Examples
├── test_pipeline.py            # Tests
├── requirements.txt            # Dependencies
├── README.md                   # Documentation
├── data/
│   ├── documents/              # Input documents
│   ├── uploads/                # User uploads
│   └── outputs/                # Generated outputs
├── logs/                       # Log files
└── configs/                    # Configuration files
```

## License

MIT License
