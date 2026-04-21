"""
Enhanced Multi Modal RAG Pipeline
Supports:
- Vector Retrieval
- Tavily Web Search (or simulated)
- NVIDIA NIM LLM
- Citations
- Query Intent Detection
"""

import os
from pathlib import Path
from enum import Enum

from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from tavily import TavilyClient
from openai import OpenAI


# -------------------------------------------------
# QUERY INTENT
# -------------------------------------------------

class QueryIntent(Enum):

    INFORMATIONAL = "informational"
    FACT_LOOKUP = "fact_lookup"
    COMPARISON = "comparison"
    SUMMARIZATION = "summarization"
    GENERATION = "generation"


# -------------------------------------------------
# EMBEDDING MODEL
# -------------------------------------------------

class EmbeddingModel:

    def __init__(self):

        print("Loading embedding model: all-MiniLM-L6-v2...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, text):

        return self.model.encode(text).tolist()


# -------------------------------------------------
# VECTOR STORE
# -------------------------------------------------

class MilvusVectorStore:

    def __init__(self, collection_name, dim):

        connections.connect("default", uri="./milvus_lite.db")

        if not utility.has_collection(collection_name):

            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=200),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="metadata", dtype=DataType.JSON),
            ]

            schema = CollectionSchema(fields)

            self.collection = Collection(collection_name, schema)

            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128},
            }

            self.collection.create_index("embedding", index_params)

        else:

            self.collection = Collection(collection_name)

    def insert(self, chunks):

        ids = [c["id"] for c in chunks]
        embeddings = [c["embedding"] for c in chunks]
        texts = [c["text"] for c in chunks]
        metadata = [c["metadata"] for c in chunks]

        self.collection.insert([ids, embeddings, texts, metadata])
        self.collection.flush()

    def search(self, embedding, k=3):

        self.collection.load()

        results = self.collection.search(
            data=[embedding],
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=k,
            output_fields=["text", "metadata"],
        )

        docs = []

        for hits in results:
            for hit in hits:

                docs.append({
                    "text": hit.entity.get("text"),
                    "source": hit.entity.get("metadata")["source"]
                })

        return docs


# -------------------------------------------------
# WEB SEARCH
# -------------------------------------------------

class WebSearch:

    def __init__(self):

        key = os.getenv("TAVILY_API_KEY")

        if key:

            self.client = TavilyClient(api_key=key)
            self.enabled = True

        else:

            self.enabled = False
            print("⚠️ TAVILY_API_KEY not set. Using simulated web search")

    def search(self, query):

        if not self.enabled:

            results = []

            for i in range(3):
                results.append({
                    "title": f"Simulated result {i+1}",
                    "content": f"Simulated information about {query}",
                    "url": f"https://example.com/{i+1}",
                    "source": "Simulated Web Result"
                })

            return results

        response = self.client.search(query=query, max_results=3)

        results = []

        for r in response["results"]:
            results.append({
                "title": r["title"],
                "content": r["content"],
                "url": r["url"],
                "source": r["url"]
            })

        return results


# -------------------------------------------------
# OUTPUT GENERATOR
# -------------------------------------------------

class OutputGenerator:

    def __init__(self):

        key = os.getenv("NVIDIA_API_KEY")

        if key:

            self.client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=key
            )

            self.enabled = True

        else:

            self.enabled = False
            print("⚠️ NVIDIA_API_KEY missing")

    def generate(self, query, docs, web):

        sources = docs + web

        context = ""

        for i, s in enumerate(sources):

            if "text" in s:
                context += f"[{i+1}] {s['text']}\n"
            else:
                context += f"[{i+1}] {s['content']}\n"

        prompt = f"""
Answer the question using the sources below.

Sources:
{context}

Question:
{query}
"""

        if not self.enabled:
            return "⚠️ NVIDIA_API_KEY not set. Cannot generate answer."

        completion = self.client.chat.completions.create(
            model="meta/llama-3.1-8b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500
        )

        return completion.choices[0].message.content


# -------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------

class MultiModalRAG:

    def __init__(self):

        self.embedder = EmbeddingModel()

        self.vector_store = MilvusVectorStore(
            "example_rag",
            self.embedder.dim
        )

        self.web = WebSearch()

        self.generator = OutputGenerator()

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

    # -------------------------------------------------
    # QUERY INTENT DETECTION
    # -------------------------------------------------

    def detect_query_intent(self, query):

        q = query.lower()

        if "summarize" in q or "summary" in q:
            return QueryIntent.SUMMARIZATION

        if "compare" in q or "difference" in q:
            return QueryIntent.COMPARISON

        if "generate" in q or "write" in q:
            return QueryIntent.GENERATION

        if "what is" in q or "define" in q:
            return QueryIntent.FACT_LOOKUP

        return QueryIntent.INFORMATIONAL


    # -------------------------------------------------
    # INGEST DOCUMENTS
    # -------------------------------------------------

    def ingest_documents(self, files):

        chunks = []

        for f in files:

            loader = TextLoader(f)
            docs = loader.load()

            for d in docs:

                split_chunks = self.splitter.split_text(d.page_content)

                for i, c in enumerate(split_chunks):

                    emb = self.embedder.embed(c)

                    chunks.append({
                        "id": f"{Path(f).stem}_{i}",
                        "embedding": emb,
                        "text": c,
                        "metadata": {"source": f}
                    })

        self.vector_store.insert(chunks)

        print(f"✓ Inserted {len(chunks)} chunks")


    # -------------------------------------------------
    # QUERY
    # -------------------------------------------------

    def query(self, question):

        print("\n================================================================================")
        print("QUERY PROCESSING")
        print("================================================================================\n")

        intent = self.detect_query_intent(question)

        print(f"📝 Query: {question}")
        print(f"🎯 Query Intent: {intent.value}")
        print("📤 Output Type: text\n")

        print("🔍 Retrieving context...")

        emb = self.embedder.embed(question)

        vector_results = self.vector_store.search(emb, 4)

        print(f"   ├─ Vector DB: {len(vector_results)} chunks")

        web_results = self.web.search(question)

        print(f"   └─ Tavily: {len(web_results)} results\n")

        print("✍️  Generating text output with NVIDIA Llama 3.1 8B...\n")

        answer = self.generator.generate(question, vector_results, web_results)

        return answer, vector_results, web_results