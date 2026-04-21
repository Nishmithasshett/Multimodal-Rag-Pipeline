from multimodal_rag_pipeline import MultiModalRAG

from dotenv import load_dotenv
load_dotenv()
def main():

    print("\n================================================================================")
    print("MULTI-MODAL RAG PIPELINE - EXAMPLE")
    print("================================================================================\n")

    rag = MultiModalRAG()

    print("\n================================================================================")
    print("✓ Pipeline initialized successfully!")
    print("================================================================================\n")

    print("📋 Configuration:")
    print("  • LLM: NVIDIA Llama 3.1 8B")
    print("  • Web Search: Simulated (set TAVILY_API_KEY)")
    print("  • Reranking: Disabled")
    print("  • Vector DB: Milvus Lite (example_rag)")
    print("  • Embedding: 384D\n")

    rag.ingest_documents([
        "data/documents/machine_learning.txt",
        "data/documents/deep_learning.txt",
        "data/documents/ai_ml_guide.txt",
        "data/documents/llm_overview.txt"
    ])

    answer, vector, web = rag.query("Generate architecture diagram of RAG pipeline")

    print("\n================================================================================")
    print("🤖 RAG ASSISTANT RESPONSE")
    print("================================================================================\n")

    print("💬 Answer:\n")
    print(answer)

    print("\n📚 Sources:\n")
    

    i = 1
    seen = set()
    for v in vector:
        if v["source"] not in seen:
         print(f"[{i}] 📄 {v['source']}")
         seen.add(v["source"])
         i += 1

    for w in web:
        print(f"[{i}] 🌐 {w['source']}")
        i += 1



if __name__ == "__main__":
    main()