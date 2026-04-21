"""Quick Test Script - Validate RAG Pipeline Installation"""

import sys


def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import pymilvus
        print("  ✓ pymilvus")
        from sentence_transformers import SentenceTransformer
        print("  ✓ sentence-transformers")
        import langchain
        print("  ✓ langchain")
        import numpy as np
        print("  ✓ numpy")
        from PIL import Image
        print("  ✓ pillow")
        return True
    except ImportError as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    print("="*60)
    print("RAG PIPELINE VALIDATION TEST")
    print("="*60 + "\n")
    
    if not test_imports():
        print("\n❌ Import test failed!")
        print("Please run: pip install -r requirements.txt")
        sys.exit(1)
    
    print("\n✅ ALL TESTS PASSED!")
    print("Run 'python example_usage.py' to see it in action.")


if __name__ == "__main__":
    main()
