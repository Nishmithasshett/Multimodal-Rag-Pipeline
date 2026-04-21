#!/bin/bash
# Setup script for the RAG pipeline

echo "Setting up Multi-Modal RAG Pipeline..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo "Run './run.sh' to start the example"
