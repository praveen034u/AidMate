# Make sure Ollama is running
ollama serve

# Pull the tiny CPU-friendly embedding model
ollama pull all-minilm:latest

# Pull the lightweight chat model you’re aliasing “gpt-oss” to
ollama pull phi3:mini

# Verify
ollama list