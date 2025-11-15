import ollama
print("bge-m3 dims:", len(ollama.embeddings(model="bge-m3", prompt="Zorunlu staj için gerekli form nedir?")["embedding"]))
print("nomic dims:", len(ollama.embeddings(model="nomic-embed-text", prompt="What is the mandatory internship form?")["embedding"]))
