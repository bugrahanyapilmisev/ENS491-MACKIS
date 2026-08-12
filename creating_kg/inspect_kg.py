import chromadb
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

c = chromadb.PersistentClient(path=r'C:\bitirme3\ENS491-MACKIS\creating_database\chroma_db_v2')
coll = c.get_collection('mysu_v3_qwen3')
print(f'Collection: {coll.name}, count: {coll.count()}')

# Try a basic query to see if IDs work
peek = coll.peek(limit=3)
ids = peek['ids']
print(f'Peek IDs: {ids}')

# Try fetching by ID (this is what fails)
try:
    result = coll.get(ids=ids[:1], include=['embeddings'])
    print(f'Get by ID succeeded: got {len(result["ids"])} results')
    if result['embeddings']:
        print(f'Embedding dim: {len(result["embeddings"][0])}')
except Exception as e:
    print(f'Get by ID FAILED: {e}')

# Try a vector query
try:
    import numpy as np
    dummy_vec = np.zeros(4096).tolist()
    result = coll.query(query_embeddings=[dummy_vec], n_results=3, include=['documents','distances'])
    print(f'Query succeeded: {len(result["ids"][0])} results')
except Exception as e:
    print(f'Query FAILED: {e}')
