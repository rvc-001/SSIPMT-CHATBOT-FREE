import json
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# --- Configuration ---
MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
KNOWLEDGE_FILE = 'knowledge_base.json'
INDEX_FILE = 'vector_index.faiss'
META_FILE = 'metadata.pkl'

def build_index():
    print("⏳ Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)
    
    print(f"📖 Reading {KNOWLEDGE_FILE}...")
    with open(KNOWLEDGE_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    documents_to_embed = []
    metadata = []
    
    print(f"⚙️  Processing {len(data)} documents...")
    
    for item in data:
        # STRATEGY: KEYWORD STUFFING
        # We combine Topic + Keywords + Text.
        # This allows single-word queries like "Placement" to match the 'keywords' section powerfully.
        rich_text = f"{item['topic']} {item['keywords']} {item['english_text']}"
        
        documents_to_embed.append(rich_text)
        metadata.append(item) 

    print("⚡ Generating vectors...")
    embeddings = model.encode(documents_to_embed, convert_to_numpy=True, normalize_embeddings=True)
    
    # Create Index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    print("💾 Saving index and metadata...")
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, 'wb') as f:
        pickle.dump(metadata, f)
        
    print("✅ Build complete! The AI is now keyword-aware.")

if __name__ == "__main__":
    build_index()