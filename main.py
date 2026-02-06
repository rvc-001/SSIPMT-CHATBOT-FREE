from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import faiss
import pickle
import re
from sentence_transformers import SentenceTransformer

# --- Configuration ---
MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
INDEX_FILE = 'vector_index.faiss'
META_FILE = 'metadata.pkl'
SIMILARITY_THRESHOLD = 0.30  # Lowered slightly for short queries

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
index = None
metadata = []

# --- EXTENDED VOCAB MAP ---
HINGLISH_VOCAB = {
    # Question Words
    "konse": "which", "konsa": "which", "kaun": "which", "kon": "which", "koun": "which",
    "kya": "what", "kyu": "why", "kyun": "why",
    "kaise": "how", "kaisa": "how",
    "kahan": "where", "kidhar": "where",
    "kab": "when",
    "kitna": "how much", "kitni": "how much", "kitne": "how much",
    
    # Context words (Crucial for "Placement ke baare me")
    "baare": "about", "bare": "about",
    "me": "in", "mein": "in", "ma": "in",
    "ka": "of", "ki": "of", "ke": "of", "ko": "to",
    "aur": "and", "evam": "and", "tatha": "and",
    "sath": "with",
    
    # Verbs / Auxiliaries
    "hai": "is", "hain": "are", "ha": "is", "ho": "are", "h": "is",
    "tha": "was", "thi": "was", "the": "were",
    "chahiye": "want", "mangta": "want",
    "batao": "tell", "btao": "tell", "bol": "tell", "bataiye": "tell",
    "lu": "take", "lena": "take", "le": "take",
    "lagti": "provided", "lagta": "provided", 
    "milti": "available", "milta": "available", "milega": "available",
    "aati": "comes", "aata": "comes", # "Companies konsi aati hai"
    
    # Noun corrections
    "naukri": "job",
}

GREETINGS = {
    "hi": "Hello! I am Sankalp. Ask me about Admissions, Placements, or Fees.",
    "hello": "Hello! How can I help you today?",
    "hola": "Hola! How can I help you?",
    "namaste": "Namaste! Main Sankalp hoon. Kahiye kaise madad karun?",
    "hey": "Hi there! Need info on B.Tech courses?",
}

@app.on_event("startup")
async def load_resources():
    global model, index, metadata
    print("🚀 System starting up...")
    model = SentenceTransformer(MODEL_NAME)
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, 'rb') as f:
        metadata = pickle.load(f)
    print("✅ System Ready!")

class ChatRequest(BaseModel):
    message: str

def get_response_language(text):
    text_lower = text.lower()
    if re.search(r'[\u0900-\u097F]', text):
        return 'hindi'
    words = re.findall(r'\w+', text_lower)
    if any(word in HINGLISH_VOCAB for word in words):
        return 'hindi'
    return 'english'

def normalize_query(text):
    text_lower = text.lower()
    # Normalize: Split by words but preserve sentence structure
    tokens = re.findall(r'\w+|[^\w\s]', text_lower, re.UNICODE)
    
    normalized_tokens = []
    for token in tokens:
        if token in HINGLISH_VOCAB:
            normalized_tokens.append(HINGLISH_VOCAB[token])
        else:
            normalized_tokens.append(token)
            
    return " ".join(normalized_tokens)

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    user_message = request.message.strip()
    
    if user_message.lower() in GREETINGS:
        return {"reply": GREETINGS[user_message.lower()]}

    # 1. Normalize (Hinglish -> English Intent)
    search_query = normalize_query(user_message)

    # 2. Search
    query_vector = model.encode([search_query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(query_vector, k=1)
    
    best_score = float(D[0][0])
    best_idx = int(I[0][0])
    
    print(f"User: '{user_message}' | Search: '{search_query}' | Score: {best_score:.4f}")

    # 3. Threshold Check
    if best_score < SIMILARITY_THRESHOLD:
        return {
            "reply": "Sorry, I didn't find specific information on that. Please ask about **Admission**, **Placement**, **Fees**, or **Courses**."
        }
    
    # 4. Result Logic
    target_lang = get_response_language(user_message)
    result_item = metadata[best_idx]
    
    if target_lang == 'hindi':
        reply_text = result_item['hindi_text']
    else:
        reply_text = result_item['english_text']

    return {"reply": reply_text}