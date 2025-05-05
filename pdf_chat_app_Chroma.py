import fitz  # PyMuPDF
import cohere
import chromadb
from chromadb.config import Settings

# ========== CONFIG ==========
PDF_PATH = "/content/Demo1.pdf"
COHERE_API_KEY = "B0n3BcGthprXNg5s4z6BmHdsD2hnH1iLcb5eeWnp"
EMBED_MODEL = "embed-v4.0"
GEN_MODEL = "command-r-plus-08-2024"
CHUNK_SIZE = 500
PERSIST_DIR = "./chroma_store"  # must remain constant

# Use a single Chroma client globally with consistent settings
chroma_client = chromadb.Client(Settings(persist_directory=PERSIST_DIR, anonymized_telemetry=False))

# ========== STEP 1: Extract Text from PDF ==========
def extract_text_from_pdf(path):
    doc = fitz.open(path)
    text = "\n".join(page.get_text() for page in doc)
    return text

# ========== STEP 2: Chunk Text ==========
def chunk_text(text, chunk_size=CHUNK_SIZE):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

# ========== STEP 3: Embed and Store ==========
def create_vector_store(chunks, embeddings):
    # Remove old collection if it exists
    if "pdf_chunks" in [col.name for col in chroma_client.list_collections()]:
        chroma_client.delete_collection("pdf_chunks")

    collection = chroma_client.create_collection(name="pdf_chunks")
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    return collection

# ========== STEP 4: Embed Query and Retrieve ==========
def get_top_chunks(collection, query_embedding, top_k=3):
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results['documents'][0]

# ========== STEP 5: Build Prompt and Generate Answer ==========
def build_prompt(chunks, user_query):
    context = "\n".join(f"- {chunk}" for chunk in chunks)
    return f"Context:\n{context}\n\nQuestion:\n{user_query}"

def generate_answer(co, prompt):
    response = co.generate(model=GEN_MODEL, prompt=prompt, max_tokens=600)
    return response.generations[0].text.strip()

# ========== MAIN ==========
def main():
    # Init Cohere
    co = cohere.Client(COHERE_API_KEY)

    # Step 1: Extract and Chunk
    full_text = extract_text_from_pdf(PDF_PATH)
    chunks = chunk_text(full_text)

    # Step 2: Embed Chunks
    embeddings = co.embed(texts=chunks, model=EMBED_MODEL).embeddings

    # Step 3: Store in Chroma
    collection = create_vector_store(chunks, embeddings)

    # Step 4: Accept Query Loop
    while True:
        user_query = input("\nAsk a question (or type 'exit'): ")
        if user_query.lower() == 'exit':
            break

        query_embedding = co.embed(texts=[user_query], model=EMBED_MODEL).embeddings[0]
        top_chunks = get_top_chunks(collection, query_embedding)
        prompt = build_prompt(top_chunks, user_query)

        # Step 5: Generate and Show Answer
        answer = generate_answer(co, prompt)
        print("\n📘 Answer:", answer)

if __name__ == "__main__":
    main()
