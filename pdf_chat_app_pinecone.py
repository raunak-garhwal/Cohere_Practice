import fitz  # PyMuPDF
import cohere
import pinecone
import time

# === CONFIGURATION ===
PDF_PATH = "your_pdf.pdf"
COHERE_API_KEY = "your-cohere-api-key"
PINECONE_API_KEY = "your-pinecone-api-key"
PINECONE_ENV = "your-pinecone-environment"  # e.g., "gcp-starter"
INDEX_NAME = "pdf-chat"
CHUNK_SIZE = 500

# === STEP 1: Extract Text ===
def extract_text_from_pdf(path):
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)

# === STEP 2: Chunk Text ===
def chunk_text(text, chunk_size=CHUNK_SIZE):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# === MAIN SETUP ===
def main():
    # Init Cohere and Pinecone
    co = cohere.Client(COHERE_API_KEY)
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

    # Create Pinecone index if it doesn't exist
    if INDEX_NAME not in pinecone.list_indexes():
        pinecone.create_index(name=INDEX_NAME, dimension=1024, metric="cosine")
        time.sleep(5)  # wait for index to be ready

    index = pinecone.Index(INDEX_NAME)

    # === PDF Load + Embedding ===
    text = extract_text_from_pdf(PDF_PATH)
    chunks = chunk_text(text)

    print(f"Embedding {len(chunks)} chunks...")
    embeddings = co.embed(texts=chunks, model="embed-english-v3.0").embeddings

    # Upload to Pinecone
    to_upsert = [(f"id_{i}", embeddings[i], {"text": chunks[i]}) for i in range(len(chunks))]
    index.upsert(vectors=to_upsert)
    print("Chunks embedded and stored in Pinecone.")

    # === User Query Loop ===
    while True:
        user_query = input("\nAsk a question (or type 'exit'): ")
        if user_query.lower() == "exit":
            break

        query_embed = co.embed(texts=[user_query], model="embed-english-v3.0").embeddings[0]

        # Search top 3 relevant chunks
        results = index.query(vector=query_embed, top_k=3, include_metadata=True)
        top_chunks = [match['metadata']['text'] for match in results['matches']]

        # Build prompt for answer generation
        prompt = f"Context:\n" + "\n".join(f"- {chunk}" for chunk in top_chunks)
        prompt += f"\n\nQuestion:\n{user_query}"

        # Generate response
        response = co.generate(model="command-r", prompt=prompt, max_tokens=300)
        print("\n📘 Answer:", response.generations[0].text.strip())

if __name__ == "__main__":
    main()
