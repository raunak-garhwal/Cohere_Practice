import fitz  # PyMuPDF
import cohere
import numpy as np

# === CONFIG ===
PDF_PATH = "Demo1.pdf"
COHERE_API_KEY = "B0n3BcGthprXNg5s4z6BmHdsD2hnH1iLcb5eeWnp"
CHUNK_SIZE = 250

# === INIT COHERE ===
co = cohere.Client(COHERE_API_KEY)

# === STEP 1: Extract PDF Text ===
def extract_text_from_pdf(path):
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)

# === STEP 2: Chunk Text ===
def chunk_text(text, chunk_size=CHUNK_SIZE):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# === STEP 3: Compute Embeddings ===
def get_embeddings(texts):
    response = co.embed(texts=texts, model="embed-v4.0")
    return np.array(response.embeddings)

# === STEP 4: Cosine Similarity Search ===
def cosine_similarity(a, b):
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b, axis=1)[:, np.newaxis]
    return np.dot(b_norm, a_norm)

# === MAIN ===
def main():
    text = extract_text_from_pdf(PDF_PATH)
    chunks = chunk_text(text)
    print(f"🔍 Embedding {len(chunks)} chunks...")
    chunk_embeddings = get_embeddings(chunks)
    print(chunks)
    print(chunk_embeddings)

    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            break

        query_embed = get_embeddings([query])[0]
        sims = cosine_similarity(query_embed, chunk_embeddings)
        top_idxs = sims.argsort()[-3:][::-1]
        top_chunks = [chunks[i] for i in top_idxs]

        prompt = "Context:\n" + "\n".join(f"- {c}" for c in top_chunks)
        prompt += f"\n\nQuestion:\n{query}"

        response = co.generate(model="command-r-plus-08-2024", prompt=prompt, max_tokens=600)
        print("\n📘 Answer:", response.generations[0].text.strip())

if __name__ == "__main__":
    main()
