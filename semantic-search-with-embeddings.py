import cohere
import numpy as np

co = cohere.ClientV2("FLfuHNGPxELTYaAtxwDapzbZUQlTfYHG9wniXnNt")

### STEP 1: Embed the documents

# Define the documents
documents = [
    "New employees are encouraged to join the Slack workspace for real-time updates and team communication.",
    "You can connect with teammates during weekly team lunches held every Friday in the cafeteria.",
    "The company organizes monthly team-building events like bowling, escape rooms, and hackathons.",
    "Our open office layout encourages spontaneous collaboration and quick brainstorming sessions.",
    "Employees use the random Slack channel to share fun content and spark informal conversations."
]

# Constructing the embed_input object
embed_input = [
    {"content": [{"type": "text", "text": doc}]} for doc in documents
]

# Embed the documents
doc_emb = co.embed(
    inputs=embed_input,
    model="embed-v4.0",
    output_dimension=1024,
    input_type="search_document",
    embedding_types=["float"],
).embeddings.float

### STEP 2: Embed the query

# Add the user query
query = "How can I interact with my coworkers?"

query_input = [{"content": [{"type": "text", "text": query}]}]

# Embed the query
query_emb = co.embed(
    inputs=query_input,
    model="embed-v4.0",
    input_type="search_query",
    output_dimension=1024,
    embedding_types=["float"],
).embeddings.float

### STEP 3: Return the most similar documents

# Calculate similarity scores
scores = np.dot(query_emb, np.transpose(doc_emb))[0]

# Sort and filter documents based on scores
top_n = 2
top_doc_idxs = np.argsort(-scores)[:top_n]

# Display search results
for idx, docs_idx in enumerate(top_doc_idxs):
    print(f"Rank: {idx+1}")
    print(f"Document: {documents[docs_idx]}\n")
