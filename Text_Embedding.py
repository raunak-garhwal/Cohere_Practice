import cohere

co = cohere.ClientV2(api_key="B0n3BcGthprXNg5s4z6BmHdsD2hnH1iLcb5eeWnp")

texts = [
    "Hello from Cohere!",
    "مرحبًا من كوهير!",
    "Hallo von Cohere!",
    "Bonjour de Cohere!",
    "¡Hola desde Cohere!",
    "Olá do Cohere!",
    "Ciao da Cohere!",
    "您好，来自 Cohere!",
    "कोहेरे से नमस्ते!"
]

response = co.embed(
    model="embed-v4.0",
    texts=texts,
    input_type="classification",
    output_dimension=1024,
    embedding_types=["float"]
)

embeddings = response.embeddings.float
print(embeddings[0][:5])
