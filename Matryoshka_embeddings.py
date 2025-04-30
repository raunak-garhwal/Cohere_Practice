import cohere

co = cohere.ClientV2(api_key="B0n3BcGthprXNg5s4z6BmHdsD2hnH1iLcb5eeWnp")

texts = ["hello"]

response = co.embed(
    model="embed-v4.0",
    texts=texts,
    output_dimension=1024,
    input_type="classification",
    embedding_types=["float"]
)

print(response.embeddings.float)  # returns a vector that is 1024 dimensions
print(len(response.embeddings.float[0]))
