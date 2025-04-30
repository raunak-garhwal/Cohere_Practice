import cohere

co = cohere.ClientV2(api_key="B0n3BcGthprXNg5s4z6BmHdsD2hnH1iLcb5eeWnp")

texts = ["hello","56"]

response = co.embed(
    model="embed-v4.0",
    texts=texts,
    output_dimension=1024,
    input_type="classification",
    embedding_types=["float"]
)

print(response.embeddings.float)   # returns a list of embeddings for each text in the input list
print(len(response.embeddings.float))   # returns the number of texts in the input list
print(len(response.embeddings.float[0]))   # returns a vector that is 1024 dimensions
