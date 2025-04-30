import cohere
import numpy as np

co = cohere.ClientV2(api_key="B0n3BcGthprXNg5s4z6BmHdsD2hnH1iLcb5eeWnp")

texts = ["hello"]

res = co.embed(
    model="embed-v4.0",
    texts=["hello"],
    input_type="search_document",
    embedding_types=["ubinary"],
    output_dimension=1024,
)
print(
    f"Embed v4 Binary at 1024 dimensions results in length {len(res.embeddings.ubinary[0])}"
)

query_emb_bin = np.asarray(res.embeddings.ubinary[0], dtype="uint8")
query_emb_unpacked = np.unpackbits(query_emb_bin, axis=-1).astype(
    "int"
)
query_emb_unpacked = 2 * query_emb_unpacked - 1
print(
    f"Embed v4 Binary at 1024 unpacked will have dimensions: {len(query_emb_unpacked)}"
)

