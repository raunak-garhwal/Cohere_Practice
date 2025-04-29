import cohere
import numpy as np

co = cohere.ClientV2(api_key="B0n3BcGthprXNg5s4z6BmHdsD2hnH1iLcb5eeWnp")

# get the embeddings
phrases = ["i love soup", "soup is my favorite", "london is far away"]

model = "embed-v4.0"
input_type = "search_query"

res = co.embed(
    texts=phrases,
    model=model,
    input_type=input_type,
    output_dimension=1024,
    embedding_types=["float"],
)

(soup1, soup2, london) = res.embeddings.float
print(res.embeddings.float)


# compare them
def calculate_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


print(
    f"For the following sentences:\n1: {phrases[0]}\n2: {phrases[1]}\3 \nThe similarity score is: {calculate_similarity(soup1, soup2):.2f}\n"
)
print(
    f"For the following sentences:\n1: {phrases[0]}\n2: {phrases[2]}\3 \nThe similarity score is: {calculate_similarity(soup1, london):.2f}"
)
