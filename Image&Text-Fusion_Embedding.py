import cohere
import base64

co = cohere.ClientV2(api_key="B0n3BcGthprXNg5s4z6BmHdsD2hnH1iLcb5eeWnp")

# Embed an Images and Texts separately
with open("cat.png", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

# Step 3: Format as data URL
data_url = f"data:image/png;base64,{encoded_string}"

example_doc = [
    {"type": "text", "text": "This is a Scottish Fold Cat"},
    {"type": "image_url", "image_url": {"url": data_url}},
]  # This is where we're fusing text and images.

res = co.embed(
    model="embed-v4.0",
    inputs=[{"content": example_doc}],
    input_type="search_document",
    embedding_types=["float"],
    output_dimension=1024,
).embeddings.float

# This will return a list of length 1 with the texts and image in a combined embedding
print(res)
