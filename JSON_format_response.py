import cohere

co = cohere.ClientV2(api_key="B0n3BcGthprXNg5s4z6BmHdsD2hnH1iLcb5eeWnp")

res = co.chat(
    model="command-a-03-2025",
    messages=[
        {
            "role": "user",
            "content": "What is LLM describe it in json format?",
        }
    ],
    response_format={"type": "json_object"},
)

print(res.message.content[0].text)
