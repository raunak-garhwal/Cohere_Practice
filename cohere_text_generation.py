import cohere

co = cohere.ClientV2("B0n3BcGthprXNg5s4z6BmHdsD2hnH1iLcb5eeWnp")

response = co.chat(
    model="command-a-03-2025",
    messages=[
        {
            "role": "user",
            "content": "I'm joining a new startup called Co1t today. Could you help me write a one-sentence introduction message to my teammates.",
        }
    ],
)

# response = co.chat(
#     model="command-a-03-2025",
#     messages=[
#         {
#             "role": "system",
#             "content": "You respond in concise sentences.",
#         },
#         {   
#             "role": "user",
#             "content": "Hello"
#         },
#         {
#             "role": "assistant",
#             "content": "Hi, how can I help you today?",
#         },
#         {
#             "role": "user",
#             "content": "I'm joining a new startup called Co1t today. Could you help me write a one-sentence introduction message to my teammates.",
#         },
#     ],
# )

print(response.message.content[0].text)

res = co.chat_stream(
    model="command-a-03-2025",
    messages=[
        {
            "role": "user",
            "content": "I'm joining a new startup called Co1t today. Could you help me write a one-sentence introduction message to my teammates.",
        }
    ],
)

for chunk in res:
    if chunk:
        if chunk.type == "content-delta":
            print(chunk.delta.message.content.text, end="")

