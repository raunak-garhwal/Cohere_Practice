import cohere

co = cohere.ClientV2("B0n3BcGthprXNg5s4z6BmHdsD2hnH1iLcb5eeWnp")

while True:
    user_message = input("\n\nEnter your prompt: ")

    bot_response = co.chat_stream(
        model="command-a-03-2025",
        # model="command-r7b-12-2024",
        # model="command-r-plus-08-2024",
        # model="command-r-08-2024",
        messages=[{"role": "user", "content": user_message}],
    )
    
    print("Bot's message: ", end="")

    for event in bot_response:
        if event:
            if event.type == "content-delta":
                print(event.delta.message.content.text, end="")
