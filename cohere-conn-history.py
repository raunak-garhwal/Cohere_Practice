import uuid
import cohere
from cohere import ChatConnector
from typing import List

co = cohere.Client("B0n3BcGthprXNg5s4z6BmHdsD2hnH1iLcb5eeWnp")

class Chatbot:
    def __init__(self, connectors: List[str]):
        """
        Initializes an instance of the Chatbot class.

        """
        self.conversation_id = str(uuid.uuid4())
        self.connectors = [ChatConnector(id=connector) for connector in connectors]

    def run(self):
        """
        Runs the chatbot application.

        """
        while True:
            # Get the user message
            message = input("\n\nUser: ")

            # Typing "quit" ends the conversation
            if message.lower() == "quit":
                print("\nEnding chat.....\n")
                break

            # Generate response
            response = co.chat_stream(
                    message=message,
                    model="command-a-03-2025",
                    conversation_id=self.conversation_id,
                    connectors=self.connectors,
            )

            # Print the chatbot response, citations and documents
            print("\nChatbot: ", end="")
            # citations = []
            # cited_documents = []

            # # Display response
            for event in response:
                if event.event_type == "text-generation":
                    print(event.text, end="")
            #     elif event.event_type == "citation-generation":
            #         citations.extend(event.citations)
            #     elif event.event_type == "stream-end":
            #         cited_documents = event.response.documents

            # # Display citations and source documents
            # if citations:
            #   print("\n\nCITATIONS:")
            #   for citation in citations:
            #     print(citation)

            #   print("\nDOCUMENTS:")
            #   for document in cited_documents:
            #     print({'id': document['id'],
            #           'snippet': document['snippet'][:400] + '...',
            #           'title': document['title'],
            #           'url': document['url']})

            # print(f"\n{'-'*100}\n")

# Define the connector
connectors = ["web-search"]

# Create an instance of the Chatbot class
chatbot = Chatbot(connectors)

# Run the chatbot
chatbot.run()
