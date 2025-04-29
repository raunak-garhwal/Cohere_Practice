import uuid
import cohere
from cohere import ChatConnector
from typing import List
import PyPDF2

co = cohere.Client("B0n3BcGthprXNg5s4z6BmHdsD2hnH1iLcb5eeWnp")

def load_pdf_text(pdf_path):
    pdf_text = ""
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            pdf_text += page.extract_text() + "\n"
    return pdf_text

class Chatbot:
    def __init__(self, connectors: List[str], document_text: str):
        self.conversation_id = str(uuid.uuid4())
        self.connectors = [ChatConnector(id=connector) for connector in connectors]
        self.document_text = document_text

    def run(self):
        while True:
            message = input("\n\nUser: ")

            if message.lower() == "quit":
                print("Ending chat.")
                break

            full_message = f"Refer to the following document content:\n\n{self.document_text}\n\nNow answer this question:\n{message}"

            response = co.chat_stream(
                message=full_message,
                model="command-a-03-2025",
                conversation_id=self.conversation_id,
                connectors=self.connectors,
            )

            print("\nChatbot:")

            for event in response:
                if event.event_type == "text-generation":
                    print(event.text, end="")

if __name__ == "__main__":
    pdf_path = "Demo.pdf"
    document_text = load_pdf_text(pdf_path)

    connectors = ["web-search"]

    chatbot = Chatbot(connectors=connectors, document_text=document_text)
    chatbot.run()
