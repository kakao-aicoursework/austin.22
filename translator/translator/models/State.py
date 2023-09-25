import pynecone as pc
from datetime import datetime

from translator.models.Message import Message
from translator.gateways.llm import LLMGateway

gateway = LLMGateway(model_name="gpt-3.5-turbo-16k", debug=False)


class State(pc.State):
    """The app state."""
    text: str = ""
    messages: list[Message] = [
        Message(
            original_text="fff",
            text="Answer will appear here.",
            created_at=datetime.now().strftime("%B %d, %Y %I:%M %p"),
        ),
    ]

    # @pc.var
    def output(self) -> str:
        if not self.text.strip():
            return "Answer will appear here."
        answer = gateway.ask_with_langchain(self.text)
        return answer

    def post(self):
        text = self.output()
        new_message = Message(
            original_text=self.text,
            text=text,
            created_at=datetime.now().strftime("%B %d, %Y %I:%M %p"),
        )
        self.messages += [new_message]
