from pynecone.base import Base


class Message(Base):
    original_text: str
    text: str
    created_at: str