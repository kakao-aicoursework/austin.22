import os
from pathlib import Path

from langchain.memory import ConversationBufferMemory, FileChatMessageHistory


class HistoryGateway:
    def __init__(self):
        self.history_dir = os.path.abspath(os.path.join(os.path.curdir, 'data/history'))
        for file in os.listdir(self.history_dir):
            os.remove(os.path.join(self.history_dir, file))

    def load_conversation_history(self, conversation_id: str):
        file_path = os.path.join(self.history_dir, f'{conversation_id}.json')

        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                f.write("[]")

        return FileChatMessageHistory(file_path=file_path)

    def log_user_message(self, history: FileChatMessageHistory, user_message: str):
        history.add_user_message(user_message)

    def log_bot_message(self, history: FileChatMessageHistory, bot_message: str):
        history.add_ai_message(bot_message)

    def get_chat_history(self, conversation_id: str):
        history = self.load_conversation_history(conversation_id)
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="user_message",
            chat_memory=history,
        )

        return memory.buffer