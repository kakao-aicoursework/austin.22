"""Welcome to Pynecone! This file outlines the steps to create a basic app."""
import os
from datetime import datetime

import pynecone as pc
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory
from langchain.prompts.chat import ChatPromptTemplate
from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.vectorstores import Chroma
from pynecone.base import Base

from translator import prompt_templates as PromptTemplates

load_dotenv()

intent_list = [
    "bug: Related to a bug, vulnerability, unexpected error with an existing feature",
    "question: A specific question about feature, usage, or behavior",
    "not_relevant: Not relevant to the 카카오싱크, 카카오소셜 or 카카오톡채널"
]

search = GoogleSearchAPIWrapper(
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    google_cse_id=os.getenv("GOOGLE_CSE_ID")
)

search_tool = Tool(
    name="Google Search",
    description="Search Google for recent results.",
    func=search.run,
)


def query_web_search(user_message: str) -> str:
    context = {"user_message": user_message}
    return search_tool.run(user_message)


class HistoryGateway:
    def __init__(self):
        self.history_dir = os.path.abspath(os.path.join(os.path.pardir, 'data/history'))

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


class ChromaDbGateway:
    def __init__(self):
        self.data_dir = os.path.join(os.path.pardir, 'data')
        self.chroma_persist_dir = os.path.join(self.data_dir, 'upload/chroma_persist')
        self.chroma_collection_name = 'dosu-bot'
        self.db = Chroma(
            persist_directory=self.chroma_persist_dir,
            embedding_function=OpenAIEmbeddings(),
            collection_name=self.chroma_collection_name
        )
        self.retriever = self.db.as_retriever()

    def query_db(self, query: str, use_retriever: bool):
        if use_retriever:
            docs = self.retriever.get_relevant_documents(query)
        else:
            docs = self.db.similarity_search(query)

        str_docs = [doc.page_content for doc in docs]
        return str_docs


class LLMGateway:
    def __init__(self, model_name: str):
        self.llm = ChatOpenAI(model_name=model_name, temperature=0.0, max_tokens=700)
        self.chroma = ChromaDbGateway()
        self.history = HistoryGateway()

    def create_chain(self, template: str, output_key: str):
        return LLMChain(llm=self.llm, prompt=ChatPromptTemplate.from_template(
            template=template
        ), output_key=output_key, verbose=False)

    def ask_with_langchain(self, question: str, conversation_id: str = 'default_conversation_id') -> str:
        answer = ""
        history_file = self.history.load_conversation_history(conversation_id=conversation_id)

        parse_intent_chain = self.create_chain(template=PromptTemplates.parse_intent_template, output_key="intent")

        context = {
            "intent_list": "\n".join(intent_list),
            "question": question,
            "chat_history": self.history.get_chat_history(conversation_id=conversation_id)
        }
        intent = parse_intent_chain.run(context)
        print("intent: ", intent)
        if intent == "question" or intent == "bug":
            context['related_document'] = self.chroma.query_db(query=question, use_retriever=True)
            print("related_document: ", context['related_document'])

            ask_need_web_search_chain = self.create_chain(
                template=PromptTemplates.need_web_search_template,
                output_key="need_web_search"
            )
            need_web_search = ask_need_web_search_chain.run(context)
            print("need web search", need_web_search)

            if need_web_search == "N":
                ask_with_guide_chain = self.create_chain(
                    template=PromptTemplates.answer_with_guide_template,
                    output_key="answer"
                )
                answer = ask_with_guide_chain.run(context)
            else:
                ask_web_search_chain = self.create_chain(
                    template=PromptTemplates.answer_with_web_search_template,
                    output_key="answer"
                )
                answer = ask_web_search_chain.run(context)
                print("web search result:", answer)
            # process_chain = SequentialChain(
            #     chains=[],
            #     input_variables=["related_document", "chat_history", "question"],
            #     verbose=True
            # )
            #
            # resp = process_chain(context)

        else:
            answer_not_relevant_chain = self.create_chain(
                template=PromptTemplates.answer_not_relevant_template,
                output_key="answer"
            )
            answer = answer_not_relevant_chain.run(context)
        self.history.log_user_message(history=history_file, user_message=question)
        self.history.log_bot_message(history=history_file, bot_message=answer)
        return answer


gateway = LLMGateway(model_name="gpt-3.5-turbo-16k")


class Message(Base):
    original_text: str
    text: str
    created_at: str


class State(pc.State):
    """The app state."""

    text: str = ""
    messages: list[Message] = [
        Message(
            original_text="Answer will appear here.",
            text="Answer will appear here.",
            created_at=datetime.now().strftime("%B %d, %Y %I:%M %p"))
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
            created_at=datetime.now().strftime("%B %d, %Y %I:%M %p")
        )
        self.messages += [new_message]


#############################################################
# Define views.
#############################################################

def header():
    """Basic instructions to get started."""
    return pc.box(
        pc.text("🌈카카오 싱크 봇 🌈", font_size="2rem"),
        pc.text(
            "Ask anythings about KakaoSync and post them as messages!",
            margin_top="0.5rem",
            color="#666",
        ),
    )


def down_arrow():
    return pc.vstack(
        pc.icon(
            tag="arrow_down",
            color="#666",
        )
    )


def text_box(text):
    return pc.text(
        text,
        background_color="#fff",
        padding="1rem",
        border_radius="8px",
    )


def message(message):
    return pc.box(
        pc.vstack(
            text_box(message.original_text),
            down_arrow(),
            text_box(message.text),
            pc.box(
                pc.text("답변"),
                pc.text(" · ", margin_x="0.3rem"),
                pc.text(message.created_at),
                display="flex",
                font_size="0.8rem",
                color="#666",
            ),
            spacing="0.3rem",
            align_items="left",
        ),
        background_color="#f5f5f5",
        padding="1rem",
        border_radius="8px",
    )


def smallcaps(text, **kwargs):
    return pc.text(
        text,
        font_size="0.7rem",
        font_weight="bold",
        text_transform="uppercase",
        letter_spacing="0.05rem",
        **kwargs,
    )


def output():
    return pc.box(
        pc.box(
            smallcaps(
                "Output",
                color="#aeaeaf",
                background_color="white",
                padding_x="0.1rem",
            ),
            position="absolute",
            top="-0.5rem",
        ),
        pc.text(State.messages[-1].text),
        padding="1rem",
        border="1px solid #eaeaef",
        margin_top="1rem",
        border_radius="8px",
        position="relative",
    )


def index():
    """The main view."""
    return pc.container(
        header(),
        pc.input(
            placeholder="Text to ask",
            on_blur=State.set_text,
            margin_top="1rem",
            border_color="#eaeaef"
        ),
        # pc.select(
        #     list(parallel_example.keys()),
        #     value=State.src_lang,
        #     placeholder="Select a language",
        #     on_change=State.set_src_lang,
        #     margin_top="1rem",
        # ),
        # pc.select(
        #     list(parallel_example.keys()),
        #     value=State.trg_lang,
        #     placeholder="Select a language",
        #     on_change=State.set_trg_lang,
        #     margin_top="1rem",
        # ),
        output(),
        pc.button("Post", on_click=State.post, margin_top="1rem"),
        pc.vstack(
            pc.foreach(State.messages[1:], message),
            margin_top="2rem",
            spacing="1rem",
            align_items="left"
        ),
        padding="2rem",
        max_width="600px"
    )


# Add state and page to the app.
app = pc.App(state=State)
app.add_page(index, title="Chat Bot")
app.compile()
