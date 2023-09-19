"""Welcome to Pynecone! This file outlines the steps to create a basic app."""

# Import pynecone.
import openai
from datetime import datetime
from dotenv import load_dotenv

import pynecone as pc
from pynecone.base import Base

from langchain.chat_models import ChatOpenAI
from langchain.chains import SequentialChain
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain

load_dotenv()


class LLMGateway:
    def __init__(self, model_name: str, instruction_file_path: str):
        def load_instruction_file(file_path: str):
            with open(file_path, "r") as f:
                instruction = f.read()
            return instruction

        self.instruction = load_instruction_file(file_path=instruction_file_path)
        self.llm = ChatOpenAI(model_name=model_name, temperature=0.1, max_tokens=700)

    def ask(self, system_instruction: str, question: str) -> str:
        messages = [{"role": "system", "content": system_instruction},
                    {"role": "user", "content": question}]

        # API 호출
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                messages=messages)
        answer = response['choices'][0]['message']['content']
        return answer

    def ask_with_langchain_one_depth(self, question: str) -> str:
        prompt_template = f"너는 고객 응대 직원이야. 다음 가이드를 기반으로 적절한 대답을 형성해 줘. \n {self.instruction}" + "\n\n질문: {question}\n\n답변:"

        ask_chain = LLMChain(llm=self.llm, prompt=ChatPromptTemplate.from_template(
            template=prompt_template
        ), output_key="answer", verbose=True)

        process_chain = SequentialChain(
            chains=[ask_chain],
            input_variables=["question"],
            verbose=True
        )

        resp = process_chain({"question": question})

        # Return
        return resp['answer']

    def ask_with_langchain_multi_depth(self, question: str) -> str:
        ask_prompt_template = f"너는 고객 응대 직원이야. 다음 가이드를 기반으로 적절한 대답을 형성해 줘. \n {self.instruction}" + "\n\n질문: {question}\n\n답변:"
        ask_chain = LLMChain(llm=self.llm, prompt=ChatPromptTemplate.from_template(
            template=ask_prompt_template
        ), output_key="first_answer", verbose=True)

        verify_prompt_template = (f"다음 가이드를 기반으로 질문에 대한 답변이 적절한 지 검토하고 틀린 부분이 있으면 수정한 답변을 텍스트로 출력해줘 \n {self.instruction}" +
                                  "\n\n질문: {question}\n\n답변:{first_answer} \n\n답변 수정:")
        verify_chain = LLMChain(llm=self.llm, prompt=ChatPromptTemplate.from_template(
            template=verify_prompt_template
        ), output_key="answer", verbose=True)

        process_chain = SequentialChain(
            chains=[ask_chain, verify_chain],
            input_variables=["question"],
            verbose=True
        )

        resp = process_chain({"question": question})

        # Return
        return resp['answer']


gateway = LLMGateway(
    model_name="gpt-3.5-turbo-16k",
    instruction_file_path="project_data_카카오싱크.txt"
)

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

    # src_lang: str = "한국어"
    # trg_lang: str = "영어"

    # @pc.var
    def output(self) -> str:
        if not self.text.strip():
            return "Answer will appear here."
        answer = gateway.ask_with_langchain_multi_depth(self.text)
        # translated = ask_text_to_chatgpt(self.text, src_lang=self.src_lang, trg_lang=self.trg_lang)
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
