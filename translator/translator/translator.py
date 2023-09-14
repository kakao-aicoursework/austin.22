"""Welcome to Pynecone! This file outlines the steps to create a basic app."""

# Import pynecone.
import openai
import os
from datetime import datetime

import pynecone as pc
from pynecone.base import Base


# openai.api_key = "<YOUR_OPENAI_API_KEY>"
openai.api_key = ""


def ask_text_to_chatgpt(text) -> str:
    # fewshot 예제를 만들고
    # def build_fewshot(src_lang, trg_lang):
    #     src_examples = parallel_example[src_lang]
    #     trg_examples = parallel_example[trg_lang]
    #
    #     fewshot_messages = []
    #
    #     for src_text, trg_text in zip(src_examples, trg_examples):
    #         fewshot_messages.append({"role": "user", "content": src_text})
    #         fewshot_messages.append({"role": "assistant", "content": trg_text})
    #
    #     return fewshot_messages

    # system instruction 만들고
    system_instruction = f"assistant는 챗봇으로 동작한다. 사용자의 질문에 적절한 대답을 text로 출력한다."

    # messages를만들고
    # fewshot_messages = build_fewshot(src_lang=src_lang, trg_lang=trg_lang)

    messages = [{"role": "system", "content": system_instruction},
                {"role": "user", "content": text}
                ]

    # API 호출
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                            messages=messages)
    translated_text = response['choices'][0]['message']['content']
    # Return
    return translated_text


class Message(Base):
    original_text: str
    text: str
    created_at: str
    to_lang: str


class State(pc.State):
    """The app state."""

    text: str = ""
    messages: list[Message] = []
    # src_lang: str = "한국어"
    # trg_lang: str = "영어"

    @pc.var
    def output(self) -> str:
        if not self.text.strip():
            return "Answer will appear here."
        translated = ask_text_to_chatgpt(
            self.text)
            # self.text, src_lang=self.src_lang, trg_lang=self.trg_lang)
        return translated

    def post(self):
        self.messages = [
            Message(
                original_text=self.text,
                text=self.output,
                created_at=datetime.now().strftime("%B %d, %Y %I:%M %p"),
                to_lang=self.trg_lang,
            )
        ] + self.messages


# Define views.


def header():
    """Basic instructions to get started."""
    return pc.box(
        pc.text("Chat Bot 🌈", font_size="2rem"),
        pc.text(
            "Ask anythings and post them as messages!",
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
                pc.text(message.to_lang),
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
        pc.text(State.output),
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
            pc.foreach(State.messages, message),
            margin_top="2rem",
            spacing="1rem",
            align_items="left"
        ),
        padding="2rem",
        max_width="600px"
    )


# Add state and page to the app.
app = pc.App(state=State)
app.add_page(index, title="Translator")
app.compile()
