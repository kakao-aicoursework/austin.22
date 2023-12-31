"""Welcome to Pynecone! This file outlines the steps to create a basic app."""
import pynecone as pc
from dotenv import load_dotenv
from translator.models.State import State

from translator.models.Message import Message

load_dotenv()


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
        background_color="#ffea00",
        padding="1rem",
        border_radius="8px",
        align_self="flex-end",
    )


def ai_text_box(text):
    return pc.text(
        text,
        background_color="#fff",
        padding="1rem",
        border_radius="8px",
        align_self="flex-start",
    )


def message(message):
    return pc.box(
        pc.vstack(
            text_box(message.original_text),
            pc.box(
                pc.text(message.created_at),
                display="flex",
                font_size="0.8rem",
                color="#666",
                align_self="flex-end",
            ),
            ai_text_box(message.text),
            pc.box(
                pc.text(message.created_at),
                display="flex",
                font_size="0.8rem",
                color="#666",
                align_self="flex-start",
            ),
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
        # output(),
        pc.vstack(
            pc.foreach(State.messages[1:], message),
            margin_top="2rem",
            spacing="1rem",
            align_items="left",
            height="65vh",
            overflow_y="scroll",
            background_color="#f5f5f5",
        ),
        pc.hstack(
            pc.input(
                placeholder="Text to ask",
                on_blur=State.set_text,
                margin_top="auto",
                border_color="#eaeaef",
            ),
            pc.button("Post", on_click=State.post, margin="auto"),
            margin="2rem"
        ),
        padding="2rem",
        max_width="600px",
    )


# Add state and page to the app.
app = pc.App(state=State)
app.add_page(index, title="Chat Bot")
app.compile()
