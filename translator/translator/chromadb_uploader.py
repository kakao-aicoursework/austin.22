import os
import sys

from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

DATA_DIR = os.path.join(os.path.pardir, 'data')
CHROMA_PERSIST_DIR = os.path.join(DATA_DIR, 'upload/chroma_persist')
CHROMA_COLLECTION_NAME = 'dosu-bot'

load_dotenv()


def upload_embeddings_from_file(file_path):
    loader = TextLoader
    documents = loader(file_path).load()

    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    Chroma.from_documents(
        docs,
        OpenAIEmbeddings(),
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR
    )


def upload_embeddings_from_dir(dir_path):
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)

                try:
                    upload_embeddings_from_file(file_path)
                except Exception as e:
                    print(f'Failed to upload {file_path}: {e}')


if __name__ == '__main__':
    dir_path = sys.argv[1] if len(sys.argv) == 2 else os.path.join(os.path.dirname(os.getcwd()), "data")
    upload_embeddings_from_dir(dir_path)