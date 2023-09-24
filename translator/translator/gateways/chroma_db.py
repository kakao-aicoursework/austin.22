import os

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings


class ChromaDbGateway:
    def __init__(self):
        self.data_dir = os.path.join(os.path.pardir, 'translator/data')
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
