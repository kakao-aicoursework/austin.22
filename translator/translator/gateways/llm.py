import os
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain, SequentialChain

from translator import prompt_templates as PromptTemplates
from translator.gateways.chroma_db import ChromaDbGateway
from translator.gateways.history import HistoryGateway
from translator.gateways.web_search import WebSearchGateway

load_dotenv()

intent_list = [
    "bug: Related to a bug, vulnerability, unexpected error with an existing feature",
    "question: A specific question about feature, usage, or behavior",
    "not_relevant: Not relevant to the 카카오싱크, 카카오소셜 or 카카오톡채널"
]


class LLMGateway:
    def __init__(self, model_name: str, debug: bool = False):
        self.llm = ChatOpenAI(model_name=model_name, temperature=0.0, max_tokens=700)
        self.chroma = ChromaDbGateway()
        self.history = HistoryGateway()
        self.web_search = WebSearchGateway()
        self.debug = debug

    def create_chain(self, template: str, output_key: str):
        return LLMChain(llm=self.llm, prompt=ChatPromptTemplate.from_template(
            template=template
        ), output_key=output_key, verbose=True)

    def ask_with_langchain(self, question: str, conversation_id: str = 'default_conversation_id') -> str:
        if self.debug:
            print("===========================================================")
            print("question: ", question)
        answer = ""
        history_file = self.history.load_conversation_history(conversation_id=conversation_id)

        parse_intent_chain = self.create_chain(template=PromptTemplates.parse_intent_template, output_key="intent")

        context = {
            "intent_list": "\n".join(intent_list),
            "question": question,
            "chat_history": self.history.get_chat_history(conversation_id=conversation_id)
        }
        intent = parse_intent_chain.run(context)
        if self.debug:
            print("intent: ", intent)
        if intent == "question" or intent == "bug":
            context['related_document'] = self.chroma.query_db(query=question, use_retriever=True)
            if self.debug:
                print("<question or bug stage>")
                print("question:", question)
                print("related_document: ", context['related_document'])

            ask_need_web_search_chain = self.create_chain(
                template=PromptTemplates.need_web_search_template,
                output_key="need_web_search"
            )
            need_web_search = ask_need_web_search_chain.run(context)
            if self.debug:
                print("need web search", need_web_search)

            if need_web_search == "N":
                ask_with_guide_chain = self.create_chain(
                    template=PromptTemplates.answer_with_guide_template,
                    output_key="answer"
                )
                answer = ask_with_guide_chain.run(context)
            else:
                context['web_search_result'] = self.web_search.query_web_search(question)

                compress_web_search_chain = self.create_chain(
                    template=PromptTemplates.compress_web_search_template,
                    output_key="compressed_web_search"
                )
                ask_web_search_chain = self.create_chain(
                    template=PromptTemplates.answer_with_web_search_template,
                    output_key="answer"
                )

                process_chain = SequentialChain(
                    chains=[compress_web_search_chain, ask_web_search_chain],
                    input_variables=["question", "web_search_result", "related_document", "chat_history"],
                    verbose=True
                )

                answer = process_chain(context)['answer']

                if self.debug:
                    print("web search question:", question)
                    print("web search result:", context['web_search_result'])
                    print("web search stage answer:", answer)

        else:
            answer_not_relevant_chain = self.create_chain(
                template=PromptTemplates.answer_not_relevant_template,
                output_key="answer"
            )
            answer = answer_not_relevant_chain.run(context)
        self.history.log_user_message(history=history_file, user_message=question)
        self.history.log_bot_message(history=history_file, bot_message=answer)
        return answer
