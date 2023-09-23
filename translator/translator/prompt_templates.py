parse_intent_template = """Your job is to select one intent from the <intent_list>.
<intent_list>
{intent_list}
</intent>
User: {question}

intent:"""

need_web_search_template = """Please answer if you can find the answer in the following documents. select Y or N.
<document>
{related_document}
</document>
User: {question}

answer:"""

answer_with_guide_template = """You are an agent. Please refer to the following document and answer the question.
<document>
{related_document}
</document>
User: {question}

answer:"""

answer_with_web_search_template = """You are an agent. Please refer to the following web search result and answer the question.
<web_search>
{web_search}
</web_search>
User: {question}

answer:"""

answer_not_relevant_template = """you are an agent answer questions related to 카카오 싱크, 카카오톡채널, 카카오소셜. Please answer the question.
role: agent
User: {question}

answer:"""
