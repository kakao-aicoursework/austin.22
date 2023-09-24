parse_intent_template = """Your job is to select one intent from the <intent_list>.
<intent_list>
{intent_list}
</intent>
User: {question}

intent:"""

need_web_search_template = """If <document> can help you to answer the <question>, answer Y. If not, answer N.
<document>
{related_document}
</document>
User: {question}

answer:"""

answer_with_guide_template = """You are an agent. Please refer to the following document and answer the question considering the context.
<document>
{related_document}
</document>

<context>
{chat_history}
User: {question}

answer:"""

compress_web_search_template = """Your job is to extract content related to <question> from <search_results>

<question>
{question}
</question>

<search_results>
{web_search_result}
</search_results>

Compressed:"""

answer_with_web_search_template = """You are an agent. Please refer to the following web search result and document.
Answer the question considering the context. In case that you can't find the answer, apologize to the client.

<web_search_result>
{compressed_web_search}
</web_search_result>

<document>
{related_document}
</document>

<context>
{chat_history}
User: {question}

answer:"""

answer_not_relevant_template = """you are an agent answer questions related to 카카오 싱크, 카카오톡채널, 카카오소셜. Please answer the question considering the context.
role: agent

<context>
{chat_history}
User: {question}

answer:"""
