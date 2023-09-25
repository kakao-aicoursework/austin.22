parse_intent_template = """Your job is to select one user question's intent from the <intent_list>.
<intent_list>
{intent_list}
</intent>

<question>
User: {question}
</question>

intent:"""

need_web_search_template = """If <document> is sufficient to help you to answer the <question>, answer N. If not, answer Y.
<document>
{related_document}
</document>

<question>
User: {question}
</question>

answer:"""

answer_with_guide_template = """You are an agent. Please refer to the following document and answer the question considering context.
<document>
{related_document}
</document>

<context>
{chat_history}
</context>

<question>
User: {question}
</question>

Let’s think step by step.

agent:"""

compress_web_search_template = """Your job is to extract content related to <question> from <search_results>. 
Let’s think step by step.

<question>
{question}
</question>

<search_results>
{web_search_result}
</search_results>

Compressed:"""

compress_web_search_template_v2 = """Your job is to extract content related to <question> from <search_results>. 
First, gather all sentences related to <question> from <search_results>.
Let’s think step by step.

<question>
{question}
</question>

<search_results>
{web_search_result}
</search_results>

Compressed:"""

answer_with_web_search_template = """You are an agent. Please refer to the following <web search result> and <related_document>.
Answer the question considering the context. Let’s think step by step.

<web_search_result>
{compressed_web_search}
</web_search_result>

<related_document>
{related_document}
</related_document>

<context>
{chat_history}
</context>

<question>
User: {question}
</question>

Agent:"""

answer_not_relevant_template = """you are an agent answer questions related to 카카오 싱크, 카카오톡 채널, 카카오 소셜. Please answer the question considering context.
role: agent

<context>
{chat_history}
</context>

<question>
User: {question}
</question>

Agent:"""

need_web_search_template_2 = """If <context> isneed_web_search_template sufficient to help you to answer the <question>, answer N. If not, answer Y.
<context>
{chat_history}
</context>

<question>
User: {question}

answer:"""

answer_with_context_template = """Refer to <context>, <related_document> and <web_search_result> and answer the question.
Let's think step by step.

<context>
{chat_history}
</context>

<related_document>
{related_document}
</related_document>

<web_search_result>
{web_search_result}
</web_search_result>

<question>
User: {question}

Agent:"""
