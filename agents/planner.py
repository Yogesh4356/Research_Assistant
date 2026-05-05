import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import get_llm
llm = get_llm()

planner_prompt = ChatPromptTemplate.from_template("""
You are an intelligent query planner for an AI assistant.

Available agents:
- "rag": MUST use when document is uploaded and question relates to document content
- "web_search": use for current events, general knowledge, internet information
- "chitchat": use for greetings, casual talk only

STRICT RULES:
- If document is uploaded AND query asks about document content → ALWAYS use "rag"
- If query asks for facts NOT likely in document (rankings, news, statistics, current data) → "web_search" even if document uploaded
- Keywords that mean document: "document", "pdf", "file", "person", "name", "college", "present in"
- Keywords for web_search: "ranking", "rank", "news", "latest", "current", "price", "statistics", "how many", "when was"
- Only use "web_search" when no document is relevant OR query needs internet data
- "chitchat" only for greetings like "hi", "hello", "how are you"

Document uploaded: {has_document}
User Query: {query}

EXAMPLES:
Query: "what is name of person in document?" + Document: Yes → rag
Query: "what is college in document?" + Document: Yes → rag  
Query: "latest AI news?" + Document: Yes → web_search
Query: "hi" → chitchat

Respond in ONLY valid JSON:

If clarification needed:
{{
  "needs_clarification": true,
  "clarification_question": "your question here",
  "sub_queries": []
}}

If no clarification needed:
{{
  "needs_clarification": false,
  "clarification_question": "",
  "sub_queries": [
    {{
      "id": 1,
      "query": "specific sub query",
      "agent": "rag or web_search or chitchat",
      "depends_on": []
    }}
  ]
}}
""")

planner_chain = planner_prompt | llm | StrOutputParser()


def run_planner(query: str, has_document: bool = False) -> dict:
    result = planner_chain.invoke({
        "query": query,
        "has_document": "Yes" if has_document else "No"
    })

    result = result.strip()
    if "```json" in result:
        result = result.split("```json")[1].split("```")[0].strip()
    elif "```" in result:
        result = result.split("```")[1].split("```")[0].strip()

    try:
        return json.loads(result)
    except json.JSONDecodeError:
        return {
            "needs_clarification": False,
            "clarification_question": "",
            "sub_queries": [
                {
                    "id": 1,
                    "query": query,
                    "agent": "web_search",
                    "depends_on": []
                }
            ]
        }