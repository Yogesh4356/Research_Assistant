import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import get_llm
llm = get_llm()

observer_prompt = ChatPromptTemplate.from_template("""
You are an observer agent. Your job is to check if collected information is sufficient to answer the user's original query.

Original Query: {original_query}
Collected Results:
{results}

Respond in ONLY valid JSON:

If information is sufficient:
{{
  "is_sufficient": true,
  "missing_info": "",
  "additional_queries": []
}}

If information is NOT sufficient:
{{
  "is_sufficient": false,
  "missing_info": "what is missing",
  "additional_queries": [
    {{
      "id": 1,
      "query": "additional query needed",
      "agent": "rag or web_search",
      "depends_on": []
    }}
  ]
}}
""")

observer_chain = observer_prompt | llm | StrOutputParser()


def run_observer(original_query: str, results: list[dict]) -> dict:
    results_text = ""
    for i, r in enumerate(results, 1):
        results_text += f"\nResult {i} (from {r.get('agent', 'unknown')}):\n{r.get('answer', '')}\n"

    result = observer_chain.invoke({
        "original_query": original_query,
        "results": results_text
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
            "is_sufficient": True,
            "missing_info": "",
            "additional_queries": []
        }