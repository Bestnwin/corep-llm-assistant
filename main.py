from rag_pipeline import build_db, query_db
from langchain_openai import ChatOpenAI
import json


# Build vector database
build_db()

# Ask user
question = input("Ask something about capital reporting: ")

# Retrieve context
results = query_db(question)
context = "\n".join([r.page_content for r in results])

# Initialize LLM
llm = ChatOpenAI(temperature=0)

# Prompt
prompt = f"""
You are a regulatory reporting assistant.

Using the context below, produce structured JSON representing a simplified COREP template.

Context:
{context}

Return ONLY valid JSON in this format:

{{
 "template": "OwnFunds",
 "fields": [
   {{
     "code": "CET1",
     "value": "...",
     "justification": "Explain using context"
   }},
   {{
     "code": "RWA",
     "value": "...",
     "justification": "Explain using context"
   }}
 ]
}}
"""

response = llm.invoke(prompt)

# Print raw response
print("\nModel Output:\n")
print(response.content)

# Try parse JSON
try:
    parsed = json.loads(response.content)
    print("\nParsed OK ✅")
except:
    print("\nCould not parse JSON ❌")