from src.llm_handler import LLMHandler

print("Testing Groq API...")

llm = LLMHandler()

context = """
Artificial Intelligence (AI) is intelligence demonstrated by machines. 
The field was founded in 1956 at a conference at Dartmouth College.
"""

question = "When was AI founded?"
answer = llm.generate_answer(question, context)

print(f"\nQuestion: {question}")
print(f"Answer: {answer}")