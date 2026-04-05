import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class LLMHandler:
    
    def __init__(self, model_name=None):
        api_key = os.getenv("GROQ_API_KEY")
        
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env file!")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        self.model = model_name or os.getenv("MODEL_NAME", "llama3-8b-8192")
        print(f" Using Groq API (FREE)")
        print(f" Model: {self.model}")
    
    def generate_answer(self, query: str, context: str) -> str:
        prompt = f"""You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

Question: {query}

Instructions:
- Answer the question using ONLY the information from the context above
- If the context doesn't contain enough information, say "I don't have enough information to answer this question."
- Be concise and specific
- Cite which document number you're using if relevant

Answer:"""

        try:
            print("\n🤖 Generating answer with Groq...")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided documents."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def generate_answer_streaming(self, query: str, context: str):
        prompt = f"""You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

Question: {query}

Answer:"""

        try:
            print("\n🤖 Generating answer with Groq (streaming)...")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500,
                stream=True
            )
            
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            yield f"Error: {str(e)}"


if __name__ == "__main__":
    llm = LLMHandler()
    
    test_context = """
    Python is a high-level programming language.
    It was created by Guido van Rossum in 1991.
    """
    
    answer = llm.generate_answer("When was Python created?", test_context)
    print(f"\nAnswer: {answer}")
