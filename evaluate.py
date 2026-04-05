from src.document_processor import DocumentProcessor
from src.embeddings import EmbeddingManager
from src.retriever import Retriever
from src.llm_handler import LLMHandler
import os


class RAGEvaluator:
    
    def __init__(self):
        self.doc_processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
        self.embedding_manager = EmbeddingManager(collection_name="eval_documents")
        self.retriever = Retriever(self.embedding_manager, top_k=3)
        self.llm = LLMHandler()
    
    def setup_test_documents(self):
        os.makedirs("test_documents", exist_ok=True)
        
        ml_content = """Machine Learning Basics

Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It was first coined by Arthur Samuel in 1959.

There are three main types of machine learning:
1. Supervised Learning - Learning from labeled data
2. Unsupervised Learning - Finding patterns in unlabeled data  
3. Reinforcement Learning - Learning through trial and error

Popular machine learning algorithms include:
- Linear Regression
- Decision Trees
- Neural Networks
- Support Vector Machines

Machine learning is used in many applications including image recognition, natural language processing, recommendation systems, and autonomous vehicles.

Key pioneers in machine learning include:
- Arthur Samuel (coined the term in 1959)
- Geoffrey Hinton (deep learning)
- Yann LeCun (convolutional neural networks)
- Andrew Ng (popularized ML education)
"""
        
        with open("test_documents/machine_learning.txt", "w", encoding="utf-8") as f:
            f.write(ml_content)
        
        python_content = """Python Programming Language

Python is a high-level, interpreted programming language created by Guido van Rossum. It was first released in 1991.

Key Features of Python:
- Easy to learn and read
- Extensive standard library
- Cross-platform compatibility
- Large community support
- Dynamic typing

Python is widely used for:
1. Web Development
2. Data Science
3. Machine Learning
4. Automation
5. Scientific Computing

Popular Python frameworks and libraries:
- Django and Flask
- Pandas
- NumPy
- TensorFlow and PyTorch
- Matplotlib
"""
        
        with open("test_documents/python.txt", "w", encoding="utf-8") as f:
            f.write(python_content)
        
        ds_content = """Data Science Overview

Data science is an interdisciplinary field that uses scientific methods to extract insights from data.

Key Components:
1. Statistics
2. Programming
3. Domain Knowledge
4. Visualization
5. Machine Learning

Process:
- Data Collection
- Data Cleaning
- Analysis
- Modeling
- Evaluation
"""
        
        with open("test_documents/data_science.txt", "w", encoding="utf-8") as f:
            f.write(ds_content)
        
        docs1 = self.doc_processor.load_document("test_documents/machine_learning.txt")
        self.embedding_manager.add_documents(docs1)
        
        docs2 = self.doc_processor.load_document("test_documents/python.txt")
        self.embedding_manager.add_documents(docs2)
        
        docs3 = self.doc_processor.load_document("test_documents/data_science.txt")
        self.embedding_manager.add_documents(docs3)
    
    def run_evaluation(self):
        test_cases = [
            {"id": 1, "question": "Who created Python?", "expected_keywords": ["Guido van Rossum"]},
            {"id": 2, "question": "What are the main types of machine learning?", "expected_keywords": ["supervised", "unsupervised", "reinforcement"]},
            {"id": 3, "question": "When was Python first released?", "expected_keywords": ["1991"]},
            {"id": 4, "question": "What is Python used for?", "expected_keywords": ["web development", "data science", "machine learning"]},
            {"id": 5, "question": "Who coined the term machine learning?", "expected_keywords": ["Arthur Samuel", "1959"]},
            {"id": 6, "question": "What are the key components of data science?", "expected_keywords": ["statistics", "programming", "machine learning", "visualization"]},
            {"id": 7, "question": "Name some Python frameworks", "expected_keywords": ["Django", "Flask"]},
            {"id": 8, "question": "What is the data science process?", "expected_keywords": ["collection", "cleaning", "analysis", "modeling"]}
        ]
        
        results = []
        
        for test in test_cases:
            context, _ = self.retriever.retrieve(test['question'])
            
            if not context:
                results.append({'test_id': test['id'], 'passed': False})
                continue
            
            answer = self.llm.generate_answer(test['question'], context)
            answer_lower = answer.lower()
            
            found = sum(1 for kw in test['expected_keywords'] if kw.lower() in answer_lower)
            pass_rate = found / len(test['expected_keywords'])
            
            results.append({
                'test_id': test['id'],
                'passed': pass_rate >= 0.5
            })
        
        return results
    
    def cleanup(self):
        self.embedding_manager.clear_database()


def main():
    evaluator = RAGEvaluator()
    evaluator.setup_test_documents()
    input()
    evaluator.run_evaluation()
    
    if input().strip().lower() == 'yes':
        evaluator.cleanup()


if __name__ == "__main__":
    main()