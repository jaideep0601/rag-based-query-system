import os
from src.document_processor import DocumentProcessor
from src.embeddings import EmbeddingManager
from src.retriever import Retriever
from src.llm_handler import LLMHandler


class RAGSystem:
    
    def __init__(self):
        print("="*60)
        print("Initializing RAG System...")
        print("="*60)
        
        self.doc_processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
        self.embedding_manager = EmbeddingManager()
        self.retriever = Retriever(self.embedding_manager, top_k=3)
        self.llm = LLMHandler()
        
        print("\n System ready!\n")
    
    def upload_document(self, file_path: str):
        try:
            print(f"\n Processing document: {file_path}")
            chunks = self.doc_processor.load_document(file_path)
            self.embedding_manager.add_documents(chunks)
            print(f" Document uploaded successfully!")
        except Exception as e:
            print(f" Error: {str(e)}")
    
    def ask_question(self, query: str, use_streaming=False):
        try:
            print("\n" + "="*60)
            print(f" Question: {query}")
            print("="*60)
            
            if self.embedding_manager.collection.count() == 0:
                print("  No documents found! Please upload a document first.")
                return
            
            context, sources = self.retriever.retrieve(query)
            
            if not context:
                print(" No relevant information found.")
                return
            
            print("\n" + "-"*60)
            print(" ANSWER:")
            print("-"*60)
            
            if use_streaming:
                for chunk in self.llm.generate_answer_streaming(query, context):
                    print(chunk, end='', flush=True)
                print()
            else:
                answer = self.llm.generate_answer(query, context)
                print(answer)
            
            print("\n" + "-"*60)
            print(" SOURCES:")
            print("-"*60)
            for source in sources:
                print(f"  • Chunk {source['chunk_number']} (similarity: {source['similarity']})")
                print(f"    Preview: {source['content']}")
                print()
                
        except Exception as e:
            print(f"Error: {str(e)}")
    
    def clear_database(self):
        self.embedding_manager.clear_database()
        print("✓ Database cleared!")
    
    def show_stats(self):
        count = self.embedding_manager.collection.count()
        print(f"\n Documents in database: {count} chunks")


def main():
    rag = RAGSystem()
    
    print("\n" + "="*60)
    print("   Welcome to LLM+RAG Question Answering System")
    print("="*60)
    
    while True:
        print("\n" + "-"*60)
        print("OPTIONS:")
        print("  1. Upload document")
        print("  2. Ask question")
        print("  3. Show stats")
        print("  4. Clear database")
        print("  5. Exit")
        print("-"*60)
        
        try:
            choice = input("\nEnter your choice (1-5): ").strip()
        except KeyboardInterrupt:
            print("\n\n Goodbye!")
            break
        
        if choice == "1":
            file_path = input("Enter document path: ").strip()
            rag.upload_document(file_path)
            
        elif choice == "2":
            query = input("Enter your question: ").strip()
            if query:
                rag.ask_question(query)
            
        elif choice == "3":
            rag.show_stats()
            
        elif choice == "4":
            confirm = input("Are you sure? (yes/no): ").strip().lower()
            if confirm == "yes":
                rag.clear_database()
        
        elif choice == "5":
            print("\n Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
