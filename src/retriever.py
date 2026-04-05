from src.embeddings import EmbeddingManager

class Retriever:
    
    def __init__(self, embedding_manager: EmbeddingManager, top_k: int = 3):
        self.embedding_manager = embedding_manager
        self.top_k = top_k
    
    def retrieve(self, query: str) -> tuple:
        results = self.embedding_manager.search(query, top_k=self.top_k)
        
        if not results:
            return "", []
        
        context = "\n".join(
            f"[Document {i+1}]\n{result['content']}\n"
            for i, result in enumerate(results)
        )
        
        sources = [
            {
                'chunk_number': i + 1,
                'content': result['content'][:200] + "...",
                'source': result['metadata'].get('source', 'Unknown'),
                'similarity': round(result['similarity'], 3)
            }
            for i, result in enumerate(results)
        ]
        
        return context, sources