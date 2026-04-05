import os
import pickle
from typing import List

class EmbeddingManager:
    
    def __init__(self, collection_name="documents", persist_directory="./vector_store"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.db_file = os.path.join(persist_directory, f"{collection_name}.pkl")
        
        os.makedirs(persist_directory, exist_ok=True)
        
        if os.path.exists(self.db_file):
            with open(self.db_file, 'rb') as f:
                self.documents = pickle.load(f)
        else:
            self.documents = []
        
        self.collection = self
    
    def count(self):
        return len(self.documents)
    
    def add_documents(self, documents):
        for doc in documents:
            self.documents.append({
                'content': doc.page_content,
                'metadata': doc.metadata
            })
        
        self.save()
    
    def save(self):
        with open(self.db_file, 'wb') as f:
            pickle.dump(self.documents, f)
    
    def search(self, query: str, top_k: int = 3) -> List[dict]:
        if not self.documents:
            return []
        
        query_words = set(query.lower().split())
        
        scored = []
        for doc in self.documents:
            content_words = set(doc['content'].lower().split())
            overlap = len(query_words.intersection(content_words))
            total = len(query_words.union(content_words))
            score = overlap / total if total > 0 else 0
            scored.append((score, doc))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        
        return [
            {
                'content': doc['content'],
                'metadata': doc['metadata'],
                'similarity': round(score, 3)
            }
            for score, doc in scored[:top_k]
        ]
    
    def clear_database(self):
        self.documents = []
        if os.path.exists(self.db_file):
            os.remove(self.db_file)
            