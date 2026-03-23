import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI

# 🔑 Add your API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")git init

def load_and_process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    return chunks

def create_vector_db(chunks):
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(chunks, embeddings)
    return db

def ask_question(db, query):
    docs = db.similarity_search(query)

    llm = ChatOpenAI()
    response = llm.predict(
        f"Answer the question using only the context below:\n{docs}\n\nQuestion: {query}"
    )

    return response

def main():
    print("📄 Loading document...")
    chunks = load_and_process_pdf("data/sample.pdf")

    print("⚡ Creating vector database...")
    db = create_vector_db(chunks)

    while True:
        query = input("\n❓ Ask a question (or type 'exit'): ")

        if query.lower() == "exit":
            break

        answer = ask_question(db, query)
        print("\n✅ Answer:", answer)

if __name__ == "__main__":
    main()