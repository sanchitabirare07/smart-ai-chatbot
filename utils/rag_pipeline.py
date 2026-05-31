from __future__ import annotations
import os
import tempfile
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings


RAG_PROMPT = PromptTemplate(
    template="""You are a helpful AI assistant for a college or customer support portal.
Use the following retrieved context to answer the user's question accurately and concisely.
If the answer is not in the context, say "I don't have enough information to answer that."
Do not make up answers.

Context:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question"]
)


class RAGPipeline:
    def __init__(self, api_key: str, provider: str = "groq", model_name: str = "llama-3.1-8b-instant",
                 chunk_size: int = 500, top_k: int = 3):
        self.top_k = top_k
        self.vectorstore: Any = None
        self.chain: Any = None
        self.retriever: Any = None

        from langchain_groq import ChatGroq
        self.llm = ChatGroq(api_key=api_key, model_name="llama-3.1-8b-instant")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size * 0.15),
            separators=["\n\n", "\n", " ", ""]
        )

    def _load_uploaded_file(self, uploaded_file) -> list:
        suffix = os.path.splitext(uploaded_file.name)[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        try:
            loader = PyPDFLoader(tmp_path) if suffix == ".pdf" else TextLoader(tmp_path, encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = uploaded_file.name
            return docs
        finally:
            os.unlink(tmp_path)

    def build_from_directory(self, directory: str) -> None:
        import glob
        all_docs = []
        txt_files = glob.glob(f"{directory}/*.txt")
        pdf_files = glob.glob(f"{directory}/*.pdf")
        
        for filepath in txt_files:
            loader = TextLoader(filepath, encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = os.path.basename(filepath)
            all_docs.extend(docs)
            
        for filepath in pdf_files:
            loader = PyPDFLoader(filepath)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = os.path.basename(filepath)
            all_docs.extend(docs)
            
        if not all_docs:
            raise ValueError("No documents found in directory.")
            
        chunks = self.splitter.split_documents(all_docs)
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        self._build_chain()

    def _build_chain(self) -> None:
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k}
        )
        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | RAG_PROMPT
            | self.llm
            | StrOutputParser()
        )

    def query(self, question: str) -> dict:
        if self.chain is None:
            raise RuntimeError("Knowledge base not built.")
        answer = self.chain.invoke(question)
        docs = self.retriever.invoke(question)
        sources = list({doc.metadata.get("source", "unknown") for doc in docs})
        return {"answer": answer, "sources": sources}

    def save_vectorstore(self, path: str = "vectorstore/faiss_index") -> None:
        if self.vectorstore:
            self.vectorstore.save_local(path)

    def load_vectorstore(self, path: str = "vectorstore/faiss_index") -> None:
        self.vectorstore = FAISS.load_local(
            path, self.embeddings, allow_dangerous_deserialization=True
        )
        self._build_chain()
