# agent/app/services/rag.py
from pydantic import BaseModel
from typing import Optional, List
import chromadb
from openai import OpenAI


class RetrievalResult(BaseModel):
    """Single retrieval result from the vector database."""

    id: str
    score: float
    text: str
    source: Optional[str] = None


class RAGPipeline:
    """RAG pipeline backed by ChromaDB and OpenAI embeddings.

    Provides methods to:
    - Connect to a persistent ChromaDB collection
    - Embed queries with OpenAI
    - Retrieve the most relevant text chunks for a given query
    - Format retrieved context for LLM prompts
    """

    def __init__(self, collection_name: str = "startup_mentor", db_path: str = "./chroma_db") -> None:
        """
        Initialize the RAG pipeline with ChromaDB persistence and OpenAI embeddings.

        Args:
            collection_name: Name of the Chroma collection to use.
            db_path: Path where ChromaDB will persist vectors/documents.
        """
        # Persistent Chroma client (saves vectors to disk)
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)

        # OpenAI client for embeddings
        self.embedder = OpenAI()

    async def retrieve(self, query: str, top_k: int = 4) -> List[RetrievalResult]:
        """
        Retrieve the top-k most relevant text chunks from Chroma for a given query.

        Steps:
        - Create an embedding for the input query using OpenAI.
        - Perform a similarity search against the Chroma collection.
        - Return results as a list of `RetrievalResult`.

        Args:
            query: User input string to search against stored documents.
            top_k: Number of results to return (default = 4).

        Returns:
            List[RetrievalResult]: Each containing id, similarity score, text, and optional source.
        """
        # Create embedding for the query
        query_embedding = self.embedder.embeddings.create(
            model="text-embedding-3-small",
            input=query
        ).data[0].embedding

        # Search in Chroma
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        # Convert to RetrievalResult[]
        return [
            RetrievalResult(
                id=result_id,
                score=score,
                text=document,
                source=(metadatas.get("source") if metadatas else None),
            )
            for result_id, score, document, metadatas in zip(
                results["ids"][0],
                results["distances"][0],
                results["documents"][0],
                results.get("metadatas", [[]])[0],
            )
        ]

    def format_context(self, results: List[RetrievalResult]) -> str:
        """
        Format retrieval results into a context string for LLM prompts.

        Args:
            results: List of `RetrievalResult` objects.

        Returns:
            str: Formatted context block with document ids, scores, and text.
        """
        if not results:
            return ""
        lines = [f"[doc:{r.id} score={r.score:.3f}] {r.text}" for r in results]
        return "\n".join(lines)
