from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)
from langchain_openai import OpenAIEmbeddings
import uuid

COLLECTION_NAME = "user_profile"
VECTOR_SIZE = 1536  # OpenAI text-embedding-3-small dydis

class VectorStore:
    def __init__(self, config: dict):
        self.client = QdrantClient(url=config["profile"]["vector_db_url"])
        self.collection_name = config["profile"]["collection_name"]
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self._ensure_collection()

    def _ensure_collection(self):
        """Creates collection if it doesn't exist yet"""
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE
                )
            )
            print(f"✅ Collection '{self.collection_name}' created")
        else:
            print(f"✅ Collection '{self.collection_name}' found")

    def save(self, text: str, category: str, metadata: dict = {}):
        """
        Saves a text chunk with category and metadata
        
        Categories: work_experience, project, education, 
                    tech_skill, soft_skill, personality, other
        """
        vector = self.embeddings.embed_query(text)
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "text": text,
                "category": category,
                **metadata
            }
        )
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )
        print(f"💾 Saved [{category}]: {text[:60]}...")

    def search(self, query: str, category: str = None, top_k: int = 5) -> list[dict]:
        """
        Searches for relevant information semantically
        Optional: filter by category
        """
        vector = self.embeddings.embed_query(query)

        search_filter = None
        if category:
            search_filter = Filter(
                must=[FieldCondition(
                    key="category",
                    match=MatchValue(value=category)
                )]
            )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=search_filter,
            limit=top_k
        )

        return [
            {
                "text": r.payload["text"],
                "category": r.payload["category"],
                "score": r.score
            }
            for r in results.points
        ]

    def get_all(self, category: str = None) -> list[dict]:
        """Returns all saved information, optionally filtered by category"""
        scroll_filter = None
        if category:
            scroll_filter = Filter(
                must=[FieldCondition(
                    key="category",
                    match=MatchValue(value=category)
                )]
            )

        results, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=scroll_filter,
            limit=100
        )

        return [
            {
                "text": r.payload["text"],
                "category": r.payload["category"]
            }
            for r in results
        ]
    
    def clear_collection(self):
        """Clears all data from collection"""
        self.client.delete_collection(self.collection_name)
        print(f"🗑️  Collection '{self.collection_name}' cleared")
        self._ensure_collection()