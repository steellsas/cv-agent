from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue
)
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field
from typing import Optional
import uuid

COLLECTION_NAME = "user_profile"
VECTOR_SIZE = 1536

# ── Structured block model ──
class ProfileBlock(BaseModel):
    category: str                          # work_experience / project / education / tech_skill / soft_skill / personality / other
    text: str                              # normalized English summary
    source: list[str] = Field(default_factory=list)  # linkedin / github / conversation / old_cv
    confidence: str = "medium"            # low / medium / high
    metadata: dict = Field(default_factory=dict)     # company, role, period, etc.
    user_confirmed: bool = False          # confirmed by user via Fact-Checker


class VectorStore:
    def __init__(self, config: dict):
        self.client = QdrantClient(url=config["profile"]["vector_db_url"])
        self.collection_name = config["profile"]["collection_name"]
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self._ensure_collection()

    def _ensure_collection(self):
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
            )
            print(f"✅ Collection '{self.collection_name}' created")
        else:
            print(f"✅ Collection '{self.collection_name}' found")

    def save_block(self, block: ProfileBlock) -> str:
        """Saves a structured ProfileBlock to Qdrant"""
        vector = self.embeddings.embed_query(block.text)
        point_id = str(uuid.uuid4())

        self.client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "text": block.text,
                    "category": block.category,
                    "source": block.source,
                    "confidence": block.confidence,
                    "user_confirmed": block.user_confirmed,
                    **block.metadata
                }
            )]
        )
        print(f"  💾 [{block.category}] ({', '.join(block.source)}) {block.text[:60]}...")
        return point_id

    def search(self, query: str, category: str = None, top_k: int = 5) -> list[dict]:
        """Semantic search with optional category filter"""
        vector = self.embeddings.embed_query(query)

        search_filter = None
        if category:
            search_filter = Filter(must=[
                FieldCondition(key="category", match=MatchValue(value=category))
            ])

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
                "source": r.payload.get("source", []),
                "confidence": r.payload.get("confidence", "medium"),
                "user_confirmed": r.payload.get("user_confirmed", False),
                "score": r.score,
                "metadata": {k: v for k, v in r.payload.items()
                             if k not in ["text", "category", "source", "confidence", "user_confirmed"]}
            }
            for r in results.points
        ]

    def get_all(self, category: str = None, confirmed_only: bool = False) -> list[dict]:
        """Returns all blocks, optionally filtered"""
        must_conditions = []

        if category:
            must_conditions.append(
                FieldCondition(key="category", match=MatchValue(value=category))
            )
        if confirmed_only:
            must_conditions.append(
                FieldCondition(key="user_confirmed", match=MatchValue(value=True))
            )

        scroll_filter = Filter(must=must_conditions) if must_conditions else None

        results, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=scroll_filter,
            limit=100
        )

        return [
            {
                "text": r.payload["text"],
                "category": r.payload["category"],
                "source": r.payload.get("source", []),
                "confidence": r.payload.get("confidence", "medium"),
                "user_confirmed": r.payload.get("user_confirmed", False),
                "metadata": {k: v for k, v in r.payload.items()
                             if k not in ["text", "category", "source", "confidence", "user_confirmed"]}
            }
            for r in results
        ]

    def get_profile_summary(self) -> dict:
        """Returns profile completeness overview"""
        categories = ["work_experience", "project", "education",
                      "tech_skill", "soft_skill", "personality"]
        summary = {}
        for cat in categories:
            items = self.get_all(category=cat)
            summary[cat] = {
                "count": len(items),
                "sources": list(set(s for item in items for s in item["source"])),
                "confirmed": sum(1 for i in items if i["user_confirmed"])
            }
        return summary

    def clear_collection(self):
        """Clears all data"""
        self.client.delete_collection(self.collection_name)
        print(f"🗑️  Collection '{self.collection_name}' cleared")
        self._ensure_collection()