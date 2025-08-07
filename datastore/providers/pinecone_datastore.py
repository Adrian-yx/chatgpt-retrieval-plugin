import os
from typing import Any, Dict, List, Optional
# import pinecone
from pinecone import Pinecone, ServerlessSpec
from pinecone import ServerlessSpec          # <- NUEVO: para crear índices Serverless
from tenacity import retry, wait_random_exponential, stop_after_attempt
import asyncio
from loguru import logger

from datastore.datastore import DataStore
from models.models import (
    DocumentChunk,
    DocumentChunkMetadata,
    DocumentChunkWithScore,
    DocumentMetadataFilter,
    QueryResult,
    QueryWithEmbedding,
    Source,
)
from services.date import to_unix_timestamp

# ──────── Variables de entorno ──────────────────────────────────────────────────
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")  # p. ej. us-east-1-aws
PINECONE_INDEX = os.environ.get("PINECONE_INDEX")              # p. ej. index1
assert PINECONE_API_KEY and PINECONE_ENVIRONMENT and PINECONE_INDEX

# ──────── Inicializa Pinecone ───────────────────────────────────────────────────
# pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
pc = Pinecone(api_key=PINECONE_API_KEY)

UPSERT_BATCH_SIZE = 100
EMBEDDING_DIMENSION = int(os.environ.get("EMBEDDING_DIMENSION", 256))


class PineconeDataStore(DataStore):
    def __init__(self):
        # ── 1. Crea el índice si no existe ──────────────────────────────────────
        if PINECONE_INDEX not in pc.list_indexes():
            fields_to_index = list(DocumentChunkMetadata.__fields__.keys())

            try:
                logger.info(
                    f"Creating serverless index {PINECONE_INDEX} "
                    f"with metadata config {fields_to_index}"
                )
                pc.create_index(
                    name=PINECONE_INDEX,
                    dimension=EMBEDDING_DIMENSION,
                    metric="cosine",                      # <- añade la métrica
                    metadata_config={"indexed": fields_to_index},
                    spec=ServerlessSpec(                  # <- NUEVO: modo Serverless
                        cloud="aws",
                        region="us-east-1",
                    ),
                )
                # pc.create_index(                 # <- BORRADO: pods clásicos
                #     PINECONE_INDEX,
                #     dimension=EMBEDDING_DIMENSION,
                #     metadata_config={"indexed": fields_to_index},
                # )
                self.index = pc.Index(PINECONE_INDEX)
                logger.info(f"Index {PINECONE_INDEX} created successfully")
            except Exception as e:
                logger.error(f"Error creating index {PINECONE_INDEX}: {e}")
                raise

        # ── 2. Conecta si ya existía ────────────────────────────────────────────
        else:
            try:
                logger.info(f"Connecting to existing index {PINECONE_INDEX}")
                self.index = pc.Index(PINECONE_INDEX)
                logger.info(f"Connected to index {PINECONE_INDEX} successfully")
            except Exception as e:
                logger.error(f"Error connecting to index {PINECONE_INDEX}: {e}")
                raise

    # ──────── Upsert ────────────────────────────────────────────────────────────
    @retry(wait=wait_random_exponential(1, 20), stop=stop_after_attempt(3))
    async def _upsert(self, chunks: Dict[str, List[DocumentChunk]]) -> List[str]:
        doc_ids: List[str] = []
        vectors = []

        for doc_id, chunk_list in chunks.items():
            doc_ids.append(doc_id)
            logger.info(f"Upserting document_id: {doc_id}")

            for chunk in chunk_list:
                pinecone_metadata = self._get_pinecone_metadata(chunk.metadata)
                pinecone_metadata["text"] = chunk.text
                pinecone_metadata["document_id"] = doc_id
                vectors.append((chunk.id, chunk.embedding, pinecone_metadata))

        # divide en lotes y envía
        for i in range(0, len(vectors), UPSERT_BATCH_SIZE):
            batch = vectors[i : i + UPSERT_BATCH_SIZE]
            try:
                logger.info(f"Upserting batch of size {len(batch)}")
                self.index.upsert(vectors=batch)
            except Exception as e:
                logger.error(f"Error upserting batch: {e}")
                raise
        return doc_ids

    # ──────── Query ─────────────────────────────────────────────────────────────
    @retry(wait=wait_random_exponential(1, 20), stop=stop_after_attempt(3))
    async def _query(self, queries: List[QueryWithEmbedding]) -> List[QueryResult]:

        async def _single_query(query: QueryWithEmbedding) -> QueryResult:
            pinecone_filter = self._get_pinecone_filter(query.filter)
            try:
                response = self.index.query(
                    vector=query.embedding,
                    top_k=query.top_k,
                    filter=pinecone_filter,
                    include_metadata=True,
                )
            except Exception as e:
                logger.error(f"Error querying index: {e}")
                raise

            results: List[DocumentChunkWithScore] = []
            for match in response.matches:
                md = match.metadata or {}
                md_no_text = {k: v for k, v in md.items() if k != "text"}

                if (
                    "source" in md_no_text
                    and md_no_text["source"] not in Source.__members__
                ):
                    md_no_text["source"] = None

                results.append(
                    DocumentChunkWithScore(
                        id=match.id,
                        score=match.score,
                        text=md.get("text", ""),
                        metadata=md_no_text,
                    )
                )
            return QueryResult(query=query.query, results=results)

        return await asyncio.gather(*(_single_query(q) for q in queries))

    # ──────── Delete ────────────────────────────────────────────────────────────
    @retry(wait=wait_random_exponential(1, 20), stop=stop_after_attempt(3))
    async def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[DocumentMetadataFilter] = None,
        delete_all: Optional[bool] = None,
    ) -> bool:
        if delete_all:
            self.index.delete(delete_all=True)
            return True

        pinecone_filter = self._get_pinecone_filter(filter)
        if pinecone_filter:
            self.index.delete(filter=pinecone_filter)

        if ids:
            self.index.delete(filter={"document_id": {"$in": ids}})
        return True

    # ──────── Helpers ───────────────────────────────────────────────────────────
    def _get_pinecone_filter(
        self, filter: Optional[DocumentMetadataFilter] = None
    ) -> Dict[str, Any]:
        if not filter:
            return {}
        pc_filter: Dict[str, Any] = {}
        for field, value in filter.dict().items():
            if value is None:
                continue
            if field == "start_date":
                pc_filter.setdefault("created_at", {})["$gte"] = to_unix_timestamp(value)
            elif field == "end_date":
                pc_filter.setdefault("created_at", {})["$lte"] = to_unix_timestamp(value)
            else:
                pc_filter[field] = value
        return pc_filter

    def _get_pinecone_metadata(
        self, metadata: Optional[DocumentChunkMetadata] = None
    ) -> Dict[str, Any]:
        if not metadata:
            return {}
        md: Dict[str, Any] = {}
        for field, value in metadata.dict().items():
            if value is None:
                continue
            md[field] = (
                to_unix_timestamp(value) if field == "created_at" else value
            )
        return md
