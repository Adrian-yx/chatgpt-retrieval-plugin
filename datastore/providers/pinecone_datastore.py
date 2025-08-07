import os
import asyncio
from typing import Any, Dict, List, Optional

from pinecone import Pinecone, ServerlessSpec          # SDK v3
# from pinecone import ServerlessSpec                  # --- BORRAR ---
# import pinecone                                      # --- BORRAR ---
from tenacity import retry, wait_random_exponential, stop_after_attempt
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

# ────────────────────────── Variables de entorno ────────────────────────────
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX   = os.environ.get("PINECONE_INDEX", "index1")
assert PINECONE_API_KEY and PINECONE_INDEX

# ─────────────────────────── Inicializa cliente v3 ──────────────────────────
pc = Pinecone(api_key=PINECONE_API_KEY)

UPSERT_BATCH_SIZE   = 100
EMBEDDING_DIMENSION = int(os.environ.get("EMBEDDING_DIMENSION", 256))

# ───────────────────────────── Clase Datastore ──────────────────────────────
class PineconeDataStore(DataStore):
    def __init__(self) -> None:
        index_names = pc.list_indexes().names()        # v3 devuelve un IndexList
        if PINECONE_INDEX not in index_names:
            self._create_serverless_index()
        else:
            logger.info(f"Connecting to existing index {PINECONE_INDEX}")
        self.index = pc.Index(PINECONE_INDEX)

    # --------------------------------------------------------------------- #
    # 1) Crear índice serverless y, después, configurar los metadatos
    # --------------------------------------------------------------------- #
    def _create_serverless_index(self) -> None:
        fields_to_index = list(DocumentChunkMetadata.__fields__.keys())

        try:
            logger.info(f"Creating serverless index {PINECONE_INDEX}")
            pc.create_index(
                name       = PINECONE_INDEX,
                dimension  = EMBEDDING_DIMENSION,
                metric     = "cosine",
                spec       = ServerlessSpec(cloud="aws", region="us-east-1"),
                # metadata_config={"indexed": fields_to_index},      # --- BORRAR ---
            )
            # ahora sí, activamos filtrado por metadata
            logger.info(f"Configuring metadata for {PINECONE_INDEX}: {fields_to_index}")
            pc.configure_index(
                name            = PINECONE_INDEX,
                metadata_config = {"indexed": fields_to_index},
            )
        except Exception as e:
            logger.error(f"Error creating/configuring index {PINECONE_INDEX}: {e}")
            raise

    # --------------------------------------------------------------------- #
    # 2) Upsert
    # --------------------------------------------------------------------- #
    @retry(wait=wait_random_exponential(1, 20), stop=stop_after_attempt(3))
    async def _upsert(self, chunks: Dict[str, List[DocumentChunk]]) -> List[str]:
        doc_ids: List[str] = []
        vectors: List[tuple] = []

        for doc_id, chunk_list in chunks.items():
            doc_ids.append(doc_id)
            for chunk in chunk_list:
                md = self._get_pinecone_metadata(chunk.metadata)
                md.update({"text": chunk.text, "document_id": doc_id})
                vectors.append((chunk.id, chunk.embedding, md))

        for i in range(0, len(vectors), UPSERT_BATCH_SIZE):
            batch = vectors[i : i + UPSERT_BATCH_SIZE]
            logger.info(f"Upserting batch of {len(batch)} vectors")
            self.index.upsert(vectors=batch)
        return doc_ids

    # --------------------------------------------------------------------- #
    # 3) Query
    # --------------------------------------------------------------------- #
    @retry(wait=wait_random_exponential(1, 20), stop=stop_after_attempt(3))
    async def _query(self, queries: List[QueryWithEmbedding]) -> List[QueryResult]:

        async def _single(query: QueryWithEmbedding) -> QueryResult:
            response = self.index.query(
                vector           = query.embedding,
                top_k            = query.top_k,
                filter           = self._get_pinecone_filter(query.filter),
                include_metadata = True,
            )

            results: List[DocumentChunkWithScore] = []
            for match in response.matches:
                md      = match.metadata or {}
                md_no_t = {k: v for k, v in md.items() if k != "text"}

                if "source" in md_no_t and md_no_t["source"] not in Source.__members__:
                    md_no_t["source"] = None

                results.append(
                    DocumentChunkWithScore(
                        id       = match.id,
                        score    = match.score,
                        text     = md.get("text", ""),
                        metadata = md_no_t,
                    )
                )
            return QueryResult(query=query.query, results=results)

        return await asyncio.gather(*(_single(q) for q in queries))

    # --------------------------------------------------------------------- #
    # 4) Delete
    # --------------------------------------------------------------------- #
    @retry(wait=wait_random_exponential(1, 20), stop=stop_after_attempt(3))
    async def delete(                     # type: ignore[override]
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[DocumentMetadataFilter] = None,
        delete_all: Optional[bool] = None,
    ) -> bool:
        if delete_all:
            self.index.delete(delete_all=True)
            return True

        pc_filter = self._get_pinecone_filter(filter)
        if pc_filter:
            self.index.delete(filter=pc_filter)

        if ids:
            self.index.delete(filter={"document_id": {"$in": ids}})
        return True

    # --------------------------------------------------------------------- #
    # 5) Helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _get_pinecone_filter(
        filter: Optional[DocumentMetadataFilter] = None
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

    @staticmethod
    def _get_pinecone_metadata(
        metadata: Optional[DocumentChunkMetadata] = None
    ) -> Dict[str, Any]:
        if not metadata:
            return {}
        md: Dict[str, Any] = {}
        for field, value in metadata.dict().items():
            if value is None:
                continue
            md[field] = to_unix_timestamp(value) if field == "created_at" else value
        return md
