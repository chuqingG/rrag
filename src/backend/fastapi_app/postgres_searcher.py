from typing import Optional, Union

import numpy as np
from sqlalchemy import Float, Integer, String, column, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from fastapi_app.embeddings import compute_text_embedding
from fastapi_app.postgres_models import Item
from visual_bge.modeling import Visualized_BGE

class PostgresSearcher:
    def __init__(
        self,
        db_session: AsyncSession,
        embedding_column: str,
        embedding_model: Visualized_BGE,
    ):
        self.db_session = db_session
        self.embedding_model = embedding_model
        self.embedding_column = embedding_column

    def build_filter_clause(self, filters) -> tuple[str, str]:
        '''
        TODO(chuqing)
        '''
        if filters is None:
            return "", ""
        filter_clauses = []
        for filter in filters:
            if isinstance(filter["value"], str):
                filter["value"] = f"'{filter['value']}'"
            filter_clauses.append(f"{filter['column']} {filter['operator']} {filter['value']}")
        filter_clause = " AND ".join(filter_clauses)
        if len(filter_clause) > 0:
            return f"WHERE {filter_clause}", f"AND {filter_clause}"
        return "", ""

    async def search(
        self, query_text: Optional[str], query_vector: list[float], top: int = 5, filters: Optional[list[dict]] = None
    ):
        filter_clause_where, filter_clause_and = self.build_filter_clause(filters)
        table_name = Item.__tablename__
        vector_query = f"""
            SELECT asin, RANK () OVER (ORDER BY {self.embedding_column} <=> :embedding) AS rank
                FROM {table_name}
                {filter_clause_where}
                ORDER BY {self.embedding_column} <=> :embedding
                LIMIT 30
            """
        if query_text is not None:
            fulltext_query = f"""
                SELECT asin, RANK () OVER (ORDER BY ts_rank_cd(to_tsvector('english', title), plainto_tsquery('english', '{query_text}')) DESC) AS rank
                    FROM {table_name}
                    WHERE to_tsvector('english', title) @@ plainto_tsquery('english', '{query_text}') {filter_clause_and}
                    ORDER BY ts_rank_cd(to_tsvector('english', title), plainto_tsquery('english', '{query_text}')) DESC
                    LIMIT 30
            """
        else:
            fulltext_query = ""

        hybrid_query = f"""
        WITH vector_search AS (
            {vector_query}
        ),
        fulltext_search AS (
            {fulltext_query}
        )
        SELECT
            COALESCE(vector_search.asin, fulltext_search.asin) AS asin,
            COALESCE(1.0 / (:k + vector_search.rank), 0.0) +
            COALESCE(1.0 / (:k + fulltext_search.rank), 0.0) AS score
        FROM vector_search
        FULL OUTER JOIN fulltext_search ON vector_search.asin = fulltext_search.asin
        ORDER BY score DESC
        LIMIT 20
        """

        if query_text is not None and len(query_vector) > 0:
            sql = text(hybrid_query).columns(
                column("asin", String),   
                column("score", Float)   
            )
            results = (
                await self.db_session.execute(
                    sql,
                    {"embedding": np.array(query_vector), "query": query_text, "k": 60},
                )
            ).fetchall()
        elif len(query_vector) > 0:
            sql = text(vector_query).columns(
                column("asin", String),  
                column("rank", Integer)   
            )
            results = (
                await self.db_session.execute(
                    sql,
                    {"embedding": np.array(query_vector)},
                )
            ).fetchall()
        elif query_text is not None:
            sql = text(fulltext_query).columns(
                column("asin", String),   
                column("rank", Integer)    
            )
            results = (
                await self.db_session.execute(
                    sql
                )
            ).fetchall()
        else:
            raise ValueError("Both query text and query vector are empty")

        # Convert results to SQLAlchemy models
        row_models = []
        for asin, _ in results[:top]:
            item = await self.db_session.execute(select(Item).where(Item.asin == asin))
            row_models.append(item.scalar())
        return row_models

    async def search_and_embed(
        self,
        query_text: Optional[str] = None,
        top: int = 5,
        enable_vector_search: bool = False,
        enable_text_search: bool = False,
        filters: Optional[list[dict]] = None,
    ) -> list[Item]:
        
        # calculate embedding
        vector: list[float] = []
        if enable_vector_search and query_text is not None:
            vector = await compute_text_embedding(
                query_text,
                self.embedding_model,
            )
        if not enable_text_search:
            query_text = None

        return await self.search(query_text, vector, top, filters)
