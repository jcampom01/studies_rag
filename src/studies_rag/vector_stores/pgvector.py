"""Vector store on PostgreSQL with the pgvector extension.

Layout in the database:
  vector_indexes   one row per embedding set: its name and manifest
  chunks_<name>    one table per set: chunk_id, text, metadata (jsonb), embedding (vector)

The extension must already exist in the database (it is set up by whoever
administers it); the tables are created here. Host and port come from the
options; database name, user and password from the POSTGRES_DB,
POSTGRES_USER and POSTGRES_PASSWORD environment variables.
"""
import re

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.types.json import Jsonb

from studies_rag.chunkers import Chunk
from studies_rag.config import require_env
from studies_rag.vector_stores.base import SearchResult, VectorStore

# Set names become part of table names: keep them to safe identifiers
NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def chunks_table(name: str) -> sql.Identifier:
    if not NAME_RE.match(name):
        raise ValueError(f"invalid set name {name!r}: use lowercase letters, digits and _")
    return sql.Identifier(f"chunks_{name}")


class PgVectorStore(VectorStore):
    def __init__(self, host: str = "localhost", port: int = 5432):
        # autocommit: each statement commits on its own; replace() opens its own transaction
        self.conn = psycopg.connect(
            host=host,
            port=port,
            dbname=require_env("POSTGRES_DB"),
            user=require_env("POSTGRES_USER"),
            password=require_env("POSTGRES_PASSWORD"),
            autocommit=True,
        )
        self._check_extension()
        register_vector(self.conn)  # numpy arrays <-> vector columns

    def _check_extension(self) -> None:
        row = self.conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
        if row is None:
            self.conn.close()
            raise RuntimeError("the 'vector' extension is not installed in this database "
                               "(an administrator must run: CREATE EXTENSION vector)")

    def replace(self, name: str, chunks: list[Chunk], vectors: np.ndarray, manifest: dict) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        table = chunks_table(name)
        dimension = int(vectors.shape[1])

        # One transaction: on any error everything is rolled back and the old set stays as it was
        with self.conn.transaction():
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS vector_indexes (
                    name       text PRIMARY KEY,
                    manifest   jsonb NOT NULL,
                    loaded_at  timestamptz NOT NULL DEFAULT now()
                )""")
            self.conn.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(table))
            self.conn.execute(sql.SQL("""
                CREATE TABLE {} (
                    chunk_id   text PRIMARY KEY,
                    text       text NOT NULL,
                    metadata   jsonb NOT NULL,
                    embedding  vector({}) NOT NULL
                )""").format(table, sql.Literal(dimension)))

            # COPY: bulk load, much faster than one INSERT per row
            copy_sql = sql.SQL("COPY {} (chunk_id, text, metadata, embedding) FROM STDIN WITH (FORMAT BINARY)")
            with self.conn.cursor() as cur, cur.copy(copy_sql.format(table)) as copy:
                copy.set_types(["text", "text", "jsonb", "vector"])
                for chunk, vector in zip(chunks, vectors):
                    copy.write_row((chunk.chunk_id, chunk.text, Jsonb(chunk.metadata), vector))

            # Indexes built after loading: faster than updating them row by row
            self.conn.execute(sql.SQL("CREATE INDEX ON {} USING hnsw (embedding vector_cosine_ops)").format(table))
            self.conn.execute(sql.SQL("CREATE INDEX ON {} USING gin (metadata jsonb_path_ops)").format(table))
            self.conn.execute(sql.SQL("ANALYZE {}").format(table))  # statistics for the query planner

            self.conn.execute("""
                INSERT INTO vector_indexes (name, manifest) VALUES (%s, %s)
                ON CONFLICT (name) DO UPDATE SET manifest = EXCLUDED.manifest, loaded_at = now()""",
                (name, Jsonb(manifest)))

    def manifest(self, name: str) -> dict | None:
        # The registry table doesn't exist until the first replace()
        if self.conn.execute("SELECT to_regclass('vector_indexes')").fetchone()[0] is None:
            return None
        row = self.conn.execute("SELECT manifest FROM vector_indexes WHERE name = %s", (name,)).fetchone()
        return row[0] if row else None

    def search(self, name: str, query_vector: np.ndarray, k: int = 5,
               filters: dict | None = None) -> list[SearchResult]:
        # <=> is the cosine distance (0 = same direction); similarity = 1 - distance
        where = sql.SQL("WHERE metadata @> %(filters)s") if filters else sql.SQL("")
        query = sql.SQL("""
            SELECT chunk_id, text, metadata, 1 - (embedding <=> %(query)s) AS score
            FROM {table}
            {where}
            ORDER BY embedding <=> %(query)s
            LIMIT %(k)s""").format(table=chunks_table(name), where=where)
        params = {"query": query_vector, "k": k, "filters": Jsonb(filters or {})}
        rows = self.conn.execute(query, params).fetchall()
        return [SearchResult(Chunk(chunk_id, text, metadata), float(score))
                for chunk_id, text, metadata, score in rows]

    def close(self) -> None:
        self.conn.close()
