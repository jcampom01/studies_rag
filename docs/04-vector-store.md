# Step 4 · Vector store

**English** · [Español](04-vector-store.es.md) · [← Embeddings](03-embeddings.md) · [Next: Retrieval + generation →](05-retrieval-generation.md)

Loads the chunks and their vectors into **PostgreSQL with the pgvector extension**, where they can be searched by similarity and filtered by metadata.

```
chunks_<chunker>.jsonl + embeddings/<chunker>/  ──index──►  PostgreSQL: table chunks_<chunker>
```

## Why a database

With a few thousand vectors, a NumPy dot product over the whole matrix takes milliseconds, so speed is not the reason. A database adds:

- **A service.** Any process (the CLI today, an HTTP API tomorrow) queries over the network, without loading files into memory.
- **Text, metadata and vector in one place.** A search returns the passage and where it comes from, not a row number.
- **Metadata filters**, for example "only this collection" or "only this document".
- **Transactions.** A set is replaced all at once or not at all.
- **Scale.** Approximate indexes (HNSW) for when the corpus grows to hundreds of thousands of chunks.

**pgvector** was chosen over dedicated vector databases (Qdrant, Chroma, Milvus…) because many production systems already run PostgreSQL. Vectors then live next to the rest of the application's data, with the same backups and tooling, and there is no extra service to operate.

pgvector is an *extension* of PostgreSQL. It adds:

- the `vector(n)` type;
- distance operators: `<=>` cosine, `<#>` negative inner product, `<->` Euclidean;
- approximate nearest-neighbour indexes (HNSW, IVFFlat).

## The database service

`docker-compose.yml` runs `pgvector/pgvector:0.8.7-pg17` (PostgreSQL 17 with the extension built in):

- **Credentials** come from `.env` (`POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`).
- **The port** is published on `127.0.0.1` only.
- **The data** lives in a named volume (`pgdata`) and survives the container being recreated.
- **The extension** is created by `db/init/01-extensions.sql`. PostgreSQL runs the scripts in `/docker-entrypoint-initdb.d` once, when the database is first created.

  This mirrors production: extensions need elevated privileges and are set up by whoever administers the database. The application only checks that the extension exists and fails with a clear message if it doesn't.

## Schema

```sql
-- one row per embedding set: plays the role of manifest.json
CREATE TABLE vector_indexes (
    name       text PRIMARY KEY,           -- "structured", "langchain"
    manifest   jsonb NOT NULL,
    loaded_at  timestamptz NOT NULL DEFAULT now()
);

-- one table per embedding set
CREATE TABLE chunks_structured (
    chunk_id   text PRIMARY KEY,
    text       text NOT NULL,
    metadata   jsonb NOT NULL,
    embedding  vector(1024) NOT NULL       -- dimension taken from the vectors
);
CREATE INDEX ON chunks_structured USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON chunks_structured USING gin (metadata jsonb_path_ops);
```

- **The tables are created by the code**, because they depend on the data: the dimension and the set names. The extension is infrastructure, so it is created by the init script.
- **Metadata goes in one `jsonb` column**, not one column per field. That keeps the schema generic: any adapter's fields work without migrations.
- **One table per set** keeps the sets apart, and each HNSW index only contains its own vectors.
- **The manifest is stored in the database.** The query side may not have access to the files of step 3, but it can still check that it embeds questions with the same model, revision and instruction.

## Loading: one transaction

`replace()` does everything inside one transaction:

```
CREATE TABLE IF NOT EXISTS vector_indexes
DROP TABLE IF EXISTS chunks_<name>
CREATE TABLE chunks_<name> (... vector(<dimension>))
COPY ... FROM STDIN (FORMAT BINARY)          bulk load, much faster than one INSERT per row
CREATE INDEX ... hnsw / gin                  built after loading: faster than updating them row by row
ANALYZE                                      statistics for the query planner
INSERT INTO vector_indexes ... ON CONFLICT DO UPDATE
COMMIT
```

In PostgreSQL even `CREATE` and `DROP TABLE` are transactional, so an error anywhere rolls everything back and the previous set stays as it was.

Other details:

- **Table names can't be query parameters.** Set names are validated against `^[a-z][a-z0-9_]*$` and quoted with `psycopg.sql.Identifier`, which rules out SQL injection.
- **The connection uses autocommit.** Searches don't leave idle open transactions; only `replace()` opens one.
- **`index` skips a set** when the manifest stored in the database equals the one on disk.

## Search

```sql
SELECT chunk_id, text, metadata, 1 - (embedding <=> %(query)s) AS score
FROM chunks_structured
WHERE metadata @> %(filters)s           -- only when filters are given
ORDER BY embedding <=> %(query)s
LIMIT %(k)s
```

- **`<=>` is the cosine distance**: 0 means the same direction. The score returned is `1 - distance`, the cosine similarity.
- **`ORDER BY … <=> … LIMIT k`** is the shape that lets PostgreSQL use the HNSW index.
- **Filters use JSONB containment.** `{"collection": "course_a"}` matches the chunks whose metadata contains that pair, and the GIN index speeds it up.

The contract:

```python
class VectorStore(ABC):
    def replace(self, name, chunks, vectors, manifest) -> None: ...   # all or nothing
    def manifest(self, name) -> dict | None: ...
    def search(self, name, query_vector, k=5, filters=None) -> list[SearchResult]: ...
    def close(self) -> None: ...                                       # also usable with `with`
```

## What was measured

On the development corpus (about 2,700 and 2,900 chunks per set):

- **Same results as an exact NumPy search.** Five test questions gave the same top-5 in 9 of 10 cases. The difference in the remaining case comes from the vector norms (see [Embeddings](03-embeddings.md#running-it)), not from the index: PostgreSQL computes the exact cosine.
- **The HNSW index is not used yet.** `EXPLAIN` shows a sequential scan plus a sort. With a few thousand rows, the planner estimates that reading the whole table is cheaper than using the index, and it is right. The search is therefore exact. HNSW takes over as the table grows.
- **Filtered search returns k results.** Because of the sequential scan, a strict filter (37 matching rows, k = 20) returned 20 results. With the HNSW index in use, PostgreSQL filters *after* the approximate search and can return fewer than k rows. pgvector 0.8 solves it with `SET hnsw.iterative_scan = strict_order`.
- **JSONB filters are type-strict.** `{"doc_number": "9"}` matches nothing when the stored value is the number `9`.

## Running it

```bash
docker compose up -d db
PYTHONPATH=src .venv/bin/python -m studies_rag.index
```

Settings:

- the `index` section: `vector_store` and `vector_store_options` (`host`, `port`; use `db` as the host from inside the Compose network);
- the `embed` section, which says where the chunks and embeddings are.

To look at the data:

```bash
docker compose exec db psql -U rag -d rag -c "\dt"
docker compose exec db psql -U rag -d rag -c "SELECT name, manifest->>'count', loaded_at FROM vector_indexes;"
```

(Replace `rag` with the user and database you set in `.env`.)
