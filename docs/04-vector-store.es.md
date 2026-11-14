# Paso 4 · Base de datos vectorial

[English](04-vector-store.md) · **Español** · [← Embeddings](03-embeddings.es.md) · [Siguiente: Recuperación + generación →](05-retrieval-generation.es.md)

Carga los chunks y sus vectores en **PostgreSQL con la extensión pgvector**, donde se pueden buscar por similitud y filtrar por metadatos.

```
chunks_<chunker>.jsonl + embeddings/<chunker>/  ──index──►  PostgreSQL: tabla chunks_<chunker>
```

## Por qué una base de datos

Con unos pocos miles de vectores, un producto escalar de NumPy sobre toda la matriz tarda milisegundos, así que la velocidad no es el motivo. Una base de datos aporta:

- **Un servicio.** Cualquier proceso (hoy la terminal, mañana una API HTTP) consulta por red, sin cargar ficheros en memoria.
- **Texto, metadatos y vector en un solo sitio.** Una búsqueda devuelve el pasaje y su procedencia, no un número de fila.
- **Filtros por metadatos**, por ejemplo "solo esta colección" o "solo este documento".
- **Transacciones.** Un conjunto se sustituye entero o no se toca.
- **Escala.** Índices aproximados (HNSW) para cuando el corpus crezca a cientos de miles de chunks.

Se eligió **pgvector** frente a bases de datos vectoriales dedicadas (Qdrant, Chroma, Milvus…) porque muchos sistemas en producción ya usan PostgreSQL. Así los vectores viven junto al resto de datos de la aplicación, con las mismas copias de seguridad y herramientas, y no hay un servicio más que mantener.

pgvector es una *extensión* de PostgreSQL. Añade:

- el tipo `vector(n)`;
- operadores de distancia: `<=>` coseno, `<#>` producto escalar negativo, `<->` euclídea;
- índices aproximados de vecinos más cercanos (HNSW, IVFFlat).

## El servicio de base de datos

`docker-compose.yml` ejecuta `pgvector/pgvector:0.8.7-pg17` (PostgreSQL 17 con la extensión incluida):

- **Las credenciales** vienen del `.env` (`POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`).
- **El puerto** solo se publica en `127.0.0.1`.
- **Los datos** viven en un volumen con nombre (`pgdata`) y sobreviven a que se recree el contenedor.
- **La extensión** la crea `db/init/01-extensions.sql`. PostgreSQL ejecuta los scripts de `/docker-entrypoint-initdb.d` una sola vez, cuando se crea la base de datos.

  Así se hace en producción: las extensiones necesitan privilegios elevados y las instala quien administra la base de datos. La aplicación solo comprueba que la extensión existe y, si no, falla con un mensaje claro.

## Esquema

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

- **Las tablas las crea el código**, porque dependen de los datos: la dimensión y los nombres de los conjuntos. La extensión es infraestructura, así que la crea el script de inicio.
- **Los metadatos van en una sola columna `jsonb`**, no en una columna por campo. Así el esquema es genérico: los campos de cualquier adaptador funcionan sin migraciones.
- **Una tabla por conjunto** mantiene los conjuntos separados, y cada índice HNSW solo contiene sus propios vectores.
- **El manifest se guarda en la base de datos.** El lado que consulta puede no tener acceso a los ficheros del paso 3, pero aun así puede comprobar que embebe las preguntas con el mismo modelo, revisión e instrucción.

## La carga: una sola transacción

`replace()` lo hace todo dentro de una transacción:

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

En PostgreSQL incluso `CREATE` y `DROP TABLE` son transaccionales, así que un error en cualquier punto lo deshace todo y el conjunto anterior queda intacto.

Otros detalles:

- **Los nombres de tabla no pueden ser parámetros de una consulta.** Los nombres de conjunto se validan con `^[a-z][a-z0-9_]*$` y se escapan con `psycopg.sql.Identifier`, lo que descarta la inyección SQL.
- **La conexión usa autocommit.** Las búsquedas no dejan transacciones abiertas sin usar; solo `replace()` abre una.
- **`index` se salta un conjunto** cuando el manifest guardado en la base de datos es igual al del disco.

## Búsqueda

```sql
SELECT chunk_id, text, metadata, 1 - (embedding <=> %(query)s) AS score
FROM chunks_structured
WHERE metadata @> %(filters)s           -- only when filters are given
ORDER BY embedding <=> %(query)s
LIMIT %(k)s
```

- **`<=>` es la distancia coseno**: 0 significa la misma dirección. El score que se devuelve es `1 - distancia`, la similitud coseno.
- **`ORDER BY … <=> … LIMIT k`** es la forma que permite a PostgreSQL usar el índice HNSW.
- **Los filtros usan la contención de JSONB.** `{"collection": "course_a"}` encuentra los chunks cuyos metadatos contienen ese par, y el índice GIN lo acelera.

El contrato:

```python
class VectorStore(ABC):
    def replace(self, name, chunks, vectors, manifest) -> None: ...   # all or nothing
    def manifest(self, name) -> dict | None: ...
    def search(self, name, query_vector, k=5, filters=None) -> list[SearchResult]: ...
    def close(self) -> None: ...                                       # also usable with `with`
```

## Qué se midió

En el corpus de desarrollo (unos 2.700 y 2.900 chunks por conjunto):

- **Los mismos resultados que una búsqueda exacta con NumPy.** Cinco preguntas de prueba dieron el mismo top-5 en 9 de 10 casos. La diferencia en el caso restante se debe a las normas de los vectores (ver [Embeddings](03-embeddings.es.md)), no al índice: PostgreSQL calcula el coseno exacto.
- **Todavía no se usa el índice HNSW.** `EXPLAIN` muestra un recorrido secuencial más una ordenación. Con unos pocos miles de filas, el planificador estima que leer toda la tabla es más barato que usar el índice, y tiene razón. Por tanto, la búsqueda es exacta. HNSW entra en juego cuando la tabla crece.
- **La búsqueda con filtros devuelve k resultados.** Por el recorrido secuencial, un filtro estricto (37 filas que lo cumplen, k = 20) devolvió 20 resultados. Con el índice HNSW en uso, PostgreSQL filtra *después* de la búsqueda aproximada y puede devolver menos de k filas. pgvector 0.8 lo resuelve con `SET hnsw.iterative_scan = strict_order`.
- **Los filtros JSONB son estrictos con los tipos.** `{"doc_number": "9"}` no encuentra nada cuando el valor guardado es el número `9`.

## Ejecución

```bash
docker compose up -d db
PYTHONPATH=src .venv/bin/python -m studies_rag.index
```

Configuración:

- la sección `index`: `vector_store` y `vector_store_options` (`host`, `port`; desde dentro de la red de Compose el host es `db`);
- la sección `embed`, que dice dónde están los chunks y los embeddings.

Para ver los datos:

```bash
docker compose exec db psql -U rag -d rag -c "\dt"
docker compose exec db psql -U rag -d rag -c "SELECT name, manifest->>'count', loaded_at FROM vector_indexes;"
```

(Cambia `rag` por el usuario y la base de datos que pusiste en el `.env`.)
