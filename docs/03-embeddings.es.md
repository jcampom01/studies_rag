# Paso 3 · Embeddings

[English](03-embeddings.md) · **Español** · [← Chunking](02-chunking.es.md) · [Siguiente: Base de datos vectorial →](04-vector-store.es.md)

Convierte cada chunk en un **vector**: una lista de números que representa su significado, de modo que los textos que hablan de lo mismo quedan cerca unos de otros.

```
chunks_<chunker>.jsonl  ──embed──►  data/processed/embeddings/<chunker>/
                                      ├── vectors.npy       matriz float32, una fila por chunk
                                      ├── chunk_ids.json    el chunk_id de cada fila
                                      └── manifest.json     cómo se generaron los vectores
```

## El modelo

Por defecto se usa [`Qwen/Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), ejecutado en local con `sentence-transformers`:

- **1.024 dimensiones** y un contexto de hasta 32k tokens (mucho más de lo que necesita un chunk).
- **Multilingüe**: una pregunta en español encuentra un pasaje en inglés y viceversa.
- **Admite instrucciones** (ver más abajo).
- **Pequeño**: unos 1,2 GB en la GPU.

Se eligió a partir de la clasificación multilingüe de recuperación de MTEB, entre los modelos que caben en la GPU junto a un LLM.

El modelo está **fijado a un commit** de su repositorio de Hugging Face (`revision`). Sin eso, una actualización en el Hub podría cambiar los vectores sin avisar, y los vectores guardados dejarían de corresponder a las nuevas consultas.

Lo que le pasa a un texto:

```
texto ─► tokenizador (vocabulario de ≈151k tokens) ─► Transformer ─► pooling del último token ─► normalización L2 ─► vector (1024)
```

Con vectores normalizados, la similitud coseno es un producto escalar.

## Qué se embebe

### Los chunks: una cabecera de contexto + el texto

```
<collection_name> › Tema <doc_number>: <doc_title> › <section>

<texto del chunk>
```

Un chunk a menudo no dice a qué asignatura o tema pertenece: "el algoritmo se detiene cuando…" podría ser de cualquier curso. La cabecera añade ese contexto y se construye a partir de los metadatos con `context_template`. Solo se usa para embeber; el texto guardado del chunk no cambia.

### Las preguntas: una instrucción + la pregunta

```
Instruct: Given a student's question, retrieve the passages of the course notes that answer it
Query: <pregunta>
```

La búsqueda es **asimétrica**: una pregunta y el pasaje que la responde no se parecen, porque una es corta y pregunta y el otro es largo y explica. Los modelos que admiten instrucciones están entrenados para situar una *consulta con instrucción* cerca de su respuesta. La instrucción va **solo en la consulta**; los documentos se embeben tal cual (`document_prompt: null`).

La instrucción es configuración, así que forma parte de los ajustes del embedder. Una consulta embebida con otra instrucción no es comparable con los vectores guardados.

## El manifest: ahorrar trabajo y detectar desajustes

`manifest.json` guarda todo lo que determina los vectores:

```json
{
  "embedder": "sentence_transformers",
  "model": "Qwen/Qwen3-Embedding-0.6B",
  "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
  "query_prompt": "Instruct: ...\nQuery:",
  "document_prompt": null,
  "normalized": true,
  "context_template": "{collection_name} › Tema {doc_number}: {doc_title} › {section}",
  "chunks_file": "data/processed/chunks_structured.jsonl",
  "chunks_sha256": "…",
  "dimension": 1024,
  "count": 2665,
  "created_at": "2026-11-09T12:08:29+00:00"
}
```

Se usa de tres formas:

- **Saltar si no ha cambiado.** Si el manifest coincide con la configuración actual y con el SHA-256 del fichero de chunks, `embed` no hace nada. Volver a ejecutarlo solo recalcula lo que ha cambiado.
- **Comprobación de compatibilidad.** Antes de buscar, `check_compatible` compara el manifest con el embedder que va a embeber la pregunta. Otro modelo, otra revisión u otra instrucción provocan un error claro, en lugar de devolver resultados sin sentido.
- **Completo o inexistente.** Primero se borra el manifest antiguo y el nuevo se escribe el último, después de los vectores y los ids. Una ejecución que falla a mitad deja una carpeta sin manifest, que cuenta como "sin embeber", nunca como un conjunto a medio escribir.

El contrato que implementa cada embedder:

```python
class Embedder(ABC):
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...   # (n, dim), normalized float32
    def embed_query(self, text: str) -> np.ndarray: ...              # (dim,)
    def describe(self) -> dict: ...                                  # settings, stored in the manifest
```

## Qué se midió

En el corpus de desarrollo:

- **Tokens.** Unos 3,85 caracteres por token. Tokens por chunk: mediana 294, p95 371, máximo 548, muy por debajo del contexto del modelo.
- **La instrucción** subió la puntuación de los pasajes correctos y cambió el primer resultado en 2 de 5 preguntas de prueba.
- **La cabecera de contexto** mejoró la precisión temática en 4 de 5 preguntas. La proporción de chunks cuyo vecino más cercano es de la misma colección pasó del 95,9 % al 98,5 %.
- **Debilidades conocidas de los embeddings densos**, medidas como similitud coseno entre pares de frases (un par sin relación da unos 0,10):

  | Par | Similitud | Problema |
  |---|---|---|
  | "algoritmos que necesitan datos etiquetados" / "… que no necesitan …" | 0,90 | la negación casi no se nota |
  | "tema 3 de …" / "tema 8 de …" | 0,94 | los números apenas cuentan |
  | una sigla / la misma sigla con dos letras intercambiadas | 0,66 | erratas en siglas |
  | una sigla / su forma desarrollada | 0,70 | el modelo conoce algunas siglas solo en parte |

  Son los motivos habituales para usar filtros por metadatos, búsqueda por palabras clave (híbrida) y rerankers: ver el paso 7 de la hoja de ruta.

## Ejecución

```bash
PYTHONPATH=src .venv/bin/python -m studies_rag.embed
```

Configuración: la sección `embed`.

| Clave | Significado |
|---|---|
| `inputs` | nombre del chunker → fichero de chunks. Cada uno tiene su propia carpeta dentro de `output_dir` |
| `context_template` | Cabecera que se antepone a cada chunk (`null` = sin cabecera). Usa campos de los metadatos |
| `embedder`, `embedder_options` | `model`, `revision`, `query_prompt`, `document_prompt`, `batch_size`, `device` (`null` = la GPU si la hay) |

La primera ejecución descarga el modelo en la caché de Hugging Face (`~/.cache/huggingface`).

> **Nota sobre la GPU (WSL 2).** Cuando la memoria de la GPU se llena, el driver no da error: pasa a usar la memoria del sistema y se vuelve muchas veces más lento. Si los embeddings van anormalmente lentos, mira en `nvidia-smi` si otros procesos (un kernel de notebook abierto, por ejemplo) ocupan memoria de la GPU.

**Nota sobre la precisión.** El modelo calcula en media precisión, así que los vectores guardados tienen normas entre 0,998 y 1,004 en lugar de exactamente 1. Un producto escalar simple se desvía entonces del coseno real hasta unos 0,003. El operador coseno de PostgreSQL divide por las normas, así que los resultados de búsqueda son exactos.
