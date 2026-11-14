# Paso 2 · Chunking

[English](02-chunking.md) · **Español** · [← Ingesta](01-ingestion.es.md) · [Siguiente: Embeddings →](03-embeddings.es.md)

Divide las páginas en **chunks**: los fragmentos de texto que se embeben, se recuperan y se muestran al LLM.

```
data/processed/pages.jsonl  ──chunk──►  data/processed/chunks_<chunker>.jsonl   (un fichero por chunker)
```

## Por qué importa el chunking

El chunk es la unidad de recuperación, así que su tamaño es un compromiso:

- **Demasiado grande**: un solo vector tiene que representar varias ideas y no encaja bien con ninguna. Además, gasta el contexto del LLM.
- **Demasiado pequeño**: el chunk pierde el contexto que lo hace comprensible (un pie de figura sin su figura, un elemento de lista sin su título).

También importa dónde se corta. Un chunk que termina con un título, o que empieza a mitad de frase, es más difícil de recuperar y de usar para responder.

El proyecto incluye **dos chunkers a propósito**: uno propio que sigue la estructura del documento y el divisor estándar de LangChain. Los dos trabajan sobre las mismas páginas, así que se pueden comparar en todos los pasos siguientes.

## Salida: `chunks_<chunker>.jsonl`

```json
{
  "chunk_id": "9d18103009e45f5b",
  "text": "## 3.2. A section title\n\nFirst paragraph...\n\nSecond paragraph...",
  "metadata": {
    "collection": "course_a", "collection_name": "Course A",
    "doc_number": 3, "doc_title": "Topic title", "page_type": "Key ideas",
    "section": "3.2. A section title", "subsection": null,
    "source": "data/raw/course_a/topic_03.pdf",
    "pages": [6, 7]
  }
}
```

- **`chunk_id`** son los 16 primeros caracteres hexadecimales de `sha256(source + "\n" + text)`. Es estable: el mismo texto del mismo fichero siempre tiene el mismo id, entre ejecuciones y entre chunkers.
- **`pages`** es siempre una lista, porque un chunk puede abarcar varias páginas.
- **El resto de metadatos** viene de las páginas. Si las páginas de un chunk no coinciden en un campo, ese campo pasa a ser una lista con los valores distintos.

El contrato que implementa cada chunker:

```python
class Chunker(ABC):
    @abstractmethod
    def split(self, pages: Iterable[PageRecord]) -> Iterator[Chunk]: ...
```

## Chunker 1: `structured` (propio)

Sigue la estructura que conservó el adaptador: sección, luego párrafo, luego frase.

1. **Agrupar las páginas.** Las páginas consecutivas del mismo fichero y con los mismos campos `group_by` (por defecto, la sección) forman un grupo. Un chunk puede abarcar varias páginas, pero nunca mezcla dos secciones.
2. **Unir los párrafos cortados por un salto de página.** Si el último párrafo de una página no termina con puntuación de cierre y la página siguiente empieza en minúscula, los dos son un solo párrafo.
3. **Empaquetar párrafos enteros** en un chunk hasta que añadir el siguiente superaría `target_chars`.
4. **Dividir los párrafos demasiado largos.** Un párrafo de más de `max_chars` se divide de forma recursiva: primero por líneas, luego por frases y luego por palabras.
5. **Solapamiento.** Cada chunk empieza con los últimos `overlap_paragraphs` párrafos del anterior, así que una idea en la frontera aparece en los dos.
6. **No terminar nunca con un título.** Los títulos finales pasan al principio del chunk siguiente y allí sustituyen al solapamiento, porque un título abre un tema nuevo.

| Opción | Por defecto | Significado |
|---|---|---|
| `group_by` | `[]` | Campos de metadatos que definen un grupo, además del fichero. La configuración de ejemplo usa `["section"]` |
| `target_chars` | 1200 | Los chunks se llenan hasta este tamaño |
| `max_chars` | 2000 | Los párrafos más largos se dividen |
| `overlap_paragraphs` | 1 | Párrafos que se repiten al principio del chunk siguiente |

Un título es una sola línea que empieza por `#` a `######` y un espacio. La condición de una sola línea importa: los comentarios de Python dentro de los listados de código (`# compute...`) también empiezan por `#`, y ocupan varias líneas una vez que el bloque de código se mantiene unido.

## Chunker 2: `langchain`

Es el `RecursiveCharacterTextSplitter` de LangChain, usado de la forma habitual: cada página es un `Document` y se divide por separado, probando los separadores `\n\n`, `\n`, `. `, ` ` y `""` en ese orden.

| Opción | Por defecto | Significado |
|---|---|---|
| `chunk_size` | 1200 | Máximo de caracteres por chunk |
| `chunk_overlap` | 200 | Caracteres repetidos entre chunks consecutivos de una página |
| `separators` | ver arriba | Puntos de corte, probados en orden |

No sabe nada de secciones ni de títulos, y sus chunks nunca cruzan el límite de una página.

## Resultados en el corpus de desarrollo

A partir de 2.065 páginas:

| | `structured` | `langchain` |
|---|---|---|
| Chunks | 2.665 | 2.922 |
| Chunks que terminan con un título | **0** | 101 |
| Chunks diminutos | 27 | 154 |
| Chunks que empiezan a mitad de frase | 39 | 376 |
| Chunks que terminan a mitad de frase | 363 | 730 |
| Texto extra por el solapamiento | +19 % | +2 % |

- El chunker propio corta en mejores sitios, porque sabe dónde empiezan los párrafos y las secciones.
- El precio es más texto repetido: un párrafo entero de solapamiento frente a 200 caracteres, que en la práctica solo se aplican cuando una página es más larga que `chunk_size`.
- Si estas diferencias mejoran de verdad **la recuperación y las respuestas** es lo que medirá el paso 6 (evaluación).

Dos errores encontrados al medir:

- **Comentarios de código tomados como títulos.** Antes de la regla de una sola línea, un comentario como `#cancer.data` dentro de un listado se tomaba por un título, y eso además generaba un chunk demasiado grande.
- **Títulos al final de los chunks.** Pasaron de 324 a 0 cuando el adaptador empezó a marcar los títulos en Markdown y el chunker a mover los títulos finales al chunk siguiente.

## Ejecución

```bash
PYTHONPATH=src .venv/bin/python -m studies_rag.chunk
```

Configuración: la sección `chunk` (`chunkers`, `input`, `output_dir`, `chunker_options.<nombre>`). Para cada chunker, el script muestra el número de chunks y los cuartiles de sus tamaños.

Para añadir un chunker, implementa `Chunker.split`, regístralo en `chunkers/__init__.py` y añade su nombre a `chunkers` y sus opciones a `chunker_options`. Los pasos 3 y 4 necesitarán además una entrada para él en `embed.inputs`.
