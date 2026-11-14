# Paso 1 · Ingesta

[English](01-ingestion.md) · **Español** · [← README](../README.es.md) · [Siguiente: Chunking →](02-chunking.es.md)

Convierte los documentos de origen en **páginas limpias**: el texto útil de cada página, sin cabeceras, pies de página ni portadas, más metadatos que indican de dónde viene.

```
data/raw/<colección>/*.pdf  ──ingest──►  data/processed/pages.jsonl
```

## Salida: `pages.jsonl`

Un objeto JSON por línea, una línea por cada página conservada:

```json
{
  "source": "data/raw/course_a/topic_03.pdf",
  "page": 6,
  "text": "## 3.2. A section title\n\nFirst paragraph of the page...\n\n- A list item...",
  "metadata": {
    "collection": "course_a",
    "collection_name": "Course A",
    "doc_number": 3,
    "doc_title": "Topic title",
    "page_type": "Key ideas",
    "section": "3.2. A section title",
    "subsection": null
  }
}
```

- `source` y `page` (empezando en 1) identifican la página. Todos los pasos siguientes los conservan, así que una respuesta siempre puede señalar la página del PDF.
- `text` está formado por párrafos separados por una línea en blanco (`\n\n`). Los títulos llevan un prefijo Markdown (`##` para secciones, `###` para subsecciones) para que el chunker los reconozca. Los listados de código conservan sus saltos de línea.
- `metadata` es libre y lo decide el adaptador. Los pasos siguientes lo copian en cada chunk y la base de datos lo guarda como JSON, así que sirve para los prompts y para filtrar búsquedas.

## Cómo funciona

La ingesta está dividida en dos partes:

- **`ingest.py`**, la parte genérica. Busca los ficheros, descarta duplicados, pide las páginas a un adaptador, normaliza el texto y escribe el fichero JSON Lines. No sabe nada de cómo está maquetado ningún documento.
- **Un adaptador**, la parte específica de cada tipo de documento. Sabe dónde está el título, qué texto es ruido y qué páginas merecen la pena.

```python
class Adapter(ABC):
    file_pattern: str = "*"   # glob of the files this adapter reads

    @abstractmethod
    def extract(self, path: Path) -> Iterator[PageRecord]:
        """Yield the pages of a document worth indexing, with their text and metadata."""
```

`ingest.py` hace además tres cosas genéricas:

- **Busca los ficheros de forma recursiva** dentro de `raw_dir`. La subcarpeta de cada fichero es su *colección*.
- **Descarta los duplicados idénticos byte a byte** (mismo SHA-256) y muestra un aviso.
- **Aplica la normalización Unicode NFKC.** Los PDFs suelen contener ligaduras (`ﬁ`) y acentos descompuestos (`a` + un acento combinable), que se ven igual en pantalla pero son caracteres distintos. Harían que la misma palabra se embebiera y se encontrara de forma diferente. NFKC los convierte a una única forma estándar.

## El adaptador incluido: `master_notes`

Lee apuntes de curso exportados a PDF desde una plantilla web fija:

| Elemento | Cómo lo encuentra el adaptador |
|---|---|
| Número de documento y nombre de la colección | En la portada, por **tamaño de letra** (cada campo tiene el suyo) |
| Título del documento | Primera entrada del índice interno del PDF, con el nombre del fichero como alternativa |
| Sección / subsección de una página | La última entrada del índice de nivel 1 / nivel 2 que empieza en esa página o antes |
| Tipo de página | Un bloque de la cabecera con un tamaño de letra concreto. Las páginas sin él (portada, índice) se descartan |
| Cabecera y pie de página | Por posición vertical: bloques por encima de `y = 75 pt` o por debajo de `y = 790 pt` |
| Títulos y código | Títulos por tamaño de letra (≥ 12 pt, ≥ 16 pt para secciones); código por su tamaño de 9 pt en monoespaciada |

### Reconstruir los párrafos

PyMuPDF devuelve cada página como *bloques*, a menudo uno por línea visual, así que hay que reconstruir los párrafos. Un bloque **empieza un párrafo nuevo** cuando:

- cambia su tipo (título, código o texto) o su tamaño de letra;
- o el hueco vertical respecto al bloque anterior supera los **15 pt**.

El umbral de 15 pt no está puesto a ojo. Un histograma de los huecos entre bloques consecutivos de todo el corpus es claramente bimodal: las líneas de un mismo párrafo están juntas y los huecos entre párrafos son mayores. 15 pt queda entre los dos picos.

Dos casos especiales:

- **Las viñetas de lista** (`▸`, `•`) llegan como bloques separados. Se convierten en `- ` y siempre se unen al texto que les sigue.
- **Los elementos de lista cuyo contenido era una imagen** (fórmulas, por ejemplo) se quedan con un `-` solo y se descartan.

### Opciones

| Opción | Significado |
|---|---|
| `keep_page_types` | Tipos de página que se conservan (la configuración de ejemplo solo conserva las páginas de contenido principal). `null` conserva todas las que tienen cabecera de tipo de página |

### Cómo se analizó la plantilla

Todos los umbrales anteriores salen de medir los documentos, no de suponer:

1. **Mirar**: dibujar algunas páginas con sus bloques como recuadros, coloreados por tamaño de letra.
2. **Volcar**: imprimir cada bloque con su posición y su tamaño.
3. **Suponer** una regla, por ejemplo "la cabecera termina en y = 75".
4. **Medir** la regla en todo el corpus: todos los bloques de todos los PDFs en una sola tabla.
5. **Revisar los casos atípicos**: los documentos que no cumplen la regla.

En el corpus de desarrollo eso supuso 82 PDFs y unos 74.000 bloques de texto. Uno de los ficheros era un duplicado idéntico, y los demás dieron 2.065 páginas.

## Limitaciones conocidas

- **Páginas escaladas.** Algunas páginas se generan a una escala ligeramente distinta (texto a 9,8–10,3 pt, títulos a 16,6–18,8 pt), así que algunos títulos de subsección no se reconocen.
- **No todo el texto de 9 pt es código.** Algunos pies de figura usan el mismo tamaño con una fuente proporcional, así que se conservan con sus saltos de línea.
- **Varias subsecciones en una página.** La sección y la subsección se asignan por página. Cuando varias subsecciones empiezan en la misma página, todo su texto recibe la última.
- **Las fórmulas dibujadas como imagen** se pierden: no hay OCR.

## Escribir un adaptador

Para ingerir otros documentos (otra plantilla, ficheros Word, páginas web…), escribe una clase que devuelva `PageRecord` y regístrala:

```python
# src/studies_rag/adapters/my_docs.py
from pathlib import Path
from collections.abc import Iterator
import pymupdf
from studies_rag.adapters.base import Adapter, PageRecord


class MyDocsAdapter(Adapter):
    file_pattern = "*.pdf"

    def __init__(self, min_chars: int = 50):   # options come from "adapter_options"
        self.min_chars = min_chars

    def extract(self, path: Path) -> Iterator[PageRecord]:
        with pymupdf.open(path) as doc:
            for page in doc:
                text = page.get_text().strip()
                if len(text) >= self.min_chars:
                    yield PageRecord(source=str(path), page=page.number + 1, text=text,
                                     metadata={"collection": path.parent.name, "title": doc.metadata.get("title")})
```

```python
# src/studies_rag/adapters/__init__.py
ADAPTERS = {"master_notes": MasterNotesAdapter, "my_docs": MyDocsAdapter}
```

```json
"ingest": {"adapter": "my_docs", "adapter_options": {"min_chars": 50}, ...}
```

Si tus metadatos usan otros nombres de campo, actualiza también los ajustes que se refieren a ellos: `group_by` (sección `chunk`), `context_template` (sección `embed`) y `source_template` (sección `query`).

## Ejecución

```bash
PYTHONPATH=src .venv/bin/python -m studies_rag.ingest
```

Configuración: la sección `ingest` de `config.json` (`adapter`, `adapter_options`, `raw_dir`, `output`). El fichero de salida se reescribe en cada ejecución.
