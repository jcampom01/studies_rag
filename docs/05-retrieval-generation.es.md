# Paso 5 · Recuperación + generación

[English](05-retrieval-generation.md) · **Español** · [← Base de datos vectorial](04-vector-store.es.md) · [README](../README.es.md)

Responde una pregunta: recupera los chunks más cercanos de la base de datos vectorial y deja que un LLM local redacte la respuesta a partir de ellos, citando los pasajes que ha usado.

```
pregunta ─► embed_query ─► store.search ─► prompt ─► LLM (Ollama) ─► respuesta + fuentes
```

## El servicio del LLM: Ollama

`docker-compose.yml` ejecuta `ollama/ollama:0.40.11` con acceso a la GPU. Los modelos descargados viven en un volumen con nombre (`ollama`), así que sobreviven a que se recree el contenedor.

```bash
docker compose up -d ollama
docker compose exec ollama ollama pull qwen3.5:4b
```

### El modelo: `qwen3.5:4b`

- 4.200 millones de parámetros, cuantizado a Q4_K_M: unos 3,7 GB en la GPU una vez cargado.
- Multilingüe, así que responde bien en el idioma de la pregunta.

**Reparto de memoria en una GPU de 8 GB:**

| | GB |
|---|---|
| Escritorio y otros procesos | ~1,4 |
| Modelo de embeddings | ~1,2 |
| LLM | ~3,7 |
| Margen libre | ~1,9 |

Un modelo más grande (8–9B) obligaría a pasar el embedder de consultas a la CPU (`"device": "cpu"`); embeber una pregunta en CPU tarda una fracción de segundo.

### Dos comportamientos de Ollama que importan en RAG

**1. El modo "thinking" está activado por defecto.** Este modelo "razona" antes de responder, y para una respuesta de una frase escribió antes unas 1.500 palabras de razonamiento. En RAG eso solo añade latencia. Las peticiones envían `"think": false`.

**2. Los prompts que no caben se recortan sin avisar.** Enviamos un prompt de 2.435 tokens con tamaños de contexto menores:

| `num_ctx` | Tokens que leyó Ollama | Código HTTP |
|---|---|---|
| 1024 | 1.026 | 200 |
| 2048 | 1.026 | 200 |
| 2400 | 1.202 | 200 |
| 2560 | 2.435 (todo) | 200 |

Cuando el prompt no cabe, Ollama descarta una gran parte, conservando el principio y el final, responde con normalidad e informa de un número de tokens muy por debajo de `num_ctx`. En RAG, los chunks del medio del prompt desaparecerían sin dejar rastro. Por eso las peticiones envían **`"truncate": false`**, que lo convierte en un error:

```
400 {"error":"request (2435 tokens) exceeds the available context size (2048 tokens), try increasing it"}
```

Estas opciones solo existen en la **API nativa** de Ollama (`/api/chat`), no en la compatible con OpenAI. Por eso el adaptador usa la API nativa, con llamadas `httpx` directas.

## El prompt

Cada pregunta envía dos mensajes.

**System**: las instrucciones fijas, desde `config.json`:

```
You answer questions using only the numbered context passages.
Cite the passages you use with their numbers, like [1] or [2][3].
If the context does not contain the answer, say so plainly instead of guessing.
Answer in the language of the question.
```

- *Solo los pasajes*: sin esto, el modelo mezcla lo que "sabe", que en un modelo pequeño es poco y a veces erróneo.
- *Citar*: cada afirmación se puede comprobar contra su fuente.
- *Decirlo*: es lo que evita que una respuesta que falta se convierta en una inventada.

**User**: los chunks recuperados y la pregunta:

```
Context:

[1] Course A › Tema 3: Topic title › 3.2. A section title (pages 6-7)
<texto del chunk>

[2] ...

Question: <pregunta>
```

- **Los pasajes van numerados desde 1** y etiquetados con `source_template` más sus páginas. Los mismos números aparecen en las fuentes que se muestran y en las citas.
- **Van ordenados por score**, el mejor primero. Los LLM prestan más atención al principio y al final del contexto.
- **La pregunta va al final**, justo antes de que el modelo responda.
- **Los scores no se muestran** al modelo.

## La pipeline: `rag.py`

```python
class RAG:
    def __init__(self, embedder, store, llm, index_name, k, system_prompt, source_template): ...
    def retrieve(self, question, filters=None) -> list[SearchResult]   # embed + search
    def build_prompt(self, question, results) -> str
    def answer(self, question, filters=None) -> Answer
```

- **Recibe sus piezas ya construidas** (inyección de dependencias) y solo conoce sus contratos. Nada en ella se refiere a PostgreSQL, Ollama o Hugging Face.
- **Falla al arrancar, no en la primera pregunta.** El constructor comprueba que el conjunto de chunks existe en la base de datos y que su manifest es compatible con el embedder. Otro modelo u otra instrucción producirían, si no, una recuperación sin sentido y sin ningún error.
- **`retrieve` va separado de `answer`**, para poder evaluar la recuperación sola, sin llamar al LLM.
- **La respuesta es un objeto de datos**, no texto impreso:

```python
@dataclass
class Answer:
    question: str
    text: str
    sources: list[SearchResult]   # numbered from 1, as in the prompt
    cited: list[int]              # passage numbers found in the answer: [1], [2][3], [1, 3]
    retrieval_seconds: float
    completion: Completion        # prompt_tokens, output_tokens, seconds
```

Las citas a pasajes que no existen (`[9]` cuando había 5) se descartan.

El contrato del LLM:

```python
class LLM(ABC):
    def generate(self, system: str, prompt: str) -> Completion: ...
    def describe(self) -> dict: ...
```

## El punto de entrada en la terminal: `ask.py`

Construye la pipeline a partir de la configuración y responde las preguntas que se escriben:

- **El embedder** sale de la sección `embed`, porque tiene que ser el que generó los vectores guardados.
- **La base de datos vectorial** sale de la sección `index`.
- **El LLM y los ajustes de consulta** salen de la sección `query`.

Una línea vacía o Ctrl+D termina. También lee de una tubería:

```bash
echo "¿Qué diferencia hay entre bagging y boosting?" | PYTHONPATH=src .venv/bin/python -m studies_rag.ask
```

Para cada pregunta muestra:

- la respuesta;
- cada fuente recuperada con su score, marcando las citadas con `*`;
- los tiempos: recuperación (unos 0,1 s) y LLM (unos segundos).

La primera pregunta es más lenta, porque en ese momento se cargan el modelo de embeddings y, si Ollama lo había descargado de memoria, el LLM.

Conviene fijarse en las fuentes recuperadas que no se citan. Si no se cita ninguna, probablemente el modelo ha dicho que la respuesta no está en el contexto.

## Configuración: la sección `query`

| Clave | Significado |
|---|---|
| `index` | En qué conjunto de chunks buscar (`structured` o `langchain`) |
| `k` | Número de chunks que se ponen en el prompt |
| `system_prompt` | Las instrucciones fijas |
| `source_template` | Etiqueta de cada pasaje, construida con campos de los metadatos |
| `llm`, `llm_options` | `base_url`, `model`, `think`, `num_ctx`, `temperature` (`null` = la del modelo), `timeout` |

`num_ctx` es de 8.192 tokens: 5 chunks con sus etiquetas, las instrucciones y la pregunta ocupan unos 2.500. Subir `num_ctx` cuesta memoria de GPU, y cambiarlo hace que Ollama recargue el modelo.

`temperature` es 0,3, más baja que la del modelo (1,0), para que las respuestas se ciñan a los pasajes.

## Limitaciones

- **La recuperación marca el techo.** Si el chunk con la respuesta no está entre los `k` recuperados, lo mejor que puede hacer el LLM es decir que no lo sabe.
- **"Solo los pasajes" es una instrucción, no una garantía.** Un modelo pequeño a veces añade conocimiento propio. Las citas lo hacen visible y el paso 6 lo medirá.
- **No hay conversación.** Cada pregunta es independiente.
- **Todavía no hay filtros por metadatos en la terminal**, aunque `retrieve` y `answer` los admiten.
