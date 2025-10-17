import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import warnings
import urllib3
from typing import List, Dict, Any

# Ignoramos advertencias de seguridad
warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)

# --- 1. CONFIGURACIÓN (Debe ser idéntica a la del script de indexación) ---
QDRANT_IP = os.getenv("QDRANT_IP", "209.126.82.74")
QDRANT_HOSTNAME = os.getenv("QDRANT_HOSTNAME", "soluciones-qdrant.vh0e8b.easypanel.host")
COLLECTION_NAME = "openpyxl_semantic_v3" 
MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
CACHE_DIR = "/app/.cache" 
os.environ['SENTENCE_TRANSFORMERS_HOME'] = CACHE_DIR
print(f"Directorio de caché para el modelo configurado en: {CACHE_DIR}")

# --- 2. CONTEXTO FIJO (DEBE SER EXACTAMENTE EL MISMO QUE EN EL SCRIPT DE INDEXACIÓN) ---
FIXED_CONTEXT_OPTIMIZED = """
Eres un programador Python experto de élite, especializado en la librería `openpyxl`. Tu única tarea es escribir el cuerpo de una función de Python.

**REGLAS ESTRICTAS:**
1.  **DEBES** escribir una única función de Python llamada `generar_excel()`. Esta función no debe recibir argumentos.
2.  **DEBES** incluir todas las importaciones necesarias de `openpyxl` (y otras librerías como `datetime`) **DENTRO** de la función `generar_excel()`. No asumas que hay nada pre-importado.
3.  La función **DEBE** crear una nueva instancia de `Workbook` con `wb = Workbook()`.
4.  La función **DEBE** terminar retornando la instancia del workbook: `return wb`.
5.  No incluyas ningún código fuera de la definición de esta función.

**EJEMPLO DE ESTRUCTURA VÁLIDA:**

```python
def generar_excel():
    # Paso 1: Importaciones (DENTRO de la función)
    from openpyxl import Workbook
    from openpyxl.chart import BarChart, Reference
    # ... otras importaciones necesarias ...

    # Paso 2: Lógica de creación del Excel
    wb = Workbook()
    ws = wb.active
    
    # ...código para añadir datos, gráficos, estilos, etc...

    # Paso 3: Retornar el workbook
    return wb
```

A continuación, se presenta una funcionalidad específica con su descripción y ejemplo de código. Adapta este ejemplo para que siga la estructura de la función `generar_excel()` y resuelva la petición del usuario.
"""

# --- Carga de recursos globales ---
model = None
client = None

print("Cargando el modelo de embeddings... (Esto puede tardar un momento)")
try:
    model = SentenceTransformer(MODEL_NAME)
    print("Modelo cargado exitosamente.")
except Exception as e:
    print(f"Error fatal al cargar el modelo: {e}")

print("Estableciendo conexión con Qdrant...")
try:
    client = QdrantClient(
        host=QDRANT_IP, port=443, https=True, verify=False,
        prefer_grpc=False, headers={"Host": QDRANT_HOSTNAME}, timeout=20
    )
    client.get_collection(collection_name=COLLECTION_NAME)
    print(f"Conexión exitosa y la colección '{COLLECTION_NAME}' fue encontrada.")
except Exception as e:
    print(f"Error fatal al conectar o encontrar la colección en Qdrant: {e}")
    client = None

# --- Modelos de datos Pydantic (sin cambios) ---
class SearchResult(BaseModel):
    id: str
    score: float
    payload: Dict[str, Any]

class SearchResponse(BaseModel):
    status: str
    resultados: List[SearchResult]

# --- Inicialización de FastAPI ---
app = FastAPI(
    title="API de Búsqueda Semántica para OpenPyXL",
    description="Un servicio para encontrar ejemplos de código y documentación de OpenPyXL."
)

# --- Endpoint optimizado y CORREGIDO ---
@app.get("/buscar", response_model=SearchResponse)
async def search_documentation(question: str, top_k: int = 3):
    if not model or not client:
        raise HTTPException(status_code=503, detail="Servicio no disponible: Modelo o conexión a la base de datos no inicializados.")
    
    print(f"Recibida pregunta: '{question}' con top_k={top_k}")
    
    try:
        # <<< CAMBIO CLAVE: CONSTRUIR EL PROMPT PARA LA PREGUNTA >>>
        # Envolvemos la pregunta en la misma estructura usada para indexar.
        # Esto asegura que los vectores de la pregunta y de los documentos
        # "hablen el mismo idioma".
        full_query_for_embedding = (
            f"{FIXED_CONTEXT_OPTIMIZED}\n\n"
            f"## TEMA: {question}\n\n"
            f"### DESCRIPCIÓN:\n{question}"
        )

        # Ahora codificamos este texto enriquecido en lugar de la pregunta simple
        vector_pregunta = model.encode(full_query_for_embedding).tolist()
        
        search_results_raw = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector_pregunta,
            limit=top_k,
            with_payload=True 
        )
        
        print(f"Búsqueda completada. Se encontraron {len(search_results_raw)} resultados.")
        
        resultados_limpios = [
            SearchResult(id=str(hit.id), score=hit.score, payload=hit.payload) 
            for hit in search_results_raw
        ]
        
        return SearchResponse(status="success", resultados=resultados_limpios)
        
    except Exception as e:
        error_message = f"Error durante la búsqueda: {type(e).__name__}: {e}"
        print(error_message)
        raise HTTPException(status_code=500, detail=error_message)

@app.get("/")
def read_root():
    return {"status": "Servicio de búsqueda de OpenPyXL activo."}




