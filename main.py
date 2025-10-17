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

# --- 1. CONFIGURACIÓN MEJORADA ---
QDRANT_IP = os.getenv("QDRANT_IP", "209.126.82.74")
QDRANT_HOSTNAME = os.getenv("QDRANT_HOSTNAME", "soluciones-qdrant.vh0e8b.easypanel.host")
# !! CAMBIO CLAVE 1: Apuntamos a la nueva y correcta colección !!
COLLECTION_NAME = "openpyxl_semantic_v2" 
MODEL_NAME = 'all-MiniLM-L6-v2'

# !! CAMBIO CLAVE 2: Forzar un directorio de caché local para el modelo !!
# Esto soluciona problemas de estado y permisos en entornos de servidor.
cache_dir = os.path.join(os.getcwd(), ".cache")
os.makedirs(cache_dir, exist_ok=True)
os.environ['SENTENCE_TRANSFORMERS_HOME'] = cache_dir
print(f"Usando el directorio de caché para el modelo: {cache_dir}")

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

# --- Endpoint optimizado (sin cambios en la lógica, pero ahora funcionará) ---
@app.get("/buscar", response_model=SearchResponse)
async def search_documentation(question: str, top_k: int = 3):
    if not model or not client:
        raise HTTPException(status_code=503, detail="Servicio no disponible: Modelo o conexión a la base de datos no inicializados.")
    
    print(f"Recibida pregunta: '{question}' con top_k={top_k}")
    
    try:
        vector_pregunta = model.encode(question).tolist()
        
        search_results_raw = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector_pregunta,
            limit=top_k,
            with_payload=True 
        )
        
        print(f"Búsqueda completada. Se encontraron {len(search_results_raw)} resultados.")
        
        resultados_limpios = [
            SearchResult(id=hit.id, score=hit.score, payload=hit.payload) 
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
