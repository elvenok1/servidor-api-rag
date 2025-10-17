import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import warnings
import urllib3
from typing import List, Dict, Any

# Ignoramos advertencias de seguridad de conexión
warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)

# --- 1. CONFIGURACIÓN PARA SERVIDOR ---
QDRANT_IP = os.getenv("QDRANT_IP", "209.126.82.74")
QDRANT_HOSTNAME = os.getenv("QDRANT_HOSTNAME", "soluciones-qdrant.vh0e8b.easypanel.host")

# Usamos el modelo y la colección consistentes con la estrategia simplificada
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
COLLECTION_NAME = "openpyxl_semantic_v7" 

# <<< ¡CLAVE PARA SERVIDOR! Hacemos que el modelo se guarde en el volumen persistente.
CACHE_DIR = "/app/.cache" 
os.environ['SENTENCE_TRANSFORMERS_HOME'] = CACHE_DIR
print(f"Directorio de caché para el modelo configurado en: {CACHE_DIR}")


# --- Carga de recursos globales (Modelo y Cliente Qdrant) ---
model = None
client = None

print("Cargando el modelo de embeddings... (Esto puede tardar en el primer arranque)")
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
    print(f"Conexión exitosa. Colección '{COLLECTION_NAME}' encontrada.")
except Exception as e:
    print(f"Error fatal al conectar o encontrar la colección en Qdrant: {e}")
    client = None

# --- Modelos de datos Pydantic para la respuesta de la API ---
class SearchResult(BaseModel):
    id: str
    score: float
    payload: Dict[str, Any]

class SearchResponse(BaseModel):
    status: str
    resultados: List[SearchResult]

# --- Inicialización de FastAPI ---
app = FastAPI(
    title="API de Búsqueda Semántica Simplificada (Versión Servidor)",
    description="Servicio que busca directamente en Qdrant sin contexto fijo."
)

# --- Endpoint de Búsqueda (/buscar) ---
@app.get("/buscar", response_model=SearchResponse)
async def search_documentation(question: str, top_k: int = 3):
    if not model or not client:
        raise HTTPException(status_code=503, detail="Servicio no disponible: Modelo o Qdrant no inicializados.")
    
    print(f"Recibida pregunta: '{question}'")
    
    try:
        # Vectorizamos la pregunta del usuario directamente
        vector_pregunta = model.encode(question).tolist()
        
        search_results_raw = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector_pregunta,
            limit=top_k,
            with_payload=True 
        )
        
        resultados_limpios = [
            SearchResult(id=str(hit.id), score=hit.score, payload=hit.payload) 
            for hit in search_results_raw
        ]
        
        print(f"Búsqueda completada. Devolviendo {len(resultados_limpios)} resultados.")
        return SearchResponse(status="success", resultados=resultados_limpios)
        
    except Exception as e:
        error_message = f"Ocurrió un error inesperado durante la búsqueda: {e}"
        print(error_message)
        raise HTTPException(status_code=500, detail=error_message)

# --- Endpoint Raíz (/) para verificar que el servicio está vivo ---
@app.get("/")
def read_root():
    return {"status": "Servicio de búsqueda de OpenPyXL activo."}

