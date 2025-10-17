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
# !! CAMBIO CLAVE: Apuntamos a la nueva colección semántica !!
COLLECTION_NAME = "openpyxl_semantic_v2" 
MODEL_NAME = 'all-MiniLM-L6-v2'

print("Cargando el modelo de embeddings... (Esto puede tardar un momento)")
try:
    model = SentenceTransformer(MODEL_NAME)
    print("Modelo cargado exitosamente.")
except Exception as e:
    print(f"Error fatal al cargar el modelo: {e}")
    model = None

print("Estableciendo conexión con Qdrant...")
try:
    client = QdrantClient(
        host=QDRANT_IP, port=443, https=True, verify=False,
        prefer_grpc=False, headers={"Host": QDRANT_HOSTNAME}, timeout=20
    )
    # Verificamos que la colección exista al iniciar
    client.get_collection(collection_name=COLLECTION_NAME)
    print(f"Conexión exitosa y la colección '{COLLECTION_NAME}' fue encontrada.")
except Exception as e:
    print(f"Error fatal al conectar o encontrar la colección en Qdrant: {e}")
    client = None

# --- 2. MODELOS DE DATOS (Pydantic) PARA UNA RESPUESTA LIMPIA ---
class SearchResult(BaseModel):
    id: str
    score: float
    payload: Dict[str, Any]

class SearchResponse(BaseModel):
    status: str
    resultados: List[SearchResult]

# --- 3. INICIALIZACIÓN DE FASTAPI ---
app = FastAPI(
    title="API de Búsqueda Semántica para OpenPyXL",
    description="Un servicio para encontrar ejemplos de código y documentación de OpenPyXL."
)

# --- 4. ENDPOINT OPTIMIZADO ---
@app.get("/buscar", response_model=SearchResponse)
async def search_documentation(question: str, top_k: int = 3):
    """
    Recibe una pregunta, la vectoriza y busca en Qdrant los chunks más relevantes.
    Devuelve una respuesta limpia y estructurada.
    """
    if not model or not client:
        raise HTTPException(status_code=503, detail="Servicio no disponible: Modelo o conexión a la base de datos no inicializados.")
    
    print(f"Recibida pregunta: '{question}' con top_k={top_k}")
    
    try:
        # Paso 1: Convertir la pregunta en un vector
        vector_pregunta = model.encode(question).tolist()
        
        # Paso 2: Buscar en Qdrant
        search_results_raw = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector_pregunta,
            limit=top_k,
            with_payload=True 
        )
        
        print(f"Búsqueda completada. Se encontraron {len(search_results_raw)} resultados.")
        
        # Paso 3: Limpiar y estructurar la respuesta
        resultados_limpios = [
            SearchResult(
                id=hit.id,
                score=hit.score,
                payload=hit.payload
            ) for hit in search_results_raw
        ]
        
        return SearchResponse(status="success", resultados=resultados_limpios)
        
    except Exception as e:
        error_message = f"Error durante la búsqueda: {type(e).__name__}: {e}"
        print(error_message)
        raise HTTPException(status_code=500, detail=error_message)

@app.get("/")
def read_root():
    return {"status": "Servicio de búsqueda de OpenPyXL activo."}
