import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import warnings
import urllib3

# Ignoramos las advertencias de seguridad por omitir la verificación SSL, ya que es intencional.
warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)

# --- 1. CONFIGURACIÓN INICIAL ---
# Cargar las variables de entorno para no tenerlas hardcodeadas
QDRANT_IP = os.getenv("QDRANT_IP", "209.126.82.74")
QDRANT_HOSTNAME = os.getenv("QDRANT_HOSTNAME", "soluciones-qdrant.vh0e8b.easypanel.host")
COLLECTION_NAME = "openpyxl_final_v2"
MODEL_NAME = 'all-MiniLM-L6-v2'

print("Cargando el modelo de embeddings... (Esto puede tardar un momento)")

# Cargamos el modelo UNA SOLA VEZ al iniciar el servidor para máxima eficiencia
try:
    model = SentenceTransformer(MODEL_NAME)
    print("Modelo cargado exitosamente.")
except Exception as e:
    print(f"Error fatal al cargar el modelo: {e}")
    model = None

# Creamos el cliente de Qdrant UNA SOLA VEZ al iniciar
try:
    client = QdrantClient(
        host=QDRANT_IP,
        port=443,
        https=True,
        verify=False, # Importante para tu configuración
        prefer_grpc=False,
        headers={"Host": QDRANT_HOSTNAME},
        timeout=20
    )
    print("Conexión con Qdrant establecida.")
except Exception as e:
    print(f"Error fatal al conectar con Qdrant: {e}")
    client = None

# Inicializamos la aplicación FastAPI
app = FastAPI()

# --- 2. MODELOS DE DATOS ---
# Esto define cómo deben ser las peticiones que llegan a nuestra API
class SearchQuery(BaseModel):
    question: str
    top_k: int = 5  # Número de resultados a devolver por defecto

# --- 3. ENDPOINT DE LA API ---
@app.post("/buscar")
async def buscar_en_documentacion(query: SearchQuery):
    """
    Recibe una pregunta, la convierte en un vector y busca en Qdrant.
    """
    if not model or not client:
        raise HTTPException(status_code=500, detail="El servidor no está inicializado correctamente (modelo o Qdrant no disponibles).")
    
    print(f"Recibida pregunta: '{query.question}'")
    
    try:
        # Paso 1: Convertir la pregunta en un vector
        vector_pregunta = model.encode(query.question).tolist()
        
        # Paso 2: Buscar en Qdrant
        search_result = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector_pregunta,
            limit=query.top_k,
            with_payload=True  # Para que nos devuelva el texto y la fuente
        )
        
        print(f"Búsqueda completada. Se encontraron {len(search_result)} resultados.")
        
        # Paso 3: Devolver los resultados
        return {"status": "success", "resultados": search_result}
        
    except Exception as e:
        print(f"Error durante la búsqueda: {e}")
        raise HTTPException(status_code=500, detail=f"Ocurrió un error en el servidor: {e}")

@app.get("/")
def read_root():
    return {"status": "Servicio de búsqueda de OpenPyXL activo."}
