# --- Etapa 1: Base de Python ---
# Usamos una imagen oficial y ligera de Python como base.
# 'slim' es una versión que tiene lo esencial, haciendo nuestra imagen final más pequeña.
FROM python:3.11-slim

# --- Etapa 2: Configuración del Entorno ---
# Establecemos el directorio de trabajo dentro del contenedor.
# Todas las siguientes instrucciones se ejecutarán desde /app.
WORKDIR /app

# --- Etapa 3: Instalar Dependencias ---
# Copiamos solo el archivo de requerimientos primero. Docker es inteligente
# y si este archivo no cambia, usará la caché de esta capa, haciendo
# las futuras compilaciones mucho más rápidas.
COPY requirements.txt requirements.txt

# Instalamos todas las librerías necesarias.
# '--no-cache-dir' asegura que no se guarde la caché de pip, manteniendo la imagen ligera.
# '--upgrade pip' es una buena práctica para tener la última versión de pip.
RUN pip install --no-cache-dir --upgrade pip -r requirements.txt

# --- Etapa 4: Copiar el Código de la Aplicación ---
# Ahora copiamos el resto de nuestro código (main.py) al directorio de trabajo.
COPY . .

# --- Etapa 5: Exponer el Puerto ---
# Le decimos a Docker que el contenedor escuchará en el puerto 8000.
# EasyPanel usará esta información para saber a qué puerto dirigir el tráfico.
EXPOSE 8000

# --- Etapa 6: Comando de Ejecución ---
# Este es el comando que se ejecutará cuando el contenedor se inicie.
# Es el mismo que usamos para probar en local, asegurando que la aplicación
# sea accesible desde fuera del contenedor.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
