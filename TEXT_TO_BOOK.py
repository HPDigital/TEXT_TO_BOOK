"""
TEXT_TO_BOOK
"""

#!/usr/bin/env python
# coding: utf-8

# In[72]:


import os
import numpy as np
import faiss
import openai
from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Obtener la API Key desde las variables de entorno
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("No se encontró la API Key de OpenAI. Asegúrate de que el archivo .env está configurado correctamente.")

# Inicializar la API de OpenAI
openai.api_key = openai_api_key

# Funciones auxiliares
def leer_libros(directorio):
    libros = {}
    for archivo in os.listdir(directorio):
        if archivo.endswith(".txt"):
            ruta = os.path.join(directorio, archivo)
            try:
                with open(ruta, 'r', encoding='utf-8') as file:
                    libros[archivo] = file.read()
            except Exception as e:
                print(f"Error al leer el archivo {archivo}: {e}")
    return libros

def procesar_libros(libros):
    splitter = CharacterTextSplitter(chunk_size=2000)
    procesados = {}
    for nombre, texto in libros.items():
        procesados[nombre] = splitter.split_text(texto)
    return procesados

def vectorizar_libros(libros, modelo):
    embeddings, metadata = [], []
    for nombre, fragmentos in libros.items():
        for fragmento in fragmentos:
            try:
                emb = modelo.embed_documents([fragmento])[0]
                embeddings.append(emb)
                metadata.append({'libro': nombre, 'fragmento': fragmento})
            except Exception as e:
                print(f"Error al vectorizar fragmento de {nombre}: {e}")
    return np.array(embeddings), metadata

def crear_indice_faiss(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def buscar_en_faiss(query, index, modelo, metadata, k=5):
    query_embedding = modelo.embed_query(query)
    distances, indices = index.search(np.array([query_embedding]), k)
    resultados = [(metadata[i], distances[0][j]) for j, i in enumerate(indices[0])]
    return resultados

# Nueva función para combinar las respuestas en un texto fluido
def combinar_resultados_en_un_texto(resultados):
    texto_combinado = "A continuación se presenta una recopilación de fragmentos relacionados con tu consulta:\n\n"
    for i, (resultado, distancia) in enumerate(resultados):
        if i == 0:
            texto_combinado += f"Comenzando con un extracto del libro *{resultado['libro']}*, encontramos:\n"
        else:
            texto_combinado += f"\nSiguiendo en el libro *{resultado['libro']}*, se observa lo siguiente:\n"
        texto_combinado += f"{resultado['fragmento'][:500]}...\n"  # Limitar a los primeros 500 caracteres

    texto_combinado += "\nEn resumen, los fragmentos anteriores proporcionan una perspectiva completa sobre el tema investigado."
    return texto_combinado

# Nueva función para generar un texto académico usando OpenAI API con la nueva interfaz 1.0.0+
def generar_texto_academico_con_openai(query, texto_completo):
    # Crear el prompt usando el texto completo
    prompt = (f"Usando el siguiente texto completo y tu propia base de datos, "
              f"elabora un texto académico completo y detallado explicando qué es '{query}', "
              f"en qué consiste el concepto de '{query}', "
              f"y proporciona un ejemplo de '{query}':\n\n")

    # Añadir el texto completo al prompt (limitado a 2500 caracteres si es necesario)
    prompt += f"{texto_completo[:3000]}\n\n"  # Ajusta el límite de caracteres si es necesario

    try:
        # Usar el método correcto con la nueva API de OpenAI
        response = openai.chat.completions.create(
            model="gpt-4",  # Cambia a "gpt-3.5-turbo" si no tienes acceso a GPT-4
            messages=[
                {"role": "system", "content": "Eres un experto en la creación de textos académicos. Proporciona respuestas detalladas, claras, y cubre el tema a profundidad. Usa una combinación de los textos adjuntos y tu base de datos."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=6000,
            temperature=0.7
        )
        # Acceder a la respuesta correctamente
        return response.choices[0].message.content
    except Exception as e:
        return f"Error al generar el texto académico: {e}"



# Procesamiento principal
def main(directorio_libros):
    libros = leer_libros(directorio_libros)
    libros_procesados = procesar_libros(libros)
    modelo_embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    embeddings, metadata = vectorizar_libros(libros_procesados, modelo_embeddings)
    index = crear_indice_faiss(embeddings)

    # Realizar la búsqueda
    query = "Jobs to be done"
    resultados = buscar_en_faiss(query, index, modelo_embeddings, metadata)

    # Combinar los resultados en un solo texto fluido
    texto_final = combinar_resultados_en_un_texto(resultados)
    #print("Texto combinado:\n", texto_final)

    # Generar un texto académico usando OpenAI API
    texto_academico = generar_texto_academico_con_openai(query, texto_final)
    print("\nTexto académico generado por OpenAI:\n", texto_academico)

if __name__ == "__main__":
    main(r"C:\Users\HP\Desktop\CATO-CURSOS-2-2024\GER-TI CATO-2-2024\Cursos\SEMANA 8\TEXTOS")




# In[ ]:




