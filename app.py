from flask import Flask, request
from flask_cors import CORS
# Importa el módulo os para interactuar con el sistema operativo  
import os
import openai
from flask import jsonify

from PyPDF2 import PdfReader
from PyPDF2 import PdfMerger
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
# Importa get_openai_callback del módulo langchain.callbacks para obtener realimentación de OpenAI
from langchain_community.callbacks.manager import get_openai_callback
#from bs4 import BeautifulSoup
# Importa OpenAI del módulo langchain.llms para interactuar con el modelo de lenguaje de OpenAI
from langchain_community.llms import OpenAI
# Importa OpenAIEmbeddings del módulo langchain.embeddings.openai para generar incrustaciones de texto utilizando OpenAI
from langchain_community.embeddings  import OpenAIEmbeddings
# Importa la función load_dotenv del módulo dotenv para cargar variables de entorno desde un archivo .env
from dotenv import load_dotenv  



app = Flask(__name__)
CORS(app) 


# Carga las variables de entorno desde un archivo .env
load_dotenv()
openai.api_key=os.environ.get("OPENAI_API_KEY")

base_conocimiento = None



@app.route('/cargar', methods=['POST'])
def upload_file():
    global base_conocimiento

    print("comenzando")

    # Verificar si se envio mas de un archivo PDF para unirlos en uno solo
    files = request.files.getlist('file')
    print(files)
    if len(files) > 1:
        print("leyendo")
        
        merger = PdfMerger()
        for pdf in files:
            merger.append(pdf)
     
        merged_filename = 'merged_file.pdf'
        merger.write(merged_filename)
        merger.close()
       
        base_conocimiento = create_embeddings(merged_filename)
        print('Archivos PDF unidos y procesados correctamente.')
        return jsonify("Archivos PDF unidos y procesados correctamente.")
    else:
        # Si llega solo un archivo PDF, procesarlo normalmente
        file = files[0]
        base_conocimiento = create_embeddings(file)
        print('Archivo PDF procesado correctamente.')
        return jsonify("Archivo PDF procesado correctamente.")
   



@app.route('/preguntapdf', methods=['POST'])
def pregunta():
    
    pregunta = request.form['texto']
        
    respuesta = envioGPT(base_conocimiento , pregunta)
    print(respuesta)

    return jsonify(respuesta)



@app.route('/clasificar', methods=['POST'])
def classify_text():
    
    texto = request.form['text']
    # Inicializa el modelo de LangChain con la temperatura especificada
    llm = OpenAI(model_name='gpt-4-turbo', temperature= 0)

    # Generar una respuesta utilizando el LLM de LangChain y el parámetro
    prompt = f"""Eres experto determinando a que tema pertence en un contenido de cualquier longitud y puedes hacer lo siguiente:
                1. -En el texto: {texto} vas identificar implicitamente el contenido y el número al que corresponde.
                2. -Una vez hayas identificado  ambas partes , vas a ser capaz de identificar en el contenido a que etiquetas pertenecen: Cine, Política o Religión.
                3. -Tu respuesta va a ser dando el número y la etiqueta  a la que  pertenece ejemplo: 1 Política."""
    response = llm(prompt)
    print(response)
    
    return jsonify({'respuesta': response})





def create_embeddings(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    #extraer el texto de cada página
    for page in pdf_reader.pages:
        text += page.extract_text()
        
    # Divide el texto en trozos usando langchain
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
        )        
    chunks = text_splitter.split_text(text)
      # Convierte los trozos de texto en incrustaciones para formar una base de conocimientos
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    # crea la base de conocimientos
    base_conocimiento = FAISS.from_texts(chunks, embeddings)

    return base_conocimiento



def envioGPT(base_conocimiento, pregunta):

    # de la base de conocimiento escoge los 4 pdf más relevantes
    paragraph = base_conocimiento.similarity_search(pregunta, 4)
    #modelo de OPENAI
    llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"), model_name='gpt-3.5-turbo-instruct', temperature= 0)
     # Carga la cadena de preguntas y respuestas
    chain = load_qa_chain(llm, chain_type="stuff")
    with get_openai_callback() as cost:
        respuesta = chain.run(input_documents=paragraph, question=pregunta)
        print(cost)
        
    return respuesta


if __name__ == '__main__':
    # Cambia la dirección IP aquí
    app.run(debug=True, host='0.0.0.0', port=5000)



  
