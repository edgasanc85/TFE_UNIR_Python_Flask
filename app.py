import os
import openai
from flask import Flask, request, jsonify
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

app = Flask(__name__)

# Datos de Conexión
openai.api_type = "azure"
openai.api_version = "2024-02-01"

os.environ["AZURE_OPENAI_ENDPOINT"] = "https://edgasancdemo.openai.azure.com/"
os.environ["AZURE_OPENAI_API_KEY"] = "57a41271b75040089f77ac694022a12d"

# Configuración y carga inicial
embeddings = AzureOpenAIEmbeddings(
    azure_deployment="ucs-embedding-large",
    openai_api_version="2024-02-01",
)

# Montar la base de datos CHROMADB
vectordb = Chroma(persist_directory="/var/www/flaskapp/chroma_db_ac_2024", embedding_function=embeddings)
retriever = vectordb.as_retriever(
    search_kwargs={"k": 3}
)

chat = AzureChatOpenAI(
    openai_api_version="2024-02-01",
    azure_deployment="ucs-chat-gpt",
    temperature=0.0
)

qa_chain = RetrievalQA.from_chain_type(
    llm=chat,
    chain_type="stuff",
    retriever=retriever
)

@app.route('/query', methods=['POST'])
def query_qa_chain():
    data = request.get_json()
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        result = qa_chain.run(query)
        return jsonify({"query": query, "result": result}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
