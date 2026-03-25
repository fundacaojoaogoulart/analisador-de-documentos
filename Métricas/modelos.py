import os
from dotenv import load_dotenv
from google import genai
from pathlib import Path

# 1. Ajuste de Caminho para o .env
# Se o script está em 'Métricas/', subimos um nível para achar o '.env' na raiz
BASE_DIR = Path(__file__).parent.parent
load_dotenv(dotenv_path=BASE_DIR / '.env')

# 2. Inicialização
api_key = os.getenv("API_KEY")
if not api_key:
    print("ERRO: API_KEY não encontrada no arquivo .env")
    exit()

client = genai.Client(api_key=api_key)

print(f"{'NOME DO MODELO':<45} | {'SUPORTA EMBEDDING?'}")
print("-" * 70)

try:
    # O segredo: iterar sobre a lista de modelos da SDK
    for model in client.models.list():
        # Verificamos os métodos suportados (geralmente em model.supported_methods)
        metodos = getattr(model, 'supported_methods', [])
        
        # O Google às vezes retorna 'embedContent' ou 'batchEmbedContents'
        pode_embed = "SIM" if any("embed" in m.lower() for m in metodos) else "Não"
        
        print(f"{model.name:<45} | {pode_embed}")

except Exception as e:
    print(f"Erro ao listar modelos: {e}")