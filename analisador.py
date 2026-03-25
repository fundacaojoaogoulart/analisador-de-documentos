from dotenv import load_dotenv
import fitz
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List
import os
import json

# 1. CONFIGURAÇÃO DA API

load_dotenv()

api_key = os.getenv("API_KEY")
client = genai.Client(api_key=api_key)

# 2. DEFINIÇÃO DA ESTRUTURA DE DADOS

# O parâmetro 'description' ajuda a guiar a IA sobre o que significa cada campo.
class AnaliseDocumento(BaseModel):
    problema: str = Field(description="O problema principal ou dor que o documento tenta resolver.")
    objetivo: str = Field(description="O objetivo principal ou propósito do documento.")
    solucao: str = Field(description="A solução central proposta ou implementada para resolver o problema.")
    metodologia: str = Field(description="Resumo da metodologia, métodos ou passos utilizados.")
    profissionais_tecnicos: List[str] = Field(description="Lista com os cargos ou profissionais técnicos necessários mencionados no texto.")
    area_expertise: str = Field(description="A área macro de expertise (Ex: Engenharia de Software, Saúde Pública, Agronomia, etc).")

# 3. TEXTO DE TESTE

def extrair_texto_pdf(caminho_arquivo: str) -> str:
    print(f"Lendo o arquivo PDF: {caminho_arquivo}...")
    texto_completo = ""
    
    try:
        # Abre o PDF usando o PyMuPDF
        documento_pdf = fitz.open(caminho_arquivo)
        
        # Passa por todas as páginas e extrai o texto
        for numero_pagina in range(len(documento_pdf)):
            pagina = documento_pdf.load_page(numero_pagina)
            texto_completo += pagina.get_text("text") + "\n"
            
        documento_pdf.close()
        return texto_completo
        
    except Exception as e:
        print(f"Erro ao ler o PDF: {e}")
        return ""

# 4. EXECUÇÃO DO MODELO

def analisar_documento(texto: str):
    print("Enviando para o Gemini analisar...")
    
    # Recomendação: usar o gemini-1.5-pro para raciocínio complexo em textos longos.
    # O gemini-1.5-flash é mais barato/rápido, mas para essa etapa inicial o 'pro' é melhor.
    
    # O prompt instrui o que ele deve fazer de forma geral
    prompt = f"Leia o documento abaixo e extraia as informações solicitadas.\n\nDocumento:\n{texto}"
    
    # É aqui que a mágica acontece. Passamos o schema do Pydantic para a API.
    resposta = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AnaliseDocumento,
            ),
        )
    return resposta.text

# 5. EXIBINDO OS RESULTADOS

if __name__ == "__main__":
    # Substitua pelo nome ou caminho do seu arquivo PDF real
    caminho_do_meu_pdf = "Reveillon-em-Dados-2024.pdf" 
    
    if os.path.exists(caminho_do_meu_pdf):
        # 1. Extrai o texto do PDF
        texto_do_pdf = extrair_texto_pdf(caminho_do_meu_pdf)
        
        if texto_do_pdf.strip():
            # 2. Manda o texto para a IA
            resultado_json = analisar_documento(texto_do_pdf)
            
            # 3. Exibe o resultado
            dados_extraidos = json.loads(resultado_json)
            print("\n--- RESULTADO DA EXTRAÇÃO ---")
            print(json.dumps(dados_extraidos, indent=4, ensure_ascii=False))
            
            # 4. Salva o resultado em um arquivo
            with open('resultado.json', 'w', encoding='utf-8') as arquivo_saida:
                json.dump(dados_extraidos, arquivo_saida, indent=4, ensure_ascii=False)
            
            print("\nArquivo 'resultado.json' salvo com sucesso!")
        else:
            print("O PDF foi lido, mas nenhum texto foi encontrado (pode ser um PDF de imagens escaneadas).")
    else:
        print(f"Arquivo '{caminho_do_meu_pdf}' não encontrado. Verifique o caminho!")