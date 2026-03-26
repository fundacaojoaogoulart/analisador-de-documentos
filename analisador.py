import os
import json
import fitz  # PyMuPDF
import ollama
from pydantic import BaseModel, Field, ValidationError
from typing import List
from pathlib import Path

# 1. DEFINIÇÃO DA ESTRUTURA DE DADOS (Mantida igual)
class AnaliseDocumento(BaseModel):
    problema: str = Field(description="O problema principal ou dor que o documento tenta resolver.")
    objetivo: str = Field(description="O objetivo principal ou propósito do documento.")
    solucao: str = Field(description="A solução central proposta ou implementada para resolver o problema.")
    metodologia: str = Field(description="Resumo da metodologia, métodos ou passos utilizados.")
    profissionais_tecnicos: List[str] = Field(description="Lista com os cargos técnicos necessários para realização do projeto descrito.")
    area_expertise: str = Field(description="As áreas macro de expertise que envolvem o tema do documento.")

# 2. EXTRAÇÃO DE TEXTO DO PDF (Adicionado limpeza simples)
def extrair_texto_pdf(caminho_arquivo: Path) -> str:
    print(f"\n Lendo: {caminho_arquivo.name}...")
    texto_completo = ""
    try:
        documento_pdf = fitz.open(caminho_arquivo)
        for pagina in documento_pdf:
            texto_completo += pagina.get_text("text") + "\n"
        documento_pdf.close()
        # Limpeza básica para economizar tokens
        return " ".join(texto_completo.split()) 
    except Exception as e:
        print(f"Erro ao ler o PDF {caminho_arquivo.name}: {e}")
        return ""

# 3. EXECUÇÃO DO MODELO COM OLLAMA (Prompt Otimizado)
def analisar_documento_ollama(texto: str):
    print("Processando no Ollama (Llama3)...")
    
    # Schema simplificado para o prompt
    schema_json = AnaliseDocumento.model_json_schema()
    
    prompt = (
        f"Aja como um extrator de dados JSON especializado. "
        f"Extraia as informações do documento abaixo seguindo estritamente este schema: {schema_json}\n\n"
        f"REGRAS CRÍTICAS:\n"
        f"1. Retorne APENAS o objeto JSON puro.\n"
        f"2. Não envolva o JSON em chaves como 'documento' ou 'info'.\n"
        f"3. Use exatamente os nomes das chaves: problema, objetivo, solucao, metodologia, profissionais_tecnicos, area_expertise.\n"
        f"4. Se uma informação não for encontrada, preencha com 'Não informado'.\n\n"
        f"Documento:\n{texto[:12000]}" # Limitando caracteres para não estourar contexto do Llama3 padrão
    )
    
    try:
        resposta = ollama.chat(
            model='llama3',
            format='json', # Mantém a garantia de formato JSON
            messages=[{'role': 'user', 'content': prompt}]
        )
        return resposta['message']['content']
    except Exception as e:
        print(f"Erro na chamada ao Ollama: {e}")
        return "{}"

# 4. LOOP DE PROCESSAMENTO (Com lógica de reparo)
if __name__ == "__main__":
    DIRETORIO_RAIZ = Path(__file__).parent
    PASTA_RELATORIOS = DIRETORIO_RAIZ / "Documentos" / "Relatórios"
    PASTA_RESULTADOS = DIRETORIO_RAIZ / "Documentos" / "Resultados"

    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    arquivos_pdf = list(PASTA_RELATORIOS.glob("*.pdf"))

    if not arquivos_pdf:
        print(f"Nenhum arquivo PDF encontrado em: {PASTA_RELATORIOS}")
    else:
        for caminho_pdf in arquivos_pdf:
            texto_pdf = extrair_texto_pdf(caminho_pdf)
            
            if texto_pdf:
                resultado_raw = analisar_documento_ollama(texto_pdf)
                
                try:
                    dados_dict = json.loads(resultado_raw)
                    
                    # LOGICA DE REPARO: Se o LLM aninhou o resultado, tentamos pegar o nível interno
                    if "properties" in dados_dict: dados_dict = dados_dict["properties"]
                    if "analise" in dados_dict: dados_dict = dados_dict["analise"]
                    
                    analise_validada = AnaliseDocumento(**dados_dict)
                    
                    nome_saida = f"resultado_{caminho_pdf.stem}.json"
                    caminho_saida = PASTA_RESULTADOS / nome_saida
                    
                    with open(caminho_saida, 'w', encoding='utf-8') as f:
                        f.write(analise_validada.model_dump_json(indent=4, ensure_ascii=False))
                    
                    print(f"Sucesso: {nome_saida} salvo.")
                    
                except (json.JSONDecodeError, ValidationError) as e:
                    print(f"Erro de validação no arquivo {caminho_pdf.name}.")
                    # Debug: print(f"Raw output: {resultado_raw}") 
            else:
                print(f"Texto vazio em {caminho_pdf.name}.")

    print("\nProcessamento concluído!")