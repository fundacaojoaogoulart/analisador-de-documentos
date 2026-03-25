import os
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# 1. Configuração de Caminhos
BASE_DIR = Path(__file__).parent.parent
PATH_RESULTADOS = BASE_DIR / "Documentos" / "Resultados"
PATH_GABARITOS = BASE_DIR / "Documentos" / "Gabarito Humano"

# 2. Carrega o Modelo (Multilíngue)
# Na primeira vez ele vai baixar o modelo (~400MB), depois é instantâneo.
print("Carregando modelo de IA local (Sentence-BERT)...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def calcular_cosseno(vetor_a, vetor_b):
    """Calcula a similaridade entre os vetores."""
    a = np.array(vetor_a)
    b = np.array(vetor_b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norm) if norm != 0 else 0.0

def avaliar():
    if not PATH_RESULTADOS.exists() or not PATH_GABARITOS.exists():
        print("Erro: Verifique se as pastas 'Documentos/Resultado' e 'Gabarito Humano' existem.")
        return

    arquivos_res = sorted(list(PATH_RESULTADOS.glob("*.json")))
    arquivos_gab = sorted(list(PATH_GABARITOS.glob("*.json")))

    print(f"\n{'ARQUIVO':<25} | {'CAMPO':<20} | {'SCORE'}")
    print("-" * 65)

    for p_res, p_gab in zip(arquivos_res, arquivos_gab):
        with open(p_res, 'r', encoding='utf-8') as f1, open(p_gab, 'r', encoding='utf-8') as f2:
            ia = json.load(f1)
            gab = json.load(f2)

            for campo in gab.keys():
                # Converte listas em strings se necessário
                txt_ia = " ".join(ia[campo]) if isinstance(ia.get(campo), list) else str(ia.get(campo, ""))
                txt_gab = " ".join(gab[campo]) if isinstance(gab.get(campo), list) else str(gab.get(campo, ""))

                # Gera o embedding LOCALMENTE
                v_ia = model.encode(txt_ia)
                v_gab = model.encode(txt_gab)

                score = calcular_cosseno(v_ia, v_gab)
                print(f"{p_res.name[:25]:<25} | {campo:<20} | {score:.4f}")

if __name__ == "__main__":
    avaliar()