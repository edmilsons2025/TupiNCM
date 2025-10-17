# main.py

"""
Serviço de API para busca híbrida de Nomenclatura Comum do Mercosul (NCM).

Este serviço utiliza uma abordagem em duas etapas:
1.  **Mapa Semântico:** Uma busca de alta precisão baseada em um dicionário
    de conhecimento pré-definido para termos comuns.
2.  **Similaridade de Cosseno:** Um modelo de fallback que utiliza TF-IDF para
    encontrar os NCMs mais relevantes com base na similaridade textual.

Além disso, expõe um endpoint para receber feedback dos usuários, permitindo
o aprimoramento contínuo do mapa semântico.
"""

import os
import json
import re
import sqlite3
from typing import List, Dict, Any, Tuple
from datetime import datetime, timezone

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# --- Pacotes de dados NLTK ---
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Baixando pacotes NLTK necessários (stopwords, punkt)...")
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    print("Download concluído.")

try:
    import data_updater
except ImportError:
    from . import data_updater

# --- Constantes de Configuração ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DB_PATH = os.path.join(DATA_DIR, 'ibpt.db')
MAPA_SEMANTICO_PATH = os.path.join(DATA_DIR, 'mapa_semantico.json')
USER_SUGGESTIONS_PATH = os.path.join(DATA_DIR, 'user_suggestions.jsonl')


# --- Inicialização da Aplicação FastAPI ---
app = FastAPI(
    title="Serviço de Busca NCM (Modelo Híbrido)",
    version="3.2.0",
    description="API para busca inteligente de NCMs utilizando um mapa semântico e similaridade de cosseno."
)

# --- Configuração de CORS ---
origins = ["http://localhost:3000", "https://sani-ia.vercel.app", "http://192.168.10.117:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Modelos de Dados (Pydantic) ---
class NcmResult(BaseModel):
    """Representa um único resultado de busca NCM."""
    ncm: str = Field(..., description="O código NCM.", example="8473.30.41")
    descricao: str = Field(..., description="A descrição oficial do NCM.", example="Placas-mãe (motherboards)")
    score: float = Field(..., description="Pontuação de relevância (0.0 a 1.0).", example=0.95)
    source: str = Field(..., description="Origem do resultado ('mapa_semantico' ou 'similaridade').", example="mapa_semantico")

class ApiResponse(BaseModel):
    """Define a estrutura da resposta da API de busca."""
    query: str = Field(..., description="O termo de busca original.", example="placa principal")
    method: str = Field(..., description="O método utilizado para encontrar os resultados.", example="Mapa Semântico (placa de circuito impresso)")
    count: int = Field(..., description="O número de resultados retornados.", example=2)
    results: List[NcmResult]

class NcmSuggestion(BaseModel):
    """Define a estrutura para receber sugestões de classificação dos usuários."""
    original_query: str = Field(..., description="O termo de busca original que falhou.", example="Gel de Silicone")
    ncm: str = Field(..., description="O código NCM que o usuário selecionou manualmente.", example="3910.00.90")
    descricao: str = Field(..., description="A descrição do NCM selecionado.", example="Silicones em formas primárias")


# --- Variáveis Globais para o Modelo e Dados ---
df_ncm: pd.DataFrame = None
vectorizer: TfidfVectorizer = None
ncm_matrix: np.ndarray = None
MAPA_SEMANTICO: Dict[str, Any] = {}


# --- Lógica de Processamento de Linguagem Natural (NLP) ---
stemmer = SnowballStemmer('portuguese')
PORTUGUESE_STOPWORDS = set(stopwords.words('portuguese'))

def preprocess_text(text: str) -> str:
    """
    Limpa e normaliza o texto para o modelo de similaridade.
    Inclui remoção de pontuação, conversão para minúsculas, remoção de stopwords
    e aplicação de stemming.
    """
    text = re.sub(r'[.,:;()\[\]{}\'"/\\-]', ' ', text.lower())
    words = text.split()
    processed_words = [
        stemmer.stem(word) for word in words
        if word not in PORTUGUESE_STOPWORDS and len(word) > 2
    ]
    return " ".join(processed_words)


# --- Lógica de Busca Híbrida ---
def find_best_category(query: str) -> Dict[str, Any]:
    """
    Busca no Mapa Semântico a categoria que melhor corresponde à consulta.
    Utiliza um sistema de pontuação ponderado para avaliar a correspondência
    com o termo principal, sinônimos e termos técnicos.
    """
    query_lower = query.lower()
    processed_query = re.sub(r'[,.\-]', ' ', query_lower)
    query_words = set(processed_query.split())

    best_category_data = None
    max_score = 0

    for term, data in MAPA_SEMANTICO.items():
        score = 0
        
        def get_match_score(variants: List[str], weight: int) -> float:
            match_score = 0
            for variant in variants:
                variant_words = set(variant.lower().split())
                intersection = query_words.intersection(variant_words)
                if intersection:
                    match_score += sum(len(word) for word in intersection)
            return match_score * weight

        score += get_match_score([term], 3)
        score += get_match_score(data.get('sinonimos', []), 2)
        score += get_match_score(data.get('termos_tecnicos_relacionados', []), 1)

        if score > max_score:
            max_score = score
            best_category_data = data

    return best_category_data

def search_by_similarity(query: str) -> List[NcmResult]:
    """
    Realiza a busca por similaridade de cosseno como fallback.
    Transforma a consulta em um vetor TF-IDF e calcula a similaridade com
    a matriz de NCMs pré-processada.
    """
    if df_ncm is None or vectorizer is None or ncm_matrix is None:
        return []

    processed_q = preprocess_text(query)
    if not processed_q:
        return []

    query_vector = vectorizer.transform([processed_q])
    cosine_similarities = cosine_similarity(query_vector, ncm_matrix).flatten()
    related_docs_indices = cosine_similarities.argsort()[:-26:-1] # Top 25 resultados

    results = []
    for i in related_docs_indices:
        score = cosine_similarities[i]
        if score > 0.15: # Limiar de confiança
            results.append(NcmResult(
                ncm=df_ncm.iloc[i]['ncm'],
                descricao=df_ncm.iloc[i]['descricao'],
                score=round(float(score), 4),
                source='similaridade'
            ))
    return results


# --- Lógica de Inicialização do Serviço ---
@app.on_event("startup")
def load_data_and_model():
    """
    Executa na inicialização da API. Carrega o mapa semântico, verifica e
    atualiza o banco de dados se necessário, e treina o modelo TF-IDF.
    """
    global df_ncm, vectorizer, ncm_matrix, MAPA_SEMANTICO
    print("--- INICIANDO SERVIÇO DE BUSCA NCM (Híbrido) ---")

    # 1. Carregar Mapa Semântico
    if os.path.exists(MAPA_SEMANTICO_PATH):
        with open(MAPA_SEMANTICO_PATH, 'r', encoding='utf-8') as f:
            mapa_data = json.load(f)
        MAPA_SEMANTICO = {item['termo_principal'].lower(): item for item in mapa_data}
        print(f"Mapa semântico carregado com {len(MAPA_SEMANTICO)} termos.")
    else:
        print("AVISO: Mapa semântico não encontrado! A busca funcionará apenas por similaridade.")

    # 2. Verificar e Atualizar Banco de Dados
    if not os.path.exists(DB_PATH) or not data_updater.is_db_valid(DB_PATH):
        print(">>> Banco de dados ausente ou inválido. Executando atualização... <<<")
        data_updater.atualizar_banco_dados()
        print(">>> Atualização do banco de dados concluída. <<<")

    # 3. Carregar Dados do Banco para o DataFrame
    try:
        conn = sqlite3.connect(DB_PATH)
        query = "SELECT DISTINCT ncm, descricao FROM ibpt_taxes"
        df_ncm = pd.read_sql_query(query, conn)
        conn.close()
        
        if df_ncm.empty:
            raise ValueError("O banco de dados está vazio após o carregamento.")

        print(f"Carregamento final: {len(df_ncm)} NCMs únicos carregados.")
    except Exception as e:
        print(f"ERRO FATAL: Falha ao carregar dados do banco de dados: {e}")
        raise HTTPException(status_code=503, detail="Serviço indisponível: Falha crítica no carregamento dos dados NCM.") from e

    # 4. Treinar Modelo de Similaridade (TF-IDF)
    df_ncm['descricao_processada'] = df_ncm['descricao'].apply(preprocess_text)
    vectorizer = TfidfVectorizer(analyzer='word', min_df=2, ngram_range=(1, 2))
    ncm_matrix = vectorizer.fit_transform(df_ncm['descricao_processada'])
    print("--- SERVIÇO PRONTO ---")


# --- Endpoints da API ---
@app.get("/api/ncm-search", response_model=ApiResponse, summary="Busca NCM por descrição")
async def search_ncm(description: str = Query(..., min_length=3, max_length=100, description="Descrição do produto a ser classificado.")):
    """
    Realiza a busca de NCM utilizando a abordagem híbrida.
    Primeiro tenta a correspondência exata com o mapa semântico. Se não houver
    sucesso, utiliza o modelo de similaridade de cosseno como fallback.
    """
    # Etapa 1: Busca por conhecimento (Mapa Semântico)
    best_category = find_best_category(description)

    if best_category:
        results = [
            NcmResult(
                ncm=sug['ncm'],
                descricao=sug['descricao'],
                score=1.0 if sug.get('confianca') == 'alta' else 0.9,
                source='mapa_semantico'
            )
            for sug in best_category.get('sugestoes_ncm', [])
            if sug.get('confianca') != 'baixa'
        ]

        if results:
            return ApiResponse(
                query=description,
                method=f"Mapa Semântico ({best_category.get('termo_principal', 'N/A')})",
                count=len(results),
                results=results
            )

    # Etapa 2: Fallback para busca por similaridade
    results = search_by_similarity(description)
    return ApiResponse(
        query=description,
        method="Similaridade de Cosseno",
        count=len(results),
        results=results
    )

@app.post("/api/ncm-suggestion", status_code=201, summary="Registra uma sugestão de NCM do usuário")
async def add_suggestion(suggestion: NcmSuggestion):
    """
    Recebe e armazena uma sugestão de classificação de NCM feita por um usuário.
    As sugestões são salvas em um arquivo `.jsonl` para revisão posterior por um administrador,
    criando um ciclo de feedback para aprimorar o sistema.
    """
    try:
        suggestion_record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "original_query": suggestion.original_query,
            "selected_ncm": suggestion.ncm,
            "selected_descricao": suggestion.descricao,
            "add_by": "user",
            "status": "pending_review"
        }

        with open(USER_SUGGESTIONS_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(suggestion_record, ensure_ascii=False) + '\n')

        return {"message": "Sugestão recebida com sucesso. Obrigado por contribuir!"}

    except Exception as e:
        print(f"ERRO ao salvar sugestão: {e}")
        raise HTTPException(
            status_code=500,
            detail="Ocorreu um erro interno ao tentar salvar a sugestão."
        )