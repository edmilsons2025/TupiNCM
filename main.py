# -*- coding: utf-8 -*-
# main.py (VERSÃO HÍBRIDA FINAL - Refatorado com Mapa Semântico)

import os
import json
import re
import sqlite3
from typing import List, Dict, Any, Tuple

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware

# --- NLTK ---
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Baixando pacotes NLTK necessários (stopwords, punkt)...")
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    print("Download concluído.")

# --- CONFIGURAÇÃO ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'data', 'openfiscal.db')
MAPA_SEMANTICO_PATH = os.path.join(BASE_DIR, 'data', 'mapa_semantico.json') # ALTERADO

app = FastAPI(
    title="Serviço de Busca NCM (Modelo Híbrido com Mapa Semântico)",
    version="3.1.0"
)

# --- CORS ---
origins = ["http://localhost:3000", "https://seu-projeto-frontend.vercel.app"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# --- MODELOS DE DADOS ---
class NcmResult(BaseModel):
    ncm: str
    descricao: str
    score: float
    source: str # Fonte: 'mapa_semantico' ou 'similaridade'

class ApiResponse(BaseModel):
    query: str
    method: str
    count: int
    results: List[NcmResult]

# --- VARIÁVEIS GLOBAIS ---
df_ncm: pd.DataFrame = None
vectorizer: TfidfVectorizer = None
ncm_matrix: np.ndarray = None
MAPA_SEMANTICO: Dict[str, Any] = {} # ALTERADO

# --- NÚCLEO DE NLP ---
stemmer = SnowballStemmer('portuguese')
PORTUGUESE_STOPWORDS = set(stopwords.words('portuguese'))

def preprocess_text(text: str) -> str:
    text = re.sub(r'[.,:;()\[\]{}\'"/\\-]', ' ', text.lower())
    words = text.split()
    processed_words = [
        stemmer.stem(word) for word in words
        if word not in PORTUGUESE_STOPWORDS and len(word) > 2
    ]
    return " ".join(processed_words)

# --- LÓGICA DE BUSCA HÍBRIDA ---

def find_best_category(query: str) -> Tuple[str, Any]:
    """
    Encontra a melhor categoria no mapa semântico para a consulta.
    Retorna os dados da categoria.
    """
    query_lower = query.lower()
    best_category_data = None
    max_score = 0

    for term, data in MAPA_SEMANTICO.items(): # ALTERADO
        all_variants = [term] + data.get('sinonimos', [])
        score = 0
        for variant in all_variants:
            if variant in query_lower:
                score += len(variant)
        
        if score > max_score:
            max_score = score
            best_category_data = data
            
    return best_category_data

def search_by_similarity(query: str) -> List[NcmResult]:
    """
    Função de fallback que usa a similaridade de cosseno.
    """
    processed_q = preprocess_text(query)
    if not processed_q:
        return []

    query_vector = vectorizer.transform([processed_q])
    cosine_similarities = cosine_similarity(query_vector, ncm_matrix).flatten()
    related_docs_indices = cosine_similarities.argsort()[:-26:-1]
    
    results = []
    for i in related_docs_indices:
        score = cosine_similarities[i]
        if score > 0.15:
            results.append(NcmResult(
                ncm=df_ncm.iloc[i]['ncm'],
                descricao=df_ncm.iloc[i]['descricao'],
                score=round(float(score), 4),
                source='similaridade'
            ))
    return results

# --- LÓGICA DE INICIALIZAÇÃO ---
@app.on_event("startup")
def load_data_and_model():
    global df_ncm, vectorizer, ncm_matrix, MAPA_SEMANTICO # ALTERADO
    print("--- INICIANDO SERVIÇO DE BUSCA NCM (Híbrido) ---")
    
    # Carregar Mapa Semântico
    if os.path.exists(MAPA_SEMANTICO_PATH): # ALTERADO
        with open(MAPA_SEMANTICO_PATH, 'r', encoding='utf-8') as f: # ALTERADO
            mapa_data = json.load(f)
            MAPA_SEMANTICO = {item['termo_principal'].lower(): item for item in mapa_data} # ALTERADO
        print(f"Mapa semântico carregado com {len(MAPA_SEMANTICO)} termos.") # ALTERADO
    
    # Carregar Banco de Dados
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT DISTINCT ncm, descricao FROM ibpt_taxes"
    df_ncm = pd.read_sql_query(query, conn)
    conn.close()
    print(f"{len(df_ncm)} NCMs carregados.")

    # Treinar Modelo (para o fallback)
    df_ncm['descricao_processada'] = df_ncm['descricao'].apply(preprocess_text)
    vectorizer = TfidfVectorizer(analyzer='word', min_df=2, ngram_range=(1, 2))
    ncm_matrix = vectorizer.fit_transform(df_ncm['descricao_processada'])
    print("--- SERVIÇO PRONTO ---")

# --- ENDPOINT DA API ---
@app.get("/api/ncm-search", response_model=ApiResponse)
async def search_ncm(description: str = Query(..., min_length=3, max_length=100)):
    
    # Etapa 1: Tentar a busca por conhecimento (Mapa Semântico)
    best_category = find_best_category(description)
    
    if best_category:
        results = [
            NcmResult(
                ncm=sug['ncm'],
                descricao=sug['descricao'],
                score=1.0 if sug['confianca'] == 'alta' else 0.9,
                source='mapa_semantico' # ALTERADO
            )
            for sug in best_category.get('sugestoes_ncm', []) if sug['confianca'] != 'baixa'
        ]
        
        if results:
            return ApiResponse(
                query=description,
                method=f"Mapa Semântico ({best_category['termo_principal']})", # ALTERADO
                count=len(results),
                results=results
            )

    # Etapa 2: Se o mapa semântico falhar, usar a busca por similaridade (IA) como fallback
    results = search_by_similarity(description)
    
    return ApiResponse(
        query=description,
        method="Similaridade de Cosseno",
        count=len(results),
        results=results
    )