# data_updater.py

"""
Módulo de atualização de dados fiscais para o serviço TupiNCM.

Este script é responsável por popular e manter atualizado um banco de dados
SQLite local com informações fiscais de duas fontes principais:
1.  **IBPTax (IBPT):** Baixa as tabelas de alíquotas aproximadas de tributos
    por NCM para cada estado brasileiro, diretamente do repositório SVN do ACBr.
2.  **CEST (Confaz):** Realiza o scraping da página do CONFAZ para extrair
    a relação de NCMs com seus respectivos Códigos Especificadores da
    Substituição Tributária (CEST).

O script é projetado para ser robusto, utilizando sessões HTTP com retentativas,
validação de banco de dados e um mecanismo de cache para os dados do CEST
baseado em metadados (ETag/Last-Modified).

Pode ser executado como um script independente para realizar uma atualização
completa do banco de dados.
"""

import os
import json
import re
import sqlite3
import requests
from typing import List, Dict, Any
from io import StringIO
from bs4 import BeautifulSoup
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# --- Constantes de Configuração ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DB_PATH = os.path.join(DATA_DIR, 'ibpt.db')
METADATA_FILE_PATH = os.path.join(DATA_DIR, 'cest_metadata.json')

# URLS de origem dos dados
IBPT_SVN_URL = 'http://svn.code.sf.net/p/acbr/code/trunk2/Exemplos/ACBrTCP/ACBrIBPTax/tabela/'
CEST_URL = "https://www.confaz.fazenda.gov.br/legislacao/convenios/2018/CV142_18"


def get_session() -> requests.Session:
    """
    Configura e retorna uma sessão de requests com uma estratégia de retentativas.

    Isso aumenta a resiliência do script contra falhas de rede transitórias
    ao tentar baixar os dados das fontes externas.

    Returns:
        Um objeto requests.Session configurado com retentativas.
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


# --- Funções de Gerenciamento do Banco de Dados ---

def is_db_valid(db_path: str) -> bool:
    """
    Verifica se o banco de dados SQLite existe, não está vazio e contém a tabela principal.

    Args:
        db_path: O caminho para o arquivo do banco de dados SQLite.

    Returns:
        True se o banco de dados for considerado válido, False caso contrário.
    """
    if not os.path.exists(db_path) or os.path.getsize(db_path) == 0:
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Verifica se a tabela 'ibpt_taxes' existe.
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ibpt_taxes';")
        if cursor.fetchone() is None:
            conn.close()
            return False
            
        # Verifica se a tabela contém dados.
        cursor.execute("SELECT COUNT(*) FROM ibpt_taxes;")
        if cursor.fetchone()[0] == 0:
            conn.close()
            return False

        conn.close()
        return True

    except sqlite3.DatabaseError:
        # Retorna False se o arquivo estiver corrompido ou não for um DB válido.
        return False
    
def get_db_connection(db_path: str) -> sqlite3.Connection:
    """
    Estabelece e retorna uma conexão com o banco de dados SQLite.

    Cria o diretório 'data' se ele não existir e configura o modo de jornalismo
    para WAL (Write-Ahead Logging) para melhor desempenho de escrita e concorrência.

    Args:
        db_path: O caminho para o arquivo do banco de dados.

    Returns:
        Um objeto de conexão sqlite3.Connection.
    """
    data_dir = os.path.dirname(db_path)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Diretório '{data_dir}' criado.")
        
    conn = sqlite3.connect(db_path)
    # Habilita o modo WAL para melhor performance de escrita.
    conn.execute('PRAGMA journal_mode = WAL')
    return conn

def criar_tabelas(conn: sqlite3.Connection):
    """
    Cria o esquema do banco de dados, incluindo tabelas, índices e gatilhos.

    Define as tabelas `ibpt_taxes` e `cest_data`, cria índices para otimizar
    consultas e configura uma tabela virtual FTS5 para busca textual eficiente,
    juntamente com gatilhos para manter o índice de busca sincronizado.

    Args:
        conn: A conexão ativa com o banco de dados.
    """
    print('Verificando e criando tabelas, se necessário...')
    cursor = conn.cursor()
    
    # Tabela principal para dados do IBPT
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ibpt_taxes (
            ncm TEXT NOT NULL, uf TEXT NOT NULL, ex TEXT, tipo TEXT,
            descricao TEXT, aliqNacional REAL, aliqEstadual REAL,
            aliqMunicipal REAL, aliqImportado REAL, vigenciaInicio TEXT,
            vigenciaFim TEXT, chave TEXT, versao TEXT, fonte TEXT,
            PRIMARY KEY (ncm, uf)
        );
    """)
    # Tabela para dados do CEST
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cest_data (
            cest TEXT NOT NULL, ncm TEXT NOT NULL, descricao TEXT,
            PRIMARY KEY (cest, ncm)
        );
    """)
    # Índices para otimizar consultas
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_ncm_ibpt ON ibpt_taxes (ncm);')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_ncm_cest ON cest_data (ncm);')
    
    # Tabela virtual FTS5 para busca textual rápida e eficiente.
    try:
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS ibpt_search USING fts5(
                ncm, descricao, content='ibpt_taxes', content_rowid='rowid',
                tokenize = 'porter unicode61'
            );
        """)
        
        # Gatilhos para sincronizar automaticamente a tabela FTS5 com a tabela ibpt_taxes.
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS ibpt_taxes_after_insert AFTER INSERT ON ibpt_taxes BEGIN
                INSERT INTO ibpt_search(rowid, ncm, descricao) VALUES (new.rowid, new.ncm, new.descricao);
            END;
        """)
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS ibpt_taxes_after_delete AFTER DELETE ON ibpt_taxes BEGIN
                DELETE FROM ibpt_search WHERE rowid = old.rowid;
            END;
        """)
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS ibpt_taxes_after_update AFTER UPDATE ON ibpt_taxes BEGIN
                DELETE FROM ibpt_search WHERE rowid = old.rowid;
                INSERT INTO ibpt_search(rowid, ncm, descricao) VALUES (new.rowid, new.ncm, new.descricao);
            END;
        """)
        print('Tabelas, índices e FTS5 (busca textual) configurados.')
        
    except sqlite3.OperationalError as e:
        # Fallback caso a extensão FTS5 não esteja habilitada no SQLite.
        print(f"AVISO: FTS5 não pôde ser criado: {e}. A busca dependerá do modelo do FastAPI.")

    conn.commit()


# --- Lógica de Processamento e Atualização de Dados ---

def processar_ibpt(session: requests.Session, conn: sqlite3.Connection):
    """
    Realiza o scraping, download e processamento dos dados do IBPTax.

    Limpa a tabela existente, navega pelo repositório SVN do ACBr, baixa
    cada arquivo CSV de UF, processa os dados com Pandas e os insere
    no banco de dados. Ao final, reconstrói o índice FTS5.

    Args:
        session: A sessão de requests a ser utilizada para os downloads.
        conn: A conexão ativa com o banco de dados.
    """
    print('Iniciando processamento dos dados do IBPT...')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM ibpt_taxes;')

    insert_sql = """
        INSERT OR REPLACE INTO ibpt_taxes (ncm, uf, ex, tipo, descricao, aliqNacional, 
        aliqEstadual, aliqMunicipal, aliqImportado, vigenciaInicio, vigenciaFim, 
        chave, versao, fonte) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    try:
        response = session.get(IBPT_SVN_URL)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        csv_files = [link.get('href') for link in soup.find_all('a', href=re.compile(r'\.csv$'))]

        for file in csv_files:
            match = re.search(r'TabelaIBPTax([A-Z]{2})', file)
            if not match:
                continue
            uf = match.group(1)
            
            print(f"Processando IBPT para a UF: {uf}...")
            
            file_url = IBPT_SVN_URL + file
            csv_response = session.get(file_url)
            csv_response.raise_for_status()
            
            # Decodifica usando 'iso-8859-1', comum em arquivos legados brasileiros.
            csv_content = csv_response.content.decode('iso-8859-1')
            
            df = pd.read_csv(
                StringIO(csv_content), sep=';', header=None, skiprows=1,
                encoding='iso-8859-1', dtype=str
            )
            df.columns = ['codigo', 'ex', 'tipo', 'descricao', 'nacionalfederal', 
                          'importadosfederal', 'estadual', 'municipal', 'vigenciainicio', 
                          'vigenciafim', 'chave', 'versao', 'fonte']

            def clean_float(val):
                return float(str(val or '0').replace(',', '.'))

            data_to_insert = [
                (
                    str(row['codigo'] or '').replace('.', ''), uf, row['ex'], row['tipo'], row['descricao'],
                    clean_float(row['nacionalfederal']), clean_float(row['estadual']),
                    clean_float(row['municipal']), clean_float(row['importadosfederal']),
                    row['vigenciainicio'], row['vigenciafim'], row['chave'], row['versao'], row['fonte']
                ) for _, row in df.iterrows()
            ]

            cursor.executemany(insert_sql, data_to_insert)
            conn.commit()
            print(f"{len(data_to_insert)} registros inseridos para {uf}.")
        
        # Após a inserção em massa, reconstrói o índice FTS5 para garantir consistência.
        try:
            print('Reconstruindo índice de busca FTS5...')
            cursor.execute("INSERT INTO ibpt_search(ibpt_search) VALUES('rebuild');")
            conn.commit()
            print('Índice de busca FTS5 reconstruído.')
        except sqlite3.OperationalError:
            pass # Ignora se FTS5 não existe.

        print('Processamento do IBPT concluído.')
    except requests.RequestException as e:
        print(f"ERRO DE REDE ao processar IBPT: {e}")
    except Exception as e:
        print(f"ERRO INESPERADO ao processar IBPT: {e}")


def processar_cest(session: requests.Session, conn: sqlite3.Connection):
    """
    Realiza o scraping e processamento dos dados de CEST do CONFAZ.

    Verifica se houve atualização na fonte de dados remota comparando
    cabeçalhos HTTP (ETag, Last-Modified) com metadados locais. Se houver
    mudanças, baixa e processa a página, inserindo os dados no banco.

    Args:
        session: A sessão de requests a ser utilizada.
        conn: A conexão ativa com o banco de dados.
    """
    print('Iniciando verificação de atualização do CEST...')
    
    # Carrega metadados locais para verificar a necessidade de atualização.
    metadata = {}
    if os.path.exists(METADATA_FILE_PATH):
        with open(METADATA_FILE_PATH, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

    try:
        head = session.head(CEST_URL, verify=False, timeout=10)
        remote_etag = head.headers.get('etag')
        remote_last_modified = head.headers.get('last-modified')

        if (remote_etag and remote_etag == metadata.get('etag')) or \
           (remote_last_modified and remote_last_modified == metadata.get('lastModified')):
            print('Tabela CEST não foi modificada desde a última verificação. Pulando.')
            return

        print('Nova versão da tabela CEST encontrada. Iniciando download...')
        response = session.get(CEST_URL, verify=False, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        data_to_insert = []

        for table in soup.find_all('table'):
            for i, row in enumerate(table.find_all('tr')):
                if i == 0: continue # Pula o cabeçalho da tabela
                
                cells = row.find_all('td')
                if len(cells) >= 4:
                    cest = cells[1].get_text(strip=True).replace('.', '')
                    ncms = re.split(r'[\s,;]+', cells[2].get_text(strip=True))
                    description = re.sub(r'\s{2,}', ' ', cells[3].get_text(strip=True))
                    
                    for ncm in ncms:
                        clean_ncm = re.sub(r'[^\d]', '', ncm)
                        if clean_ncm:
                            data_to_insert.append((cest, clean_ncm, description))

        if not data_to_insert:
            print("AVISO: Nenhum item CEST foi extraído da página.")
            return

        cursor = conn.cursor()
        cursor.execute('DELETE FROM cest_data;')
        cursor.executemany(
            'INSERT OR IGNORE INTO cest_data (cest, ncm, descricao) VALUES (?, ?, ?)',
            data_to_insert
        )
        conn.commit()
        print(f"{cursor.rowcount} combinações CEST-NCM inseridas.")
        
        # Salva os novos metadados para controle de versão.
        new_metadata = {
            'etag': remote_etag, 'lastModified': remote_last_modified,
            'lastUpdate': pd.Timestamp.now(tz='UTC').isoformat()
        }
        with open(METADATA_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(new_metadata, f, ensure_ascii=False, indent=2)
        print('Metadados de controle do CEST foram atualizados.')

    except requests.RequestException as e:
        print(f"ERRO DE REDE ao processar CEST: {e}")
    except Exception as e:
        print(f"ERRO INESPERADO ao processar CEST: {e}")


def atualizar_banco_dados():
    """
    Orquestra o processo completo de atualização do banco de dados.

    Inicializa a conexão, cria o esquema de tabelas e chama as funções
    de processamento para IBPT e CEST em sequência, garantindo que a
    conexão com o banco seja fechada ao final.
    """
    conn = None
    try:
        session = get_session()
        conn = get_db_connection(DB_PATH)
        criar_tabelas(conn)
        
        processar_ibpt(session, conn)
        processar_cest(session, conn)
        
    except Exception as e:
        print(f"ERRO CRÍTICO no processo de atualização: {e}")
    finally:
        if conn:
            conn.close()
            print("Conexão com o banco de dados fechada após atualização.")


if __name__ == '__main__':
    print("Iniciando atualização manual do banco de dados fiscal...")
    atualizar_banco_dados()
    print("Atualização concluída.")