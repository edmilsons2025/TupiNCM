import os
import json
import re
import sqlite3
import requests
from typing import List, Dict, Any, Tuple
from io import StringIO

from bs4 import BeautifulSoup
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import csv


# --- CONFIGURAÇÃO ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Note que a pasta 'data' será criada se não existir.
DB_PATH = os.path.join(BASE_DIR, 'data', 'ibpt.db')
METADATA_FILE_PATH = os.path.join(BASE_DIR, 'data', 'cest_metadata.json')

# URLS
IBPT_SVN_URL = 'http://svn.code.sf.net/p/acbr/code/trunk2/Exemplos/ACBrTCP/ACBrIBPTax/tabela/'
CEST_URL = "https://www.confaz.fazenda.gov.br/legislacao/convenios/2018/CV142_18"


# --- Configuração de Sessão HTTP com Retentativas ---
def get_session():
    """Configura uma sessão de requests com retentativas para maior robustez."""
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


# --- FUNÇÕES DE BANCO DE DADOS ---
def get_db_connection(db_path: str) -> sqlite3.Connection:
    """Retorna uma conexão SQLite e cria o diretório 'data/' se não existir."""
    data_dir = os.path.dirname(db_path)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Diretório '{data_dir}' criado.")
        
    conn = sqlite3.connect(db_path)
    conn.execute('PRAGMA journal_mode = WAL')
    return conn

def criar_tabelas(conn: sqlite3.Connection):
    """Cria ou verifica a existência das tabelas IBPT e CEST."""
    print('Verificando e criando tabelas, se necessário...')
    
    # Tabela principal de impostos do IBPT
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ibpt_taxes (
          ncm TEXT NOT NULL,
          uf TEXT NOT NULL,
          ex TEXT,
          tipo TEXT,
          descricao TEXT,
          aliqNacional REAL,
          aliqEstadual REAL,
          aliqMunicipal REAL,
          aliqImportado REAL,
          vigenciaInicio TEXT,
          vigenciaFim TEXT,
          chave TEXT,
          versao TEXT,
          fonte TEXT,
          PRIMARY KEY (ncm, uf)
        );
    """)
    # Tabela de dados do CEST
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cest_data (
          cest TEXT NOT NULL,
          ncm TEXT NOT NULL,
          descricao TEXT,
          PRIMARY KEY (cest, ncm)
        );
    """)
    # Índices para performance
    conn.execute('CREATE INDEX IF NOT EXISTS idx_ncm_ibpt ON ibpt_taxes (ncm);')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_ncm_cest ON cest_data (ncm);')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_cest_data_cest ON cest_data (cest);')
    
    # Criação do FTS5 (Tabela Virtual de Busca) e Triggers - Recriando a lógica do Node.js
    try:
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS ibpt_search USING fts5(
              ncm, 
              descricao, 
              content='ibpt_taxes', 
              content_rowid='rowid',
              tokenize = 'porter unicode61'
            );
        """)
        
        # Gatilhos (Triggers) para manter a tabela de busca sincronizada.
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS ibpt_taxes_after_insert AFTER INSERT ON ibpt_taxes BEGIN
              INSERT INTO ibpt_search(rowid, ncm, descricao) VALUES (new.rowid, new.ncm, new.descricao);
            END;
        """)
        # A trigger de DELETE precisa de sintaxe diferente para sqlite3
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS ibpt_taxes_after_delete AFTER DELETE ON ibpt_taxes BEGIN
              DELETE FROM ibpt_search WHERE rowid = old.rowid;
            END;
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS ibpt_taxes_after_update AFTER UPDATE ON ibpt_taxes BEGIN
              DELETE FROM ibpt_search WHERE rowid = old.rowid;
              INSERT INTO ibpt_search(rowid, ncm, descricao) VALUES (new.rowid, new.ncm, new.descricao);
            END;
        """)
        print('Tabelas, índices primários e FTS5 (busca) prontos.')
        
    except sqlite3.OperationalError as e:
        # FTS5 pode não estar disponível em algumas instalações SQLite. A busca do FastAPI é o fallback.
        print(f"WARN: FTS5 não pôde ser criado: {e}. A busca usará o modelo semântico/similaridade do FastAPI.")

    conn.commit()


# --- LÓGICA DE ATUALIZAÇÃO ---

def processar_ibpt(session: requests.Session, conn: sqlite3.Connection):
    """Baixa e processa os arquivos IBPTax do repositório."""
    print('Iniciando processamento dos dados do IBPT...')
    
    conn.execute('DELETE FROM ibpt_taxes;')

    insert_sql = """
        INSERT OR REPLACE INTO ibpt_taxes (ncm, uf, ex, tipo, descricao, aliqNacional, aliqEstadual, aliqMunicipal, aliqImportado, vigenciaInicio, vigenciaFim, chave, versao, fonte)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    try:
        response = session.get(IBPT_SVN_URL)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        csv_files = [link.get('href') for link in soup.find_all('a', href=re.compile(r'\.csv$'))]

        for file in csv_files:
            match = re.search(r'TabelaIBPTax([A-Z]{2})', file)
            if not match:
                print(f"WARN: Não foi possível extrair a UF do arquivo: {file}. Pulando.")
                continue
            uf_do_arquivo = match.group(1)
            
            print(f"Processando IBPT para a UF: {uf_do_arquivo} (arquivo: {file})...")
            
            file_url = IBPT_SVN_URL + file
            csv_response = session.get(file_url, stream=True)
            csv_response.raise_for_status()
            
            # Decodifica como latin-1 ou ISO-8859-1 (comum em arquivos brasileiros)
            csv_content = csv_response.content.decode('iso-8859-1') 
            
            df_csv = pd.read_csv(
                StringIO(csv_content), 
                sep=';', 
                header=None, 
                skiprows=1,
                encoding='iso-8859-1',
                dtype=str
            )
            
            df_csv.columns = ['codigo', 'ex', 'tipo', 'descricao', 'nacionalfederal', 
                              'importadosfederal', 'estadual', 'municipal', 'vigenciainicio', 
                              'vigenciafim', 'chave', 'versao', 'fonte']

            data_to_insert = []
            for _, row in df_csv.iterrows():
                ncm_limpo = str(row['codigo'] or '').replace('.', '')
                
                def clean_float(val):
                    return float(str(val or '0').replace(',', '.'))
                    
                data_to_insert.append((
                    ncm_limpo,
                    uf_do_arquivo,
                    row['ex'],
                    row['tipo'],
                    row['descricao'],
                    clean_float(row['nacionalfederal']),
                    clean_float(row['estadual']),
                    clean_float(row['municipal']),
                    clean_float(row['importadosfederal']),
                    row['vigenciainicio'],
                    row['vigenciafim'],
                    row['chave'],
                    row['versao'],
                    row['fonte']
                ))

            conn.executemany(insert_sql, data_to_insert)
            conn.commit()
            print(f"{len(data_to_insert)} linhas inseridas para {uf_do_arquivo}.")
        
        # Reconstrói o FTS5 (se existir) após a importação em massa
        try:
            print('Reconstruindo índice de busca semântica FTS5...')
            conn.execute("INSERT INTO ibpt_search(ibpt_search) VALUES('rebuild');")
            conn.commit()
            print('Índice de busca FTS5 reconstruído com sucesso.')
        except sqlite3.OperationalError:
            pass # Ignora se a tabela FTS5 não foi criada

        print('Processamento do IBPT concluído.')

    except requests.RequestException as e:
        print(f"Erro de rede ao processar IBPT: {e}")
    except Exception as e:
        print(f"Erro inesperado ao processar IBPT: {e}")


def processar_cest(session: requests.Session, conn: sqlite3.Connection):
    """Baixa e processa os dados da tabela CEST do site do Confaz."""
    print('Iniciando verificação de atualização do CEST...')
    
    metadados_locais = {}
    if os.path.exists(METADATA_FILE_PATH):
        try:
            with open(METADATA_FILE_PATH, 'r', encoding='utf-8') as f:
                metadados_locais = json.load(f)
        except json.JSONDecodeError:
            print("WARN: Arquivo de metadados CEST corrompido.")

    try:
        head_response = session.head(CEST_URL, verify=False, timeout=10)
        etag_remoto = head_response.headers.get('etag')
        last_modified_remoto = head_response.headers.get('last-modified')

        if (etag_remoto and etag_remoto == metadados_locais.get('etag')) or \
           (last_modified_remoto and last_modified_remoto == metadados_locais.get('lastModified')):
            print('Tabela CEST não foi modificada. Pulando atualização.')
            return

        print('Nova versão da tabela CEST encontrada. Iniciando processamento completo...')

        response = session.get(CEST_URL, verify=False, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        todos_os_itens = []

        for element in soup.find_all('p', class_='A6-1Subtitulo'):
            titulo_anexo = element.get_text().strip()
            if titulo_anexo.startswith('ANEXO ') and len(titulo_anexo) < 15:
                tabela = element.find_next_sibling('table')
                if tabela:
                    for i, linha in enumerate(tabela.find_all('tr')):
                        if i == 0: continue
                        
                        celulas = linha.find_all('td')
                        if len(celulas) >= 4:
                            ncm_string = celulas[2].get_text().strip()
                            ncm_array = re.split(r'\s+', ncm_string)
                            ncm_array = [ncm for ncm in ncm_array if len(ncm) > 0]
                            
                            if ncm_array:
                                todos_os_itens.append({
                                    'CEST': celulas[1].get_text().strip(),
                                    'NCM_SH': ncm_array,
                                    'Descricao': re.sub(r'\s\s+', ' ', celulas[3].get_text().strip())
                                })

        if not todos_os_itens:
            print("WARN: Não foram encontrados itens CEST na página.")
            return

        conn.execute('DELETE FROM cest_data;')
        insert_cest = 'INSERT OR IGNORE INTO cest_data (cest, ncm, descricao) VALUES (?, ?, ?)'
        data_to_insert = []
        
        for item in todos_os_itens:
            cest_limpo = str(item['CEST'] or '').replace('.', '')
            for ncm in item['NCM_SH']:
                ncm_limpo = re.sub(r'[^\d]', '', ncm)
                if ncm_limpo:
                    data_to_insert.append((cest_limpo, ncm_limpo, item['Descricao']))

        conn.executemany(insert_cest, data_to_insert)
        conn.commit()
        print(f"{len(data_to_insert)} combinações CEST-NCM inseridas.")
        
        novos_metadados = {
            'etag': etag_remoto,
            'lastModified': last_modified_remoto,
            'lastUpdate': pd.Timestamp.now().isoformat()
        }
        
        data_dir = os.path.dirname(METADATA_FILE_PATH)
        if not os.path.exists(data_dir):
             os.makedirs(data_dir)
             
        with open(METADATA_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(novos_metadados, f, ensure_ascii=False, indent=2)
            
        print('Metadados de controle do CEST foram atualizados.')

    except requests.RequestException as e:
        print(f"Erro de rede ao processar CEST: {e}")
    except Exception as e:
        print(f"Erro inesperado ao processar CEST: {e}")


def atualizar_banco_dados():
    """Função principal para executar a atualização completa."""
    conn = None
    try:
        session = get_session()
        conn = get_db_connection(DB_PATH)
        criar_tabelas(conn)
        
        # Sequência de atualização
        processar_ibpt(session, conn)
        processar_cest(session, conn)
        
    except Exception as e:
        print(f"ERRO CRÍTICO no processo de atualização: {e}")
        # Se falhar, é crucial garantir que a conexão seja fechada
    finally:
        if conn:
            conn.close()
            print("Conexão com o banco de dados fechada após atualização.")

# --- EXPORTAÇÃO (OPCIONAL) ---
# Se for necessário, descomente e ajuste o processo de exportação abaixo:
#
# def exportar_arquivos(conn: sqlite3.Connection):
#     print('\nIniciando exportação para JSON e CSV...')
#     # ... (Lógica de exportação do Node.js transformada para Python/Pandas) ...
#     pass
#
# def main_export():
#     conn = get_db_connection(DB_PATH)
#     exportar_arquivos(conn)
#     conn.close()


if __name__ == '__main__':
    atualizar_banco_dados()