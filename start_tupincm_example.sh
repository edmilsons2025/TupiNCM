#!/bin/bash

# Caminho para o diretório do projeto
PROJECT_DIR="/home/SEU_USUARIO/projetos/TupiNCM/"

# 1. Navega para o diretório do projeto (para que caminhos relativos funcionem, incluindo o 'venv/')
cd $PROJECT_DIR

# 2. Ativa o ambiente virtual (configura PATH e variáveis de ambiente)
# Adapte o nome da pasta (venv, .venv, etc.) se necessário
source venv/bin/activate

# 3. Executa o uvicorn (o uvicorn estará agora no PATH da sessão shell)
# Remova o --reload (apenas para desenvolvimento)
uvicorn main:app --host 0.0.0.0 --port 8200
