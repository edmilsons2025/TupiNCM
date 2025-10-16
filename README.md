# Serviço de Busca Inteligente de NCM

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.100%2B-green?style=for-the-badge&logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/scikit--learn-ML-orange?style=for-the-badge&logo=scikit-learn" alt="scikit-learn">
</p>

## Descrição

API RESTful construída com Python e FastAPI que utiliza um modelo híbrido de Inteligência Artificial para buscas semânticas de NCMs (Nomenclatura Comum do Mercosul), oferecendo alta precisão e flexibilidade.

---

## 🧠 Arquitetura e Conceito

Este serviço não realiza buscas por palavra-chave simples. Ele utiliza um **Modelo Híbrido de IA** que combina:

- **Conhecimento Especialista** (Mapa Semântico)
- **Aprendizado de Máquina Estatístico** (Similaridade de Cosseno)

### Nível 1: Mapa Semântico (Busca por Conhecimento)

O arquivo `mapa_semantico.json` funciona como um "glossário especializado" do sistema. Quando uma busca é realizada, a API tenta encontrar uma correspondência neste mapa.

**Exemplo:**  
Se a busca contém termos como "placa mãe" ou "pcb", o sistema identifica o conceito "placa de circuito impresso" e retorna instantaneamente os NCMs de alta confiança definidos no arquivo.

**Vantagem:**  
Resultados rápidos e precisos para termos conhecidos.

### Nível 2: Similaridade de Cosseno (Busca por IA Estatística)

Se não houver correspondência no Mapa Semântico, o sistema utiliza um modelo de Machine Learning.

**Como funciona:**  
Utiliza TF-IDF para analisar a importância das palavras na busca em relação às descrições de NCM no banco de dados. A Similaridade de Cosseno calcula os NCMs semanticamente mais próximos da intenção da busca.

**Vantagem:**  
Encontra resultados relevantes mesmo para termos novos ou inesperados.

---

## ✨ Features

- **Busca Semântica Híbrida:** Combina conhecimento especialista e Machine Learning.
- **Alta Precisão:** Resultados instantâneos para termos definidos no Mapa Semântico.
- **Processamento de Linguagem Natural:** Stemming e remoção de stopwords para português.
- **API Interativa:** Documentação automática via Swagger UI.
- **Extensível:** Basta adicionar novos conceitos ao `mapa_semantico.json`.

---

## 🛠️ Tecnologias Utilizadas

- **Backend:** Python 3.10+
- **Framework:** FastAPI
- **Servidor ASGI:** Uvicorn
- **Machine Learning:** Scikit-learn
- **Manipulação de Dados:** Pandas
- **Processamento de Linguagem:** NLTK

---

## 📂 Estrutura do Projeto

```
ncm-service-backend/
├── data/
│   ├── cest_metadata.json            # Auxiliar para verificação de versão do banco de dados
│   ├── ibpt.db                       # Banco de dados dos NCMs
│   └── mapa_semantico_example.json   # Glossário especializado do sistema
├── data_updater.py                   # Cria, Verifica e Atualiza o Banco de Dados
├── main.py                           # Código principal da API FastAPI
├── requirements.txt                  # Dependências do Python
└── start_tupincm_example.sh          # Script de exemplo para iniciar a API por systemd
```

---

## ⚙️ Configuração e Instalação

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/edmilsons2025/TupiNCM.git
   cd ncm-service-backend
   ```

2. **Crie e ative um ambiente virtual:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   # No Windows: venv\Scripts\activate
   ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Downloads do NLTK:**  
   Na primeira execução, o servidor baixa automaticamente os pacotes `stopwords` e `punkt` do NLTK.

5. **Banco de Dados:**  
   Certifique-se de que o arquivo `openfiscal.db` está na pasta `data/`.

---

## 🚀 Executando o Serviço

Com o ambiente virtual ativado, execute:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

O servidor estará disponível em [http://127.0.0.1:8000](http://127.0.0.1:8000).

---

## 🎮 Como Usar e Testar

Acesse a documentação automática:

- [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

Teste o endpoint `GET /api/ncm-search`:

- Clique em "Try it out"
- Digite uma descrição de produto (ex: `bateria de lítio`, `fio da placa de led`)
- Clique em "Execute"

**Exemplo com cURL:**
```bash
curl -X GET "http://127.0.0.1:8000/api/ncm-search?description=placa%20mãe"
```

---

## 🔮 Melhorias Futuras

- **Frontend:** Interface dedicada para consumir a API.
- **Modelos de Embeddings:** Migrar para modelos avançados (Word2Vec, GloVe, BERT).
- **Containerização:** Dockerfile para facilitar o deploy.
- **Autenticação:** Adicionar segurança com chaves de API.

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](../LICENSE) para mais detalhes.