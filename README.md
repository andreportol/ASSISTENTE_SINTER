# SINTER — Assistente de Documentos

Aplicação Django com LangChain para responder perguntas sobre documentos locais (RAG) e consultar um banco Postgres via agente SQL. Usa FAISS para indexação, Armazenamento local em `documents/` e OpenAI para LLM/embeddings.

## Requisitos
- Python 3.10+ (recomendado 3.11).
- pip / venv.
- Chave da API OpenAI (necessária para responder perguntas).
- Acesso opcional a um Postgres se quiser usar o agente SQL; por padrão funciona com SQLite.

## Passo a passo
1) **Clonar e entrar na pasta**
```bash
git clone <sua-url-do-repo>
cd SINTER
```

2) **Criar ambiente virtual**
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

3) **Instalar dependências**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4) **Configurar variáveis de ambiente (.env na raiz)**
Use este modelo e preencha os valores:
```env
# OpenAI
OPENAI_API_KEY=sua_chave_aqui
OPENAI_MODEL=gpt-4o-mini

# Banco padrão (SQLite)
DB_ENGINE=django.db.backends.sqlite3
DB_NAME=db.sqlite3

# Postgres (opcional, necessário para o agente SQL)
# DB_ENGINE=django.db.backends.postgresql
# DB_NAME=sinter
# DB_USER=postgres
# DB_PASSWORD=sua_senha
# DB_HOST=localhost
# DB_PORT=5432

# RAG / documentos
# Caminho direto para o arquivo de regras (PDF/TXT/MD)
CORE_RULES_PATH=documents/ManualOperacional_v1.12.pdf
# Pasta onde ficam os documentos e o índice FAISS
CORE_RULES_DIR=documents
RAG_INDEX_PATH=documents/faiss_index
```

5) **Preparar documentos**
- Coloque seus PDFs/TXT/MD em `documents/`. O sistema escolhe o primeiro arquivo válido da pasta se `CORE_RULES_PATH` não estiver definido.
- O índice FAISS é salvo em `documents/faiss_index`. Se quiser reconstruir, apague essa pasta e rode novamente.

6) **Criar/migrar o banco**
```bash
python manage.py migrate
```
- Com SQLite nada adicional é necessário.
- Para Postgres, crie o banco antes e ajuste as variáveis DB_* no `.env`.

7) **Rodar o servidor**
```bash
python manage.py runserver
```
Abra http://127.0.0.1:8000 e faça perguntas. A barra lateral mostra as fontes carregadas; o histórico fica na coluna principal.

## Uso dos agentes
- **RAG (documento de regras):** usa o arquivo indicado em `CORE_RULES_PATH` ou o primeiro válido em `CORE_RULES_DIR`.
- **SQL (Postgres):** requer DB_* configurados com engine Postgres; se o engine não for Postgres, o agente SQL é desabilitado.

## Dicas e solução de problemas
- Erro de chave OpenAI: verifique `OPENAI_API_KEY`.
- Índice FAISS corrompido: apague `documents/faiss_index` e recarregue a página para rebuild automático.
- Problemas com Postgres: confira engine/host/porta/usuário/senha e se o banco existe antes do `migrate`.

## Estrutura relevante
- `manage.py` — comandos Django.
- `SINTER/settings.py` — configurações e leitura do `.env`.
- `app/core/` — agentes LangChain, views e templates.
- `documents/` — documentos de regras e índice FAISS gerado.
