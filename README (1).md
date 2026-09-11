# Agente de Análise de Sentimento (LangGraph + Groq)

Agente que classifica o sentimento de comentários em **POSITIVO**, **NEGATIVO** ou **NEUTRO**, construído com [LangGraph](https://langchain-ai.github.io/langgraph/) e usando a API da [Groq](https://groq.com/) para inferência.

## Por que um grafo, e não só uma chamada de LLM?

Uma chamada direta ao LLM funciona, mas não trata o caso em que o modelo responde fora do formato esperado (ex: "o comentário parece positivo" em vez de "POSITIVO"). Este projeto usa um grafo de estados com:

- **Nó de classificação**: envia o comentário ao LLM e recebe a resposta.
- **Validação condicional**: verifica se a resposta está entre os rótulos válidos.
  - Se válida → encerra.
  - Se inválida e ainda há tentativas → tenta classificar de novo.
  - Se esgotar as tentativas → marca como `INDEFINIDO` em vez de propagar um erro silencioso.

Essa estrutura é pequena, mas ilustra o padrão que sustenta agentes mais complexos: estado explícito, transições condicionais e tratamento de falha.

## Como rodar localmente

1. Clone o repositório e entre na pasta:
   ```bash
   git clone <url-do-seu-repo>
   cd sentiment-agent
   ```

2. Crie um ambiente virtual (opcional, mas recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

4. Copie o arquivo de exemplo de variáveis de ambiente e preencha com sua chave da Groq (gratuita em [console.groq.com/keys](https://console.groq.com/keys)):
   ```bash
   cp .env.example .env
   ```
   Edite `.env` e coloque sua chave:
   ```
   GROQ_API_KEY=sua_chave_aqui
   ```

5. Rode o agente:
   ```bash
   python sentiment_agent.py
   ```

## Interface web (Gradio)

Além do uso via terminal, o projeto inclui uma interface web simples com Gradio:

```bash
python app_gradio.py
```

Isso abre uma página local (por padrão em `http://127.0.0.1:7860`) onde é possível colar um comentário e ver o sentimento classificado. Para gerar um link público temporário (útil para compartilhar como demo de portfólio), altere a última linha do arquivo para:

```python
demo.launch(share=True)
```

No Google Colab, `share=True` é obrigatório, pois não há como acessar `127.0.0.1` do notebook diretamente.

## Como rodar no Google Colab

```python
!pip install -q langchain-groq langchain-core langgraph python-dotenv

import os
os.environ["GROQ_API_KEY"] = "sua_chave_aqui"  # ou use os Secrets do Colab

# cole o conteúdo de sentiment_agent.py aqui, ou importe o arquivo
```

> **Nunca cole sua chave de API diretamente em um notebook que será compartilhado ou versionado.** No Colab, prefira usar o gerenciador de Secrets (ícone de chave na barra lateral).

## Estrutura

```
sentiment-agent/
├── sentiment_agent.py   # código do agente (grafo LangGraph)
├── app_gradio.py        # interface web (Gradio)
├── requirements.txt     # dependências
├── .env.example         # modelo de variáveis de ambiente (sem chaves reais)
├── .gitignore            # garante que .env não seja versionado
└── README.md
```

## Possíveis extensões

- Adicionar um nó de sugestão de resposta automática quando o sentimento for NEGATIVO.
- Persistir resultados em um banco (ex: Supabase) para análise histórica.
- Expor via API (FastAPI) ou interface web (Gradio/Streamlit) para uso por terceiros.
- Trocar o provedor de LLM (Groq, OpenAI, Claude, Ollama local) via variável de ambiente.

## Autor

Heber Conrado
