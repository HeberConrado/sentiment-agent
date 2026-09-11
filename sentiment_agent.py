import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal
from dotenv import load_dotenv

# Carrega variáveis do arquivo .env (não versionado)
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY não encontrada. Crie um arquivo .env com "
        "GROQ_API_KEY=sua_chave (veja .env.example)."
    )

llm = ChatGroq(model="openai/gpt-oss-20b", api_key=GROQ_API_KEY, temperature=0)


class SentimentState(TypedDict):
    comentario: str
    sentimento: str
    tentativas: int


VALIDOS = {"POSITIVO", "NEGATIVO", "NEUTRO"}


def classificar(state: SentimentState) -> SentimentState:
    """Chama o LLM para classificar o sentimento do comentário."""
    prompt = (
        "Classifique o sentimento do comentário abaixo em UMA ÚNICA PALAVRA: "
        "POSITIVO, NEGATIVO ou NEUTRO. Responda apenas a palavra.\n\n"
        f"Comentário: {state['comentario']}"
    )
    resposta = llm.invoke([HumanMessage(content=prompt)])
    sentimento = resposta.content.strip().upper()
    return {**state, "sentimento": sentimento, "tentativas": state.get("tentativas", 0) + 1}


def validar(state: SentimentState) -> Literal["ok", "retry", "falhou"]:
    """Decide se a resposta do LLM é válida, deve ser tentada de novo, ou falhou."""
    if state["sentimento"] in VALIDOS:
        return "ok"
    if state["tentativas"] >= 2:
        return "falhou"
    return "retry"


def marcar_falha(state: SentimentState) -> SentimentState:
    """Marca o resultado como indefinido após esgotar as tentativas."""
    return {**state, "sentimento": "INDEFINIDO"}


def construir_grafo():
    """Monta e compila o grafo LangGraph do agente de análise de sentimento."""
    grafo = StateGraph(SentimentState)
    grafo.add_node("classificar", classificar)
    grafo.add_node("falha", marcar_falha)
    grafo.set_entry_point("classificar")
    grafo.add_conditional_edges(
        "classificar",
        validar,
        {"ok": END, "retry": "classificar", "falhou": "falha"},
    )
    grafo.add_edge("falha", END)
    return grafo.compile()


app = construir_grafo()


def analisar_sentimento(comentario: str) -> str:
    """Ponto de entrada simples: recebe um texto e retorna o sentimento."""
    resultado = app.invoke({"comentario": comentario, "tentativas": 0})
    return resultado["sentimento"]


if __name__ == "__main__":
    print("Agente de Análise de Sentimento (LangGraph + Groq)")
    print("Digite 'sair' para encerrar.\n")
    while True:
        comentario = input("Comentário: ")
        if comentario.lower() == "sair":
            break
        print(f"Sentimento: {analisar_sentimento(comentario)}\n")
