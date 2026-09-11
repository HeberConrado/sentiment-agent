import streamlit as st
from typing import TypedDict, Literal

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

st.set_page_config(page_title="Análise de Sentimento", page_icon="🎭")


@st.cache_resource
def carregar_grafo():
    groq_api_key = st.secrets["GROQ_API_KEY"]
    llm = ChatGroq(model="openai/gpt-oss-20b", api_key=groq_api_key, temperature=0)

    class SentimentState(TypedDict):
        comentario: str
        sentimento: str
        tentativas: int

    VALIDOS = {"POSITIVO", "NEGATIVO", "NEUTRO"}

    def classificar(state: SentimentState) -> SentimentState:
        prompt = (
            "Classifique o sentimento do comentário abaixo em UMA ÚNICA PALAVRA: "
            "POSITIVO, NEGATIVO ou NEUTRO. Responda apenas a palavra.\n\n"
            f"Comentário: {state['comentario']}"
        )
        resposta = llm.invoke([HumanMessage(content=prompt)])
        sentimento = resposta.content.strip().upper()
        return {**state, "sentimento": sentimento, "tentativas": state.get("tentativas", 0) + 1}

    def validar(state: SentimentState) -> Literal["ok", "retry", "falhou"]:
        if state["sentimento"] in VALIDOS:
            return "ok"
        if state["tentativas"] >= 2:
            return "falhou"
        return "retry"

    def marcar_falha(state: SentimentState) -> SentimentState:
        return {**state, "sentimento": "INDEFINIDO"}

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


app_grafo = carregar_grafo()

st.title("🎭 Agente de Análise de Sentimento")
st.markdown(
    "Classifica comentários como **POSITIVO**, **NEGATIVO** ou **NEUTRO**, "
    "usando um agente construído com LangGraph e a API da Groq."
)

comentario = st.text_area(
    "Comentário",
    placeholder="Cole aqui um comentário, review ou mensagem...",
    height=120,
)

exemplos = [
    "Adorei o atendimento, super rápido e educado!",
    "Produto chegou quebrado e ninguém responde meu chamado.",
    "Comprei ontem, ainda não testei.",
]
st.caption("Exemplos: " + " | ".join(exemplos))

if st.button("Analisar sentimento", type="primary"):
    if not comentario.strip():
        st.warning("Digite um comentário para analisar.")
    else:
        with st.spinner("Analisando..."):
            try:
                resultado = app_grafo.invoke({"comentario": comentario, "tentativas": 0})
                sentimento = resultado["sentimento"]
                emoji = {"POSITIVO": "🟢", "NEGATIVO": "🔴", "NEUTRO": "🟡"}.get(sentimento, "⚪")
                st.success(f"{emoji} **{sentimento}**")
            except Exception as e:
                st.error(f"Erro ao consultar o modelo: {e}")
