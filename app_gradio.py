import os
from dotenv import load_dotenv
import gradio as gr

from sentiment_agent import analisar_sentimento

load_dotenv()

EXEMPLOS = [
    "Adorei o atendimento, super rápido e educado!",
    "Produto chegou quebrado e ninguém responde meu chamado.",
    "Comprei ontem, ainda não testei.",
]


def classificar_ui(comentario: str):
    if not comentario or not comentario.strip():
        return "Digite um comentário para analisar."
    try:
        sentimento = analisar_sentimento(comentario)
    except Exception as e:
        return f"Erro ao consultar o modelo: {e}"

    emoji = {"POSITIVO": "🟢", "NEGATIVO": "🔴", "NEUTRO": "🟡"}.get(sentimento, "⚪")
    return f"{emoji} {sentimento}"


with gr.Blocks(title="Análise de Sentimento") as demo:
    gr.Markdown(
        "# Agente de Análise de Sentimento\n"
        "Classifica comentários como **POSITIVO**, **NEGATIVO** ou **NEUTRO**, "
        "usando um agente construído com LangGraph e a API da Groq."
    )
    with gr.Row():
        entrada = gr.Textbox(
            label="Comentário",
            placeholder="Cole aqui um comentário, review ou mensagem...",
            lines=4,
        )
    botao = gr.Button("Analisar sentimento", variant="primary")
    saida = gr.Textbox(label="Resultado", interactive=False)

    botao.click(fn=classificar_ui, inputs=entrada, outputs=saida)
    entrada.submit(fn=classificar_ui, inputs=entrada, outputs=saida)

    gr.Examples(examples=EXEMPLOS, inputs=entrada)

if __name__ == "__main__":
    demo.launch()
