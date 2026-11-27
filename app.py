"""Match candidato × vaga · Decision (Datathon FIAP) — app em Gradio (Hugging Face Spaces).

Duas perguntas, avaliadas no histórico da Decision:
  1. Busca: o texto do currículo ajuda a encontrar, na base, os candidatos que os hunters escolheram?
  2. Decisão: entre os candidatos prospectados, ele indica quem será contratado?
Nenhum dado de candidato é exibido; o vocabulário do modelo publicado vem só do texto das vagas.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
try:  # repositório do GitHub (src/match)
    from match.features import termos_em_comum
except ImportError:  # Space do Hugging Face (arquivos na raiz)
    from features import termos_em_comum  # type: ignore


def arquivo(*partes: str) -> Path:
    """Procura o artefato no layout do repositório e, se não existir, na raiz (layout do Space)."""
    p = ROOT.joinpath(*partes)
    return p if p.exists() else ROOT / partes[-1]


C1, C2, INK, MUTED, GRID = "#3987e5", "#d95926", "#cbd5e1", "#94a3b8", "#1a2a42"
FONT = "Archivo, system-ui, sans-serif"
M = json.loads(arquivo("reports", "metrics.json").read_text(encoding="utf-8"))
F = json.loads(arquivo("reports", "funil.json").read_text(encoding="utf-8"))
TF = joblib.load(arquivo("models", "tfidf_vagas.joblib"))
R, B = M["rankers"], {k: v for k, v in M["busca"].items() if not k.startswith("_")}
TOP = M["busca"]["_config"]["top"]
BASE_N = M["busca"]["_config"]["base_amostrada"]
Q = M["quantis_tfidf_app"]
ALE, TFR = R["Aleatório (esperado)"], R["TF-IDF"]
BT, BA = B["TF-IDF"], B["Aleatório (esperado)"]


def pct(x: float, nd: int = 1) -> str:
    return f"{x * 100:.{nd}f}%".replace(".", ",")


def num(x: float, nd: int = 0) -> str:
    return f"{x:,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def kpis(items) -> str:
    return '<div class="kpis">' + "".join(
        f'<div class="kpi"><div class="t">{t}</div><div class="v">{v}</div><div class="l">{l}</div></div>'
        for t, v, l in items) + "</div>"


def style(fig, h=380, legend=True):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family=FONT, color=INK, size=13), showlegend=legend,
                      legend=dict(orientation="h", y=1.1, x=0), bargap=0.3,
                      hoverlabel=dict(bgcolor="#0b1626", bordercolor=GRID, font=dict(family=FONT, color="#f1f5f9")))
    fig.update_xaxes(showgrid=False, linecolor=GRID, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickfont=dict(color=MUTED))
    return fig


def fig_auc(dados: dict, titulo_x: str):
    nomes = list(dados)
    vals = [dados[n]["auc_media"] for n in nomes]
    lo = [v - dados[n].get("auc_ic95", [v, v])[0] for n, v in zip(nomes, vals)]
    hi = [dados[n].get("auc_ic95", [v, v])[1] - v for n, v in zip(nomes, vals)]
    fig = go.Figure(go.Bar(
        y=nomes, x=vals, orientation="h", marker=dict(color=[MUTED if n.startswith("Aleat") else C1 for n in nomes], cornerradius=4),
        error_x=dict(type="data", symmetric=False, array=hi, arrayminus=lo, color=INK, thickness=1.2),
        text=[num(v, 2) for v in vals], textposition="inside", insidetextanchor="start", textfont=dict(color="#03070d"),
        hovertemplate="%{y}<br>AUC %{x:.3f}<extra></extra>"))
    fig.add_vline(x=0.5, line=dict(color=MUTED, dash="dot"))
    fig.update_xaxes(range=[0.3, 1.0], showgrid=True, gridcolor=GRID, title=titulo_x)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return style(fig, 60 + 52 * len(nomes), False)


def fig_funil():
    a = F["funil_alcancou"]
    fig = go.Figure(go.Funnel(y=[e.split(". ")[1] for e in a], x=list(a.values()), marker=dict(color=C1),
                              textinfo="value+percent initial", textfont=dict(color="#f1f5f9")))
    return style(fig, 360, False)


def fig_contrato():
    d = F["por_contratacao"][:8]
    fig = go.Figure(go.Bar(y=[r["tipo_contratacao"] for r in d], x=[r["taxa_contratacao"] for r in d], orientation="h",
                           marker=dict(color=C1, cornerradius=4), text=[pct(r["taxa_contratacao"]) for r in d],
                           textposition="outside", textfont=dict(color=INK), customdata=[r["prospects"] for r in d],
                           hovertemplate="%{y}<br>%{x:.1%} contratados<br>%{customdata} prospecções<extra></extra>"))
    fig.update_xaxes(tickformat=".0%", range=[0, max(r["taxa_contratacao"] for r in d) * 1.35])
    fig.update_yaxes(autorange="reversed")
    return style(fig, 360, False)


def avaliar(vaga: str, cv: str) -> str:
    if not vaga.strip() or not cv.strip():
        return "<p class='muted'>Preencha a vaga e o currículo.</p>"
    v, c = TF.transform([vaga]), TF.transform([cv])
    sim = float((v @ c.T).toarray()[0, 0])
    niveis = np.linspace(0, 1, len(Q["base"]))
    acima_base = float(np.interp(sim, Q["base"], niveis))
    acima_pros = float(np.interp(sim, Q["prospectados"], niveis))
    posicao = lambda p: "top 1%" if p >= 0.99 else f"acima de {pct(p, 0)}"  # noqa: E731
    chips = "".join(f'<span class="chip">{w}</span>' for w in termos_em_comum(vaga, cv)) or "<span class='muted'>nenhum</span>"
    return kpis([
        ("Similaridade TF-IDF", num(sim, 3), "cosseno entre vaga e currículo no vocabulário das vagas"),
        ("Frente à base", posicao(acima_base), "dos pares vaga × candidato aleatório da base"),
        ("Frente aos prospectados", posicao(acima_pros), "dos pares vaga × candidato que os hunters escolheram"),
    ]) + f'<p style="margin-top:14px"><b>Termos em comum</b><br>{chips}</p>'


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700&family=Instrument+Serif:ital@0;1&family=JetBrains+Mono:wght@400;500&display=swap');
:root, .dark { --body-background-fill: #03070d; --background-fill-primary: #03070d; --background-fill-secondary: #07111f;
  --block-background-fill: #07111f; --border-color-primary: #1a2a42; --body-text-color: #cbd5e1; --color-accent: #3987e5;
  --button-primary-background-fill: #2563eb; --button-primary-background-fill-hover: #1d4ed8; --input-background-fill: #0b1626; }
body, .gradio-container { background: #03070d !important; font-family: 'Archivo', system-ui, sans-serif !important; }
.gradio-container { max-width: 1180px !important; }
.eyebrow { font-family: 'JetBrains Mono', monospace; font-size: .78rem; letter-spacing: .14em; text-transform: uppercase; color: #a78bfa; }
.hero { font-size: clamp(2rem, 4.2vw, 3.1rem); line-height: 1.05; font-weight: 600; color: #f1f5f9; margin: .4rem 0 .8rem; }
.hero em { font-family: 'Instrument Serif', Georgia, serif; font-style: italic; font-weight: 400; color: #60a5fa; }
.lead { font-size: 1.06rem; color: #cbd5e1; max-width: 68ch; }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin: 10px 0; }
.kpi { border: 1px solid #1a2a42; border-radius: 14px; padding: 1rem 1.15rem; background: #07111f; }
.kpi .v { font-size: 1.8rem; font-weight: 600; color: #f1f5f9; line-height: 1.1; }
.kpi .l { color: #94a3b8; font-size: .88rem; margin-top: .3rem; }
.kpi .t { font-family: 'JetBrains Mono', monospace; font-size: .72rem; color: #60a5fa; text-transform: uppercase; letter-spacing: .1em; }
.note { border-left: 3px solid #8b5cf6; padding: .7rem 1rem; background: #0b1626; border-radius: 0 10px 10px 0; color: #cbd5e1; }
.muted { color: #94a3b8; font-size: .9rem; }
.chip { display: inline-block; border: 1px solid #1a2a42; background: #0c1d36; color: #c7d7fe; border-radius: 999px;
        padding: .15rem .6rem; margin: .15rem; font-family: 'JetBrains Mono', monospace; font-size: .8rem; }
footer { display: none !important; }
"""

EX_VAGA = ("Analista de Dados Pleno. Construir pipelines de dados em Python e SQL, modelar dados no Databricks e "
           "publicar dashboards em Power BI. Desejável experiência com PySpark, Git e metodologias ágeis. Inglês intermediário.")
EX_CV = ("Analista de dados com 4 anos de experiência. Python (pandas, PySpark), SQL Server e Databricks. "
         "Criação de dashboards em Power BI com DAX, versionamento com Git e trabalho em times Scrum. Inglês avançado. "
         "Graduação em Sistemas de Informação.")

av = M["avaliacao"]
with gr.Blocks(title="Match candidato × vaga · Decision") as demo:
    gr.HTML(
        '<div class="eyebrow">Decision · Datathon FIAP · NLP e Machine Learning</div>'
        '<h1 class="hero">O currículo ajuda a <em>encontrar</em> o candidato.<br>Escolher quem será contratado é outra história.</h1>'
        f'<p class="lead">No histórico de {num(av["vagas"])} vagas da Decision, testei duas coisas. Primeiro, se a similaridade entre '
        f'vaga e currículo encontra, numa amostra de {num(BASE_N)} candidatos da base, aqueles que os hunters escolheram. Depois, se ela '
        f'indica, entre os escolhidos, quem foi contratado. A resposta é sim para a primeira pergunta e quase não para a segunda.</p>'
        + kpis([
            ("Busca · AUC", num(BT["auc_media"], 2), "o prospectado fica acima de um candidato aleatório da base (0,50 = acaso)"),
            (f"Busca · top {TOP}", pct(BT[f"recall@{TOP}"], 0), f"dos prospectados estão no top 1% da lista, contra {pct(BA[f'recall@{TOP}'], 0)} ao acaso"),
            ("Decisão · AUC", num(TFR["auc_media"], 2), f"entre os prospectados (IC 95%: {num(TFR['auc_ic95'][0], 2)} a {num(TFR['auc_ic95'][1], 2)})"),
            ("Funil", num(F["prospects_por_contratacao"], 1), "prospecções por contratação na base"),
        ])
    )
    with gr.Tabs():
        with gr.Tab("1 · Busca"):
            gr.Plot(fig_auc(B, "AUC: prospectado × candidato aleatório da base"), show_label=False)
            gr.HTML(f'<div class="note"><b>Leitura.</b> O TF-IDF reproduz bem a escolha dos hunters: {pct(BT[f"recall@{TOP}"], 0)} dos candidatos '
                    f'prospectados aparecem no top {TOP} de uma lista com {num(BASE_N)} candidatos da base, {num(BT[f"recall@{TOP}"] / BA[f"recall@{TOP}"], 0)} vezes '
                    f'o acaso. Os embeddings multilíngues também funcionam, mas abaixo do TF-IDF: nas vagas de TI, termos técnicos exatos '
                    f'(SAP, ABAP, Java, AWS) pesam mais que o sentido geral do texto. O app usa um TF-IDF com vocabulário aprendido só '
                    f'das vagas (AUC {num(B["TF-IDF do app (vocabulário das vagas)"]["auc_media"], 3)}), sem nenhum texto de currículo.</div>')
        with gr.Tab("2 · Decisão"):
            ROT = {"Supervisionado: só similaridades": "Supervisionado: similaridades",
                   "Supervisionado: features do app": "Supervisionado: similaridade + requisitos",
                   "Supervisionado: todas as features": "Supervisionado: todas as variáveis"}
            gr.Plot(fig_auc({ROT.get(k, k): R[k] for k in R}, "AUC dentro da vaga: contratado × não contratado"), show_label=False)
            gr.HTML(f'<div class="note"><b>Leitura.</b> Entre os candidatos já prospectados, nenhum modelo separa bem o contratado. O melhor '
                    f'(TF-IDF) tem AUC de {num(TFR["auc_media"], 3)}, com intervalo de {num(TFR["auc_ic95"][0], 3)} a {num(TFR["auc_ic95"][1], 3)}: '
                    f'acima do acaso, mas por pouco. Colocar um contratado entre os 3 primeiros acontece em {pct(TFR["hit@3"], 0)} das vagas, contra '
                    f'{pct(ALE["hit@3"], 0)} numa lista embaralhada. Embeddings e modelos supervisionados com requisitos (nível, inglês, formação) '
                    f'ficam dentro da margem do acaso. A decisão final depende do que o currículo não mostra: entrevista, pretensão, '
                    f'disponibilidade, fit com o cliente.</div>'
                    f'<p class="muted" style="margin-top:10px">{num(av["pares"])} pares em {num(av["vagas"])} vagas ({av["criterios"]}). '
                    f'Supervisionados com validação cruzada agrupada por vaga (5 folds); intervalos por bootstrap de vagas.</p>')
        with gr.Tab("Funil"):
            with gr.Row():
                gr.Plot(fig_funil(), show_label=False)
                gr.Plot(fig_contrato(), show_label=False)
            sap = {r["sap"]: r for r in F["por_sap"]}
            gr.HTML(f'<div class="note">{num(F["prospects"])} prospecções em {num(F["vagas_com_prospects"])} vagas. Só '
                    f'{pct(F["vagas_com_contratacao_pct"], 0)} das vagas têm uma contratação registrada. Vagas SAP convertem '
                    f'{pct(sap["Vaga SAP"]["taxa_contratacao"])} das prospecções, contra {pct(sap["Vaga não SAP"]["taxa_contratacao"])} nas demais. '
                    f'{num(F["desistencias"])} desistências ficam fora do funil: a base guarda só a situação final.</div>')
        with gr.Tab("Testar um match"):
            gr.HTML('<div class="note"><b>Exemplo fictício.</b> Cole qualquer vaga e currículo; nada é salvo. '
                    'O resultado indica aderência de texto, útil para priorizar a busca, não para decidir a contratação.</div>')
            with gr.Row():
                vaga = gr.Textbox(EX_VAGA, label="Descrição da vaga", lines=8)
                cv = gr.Textbox(EX_CV, label="Currículo", lines=8)
            bt = gr.Button("Calcular aderência", variant="primary")
            out = gr.HTML()
            bt.click(avaliar, [vaga, cv], out)
            demo.load(avaliar, [vaga, cv], out)
        with gr.Tab("Método"):
            gr.Markdown(
                "**Problema.** A Decision prospecta dezenas de candidatos por vaga e entrevista sob pressão de prazo. Onde a IA "
                "economiza tempo: na busca de candidatos, na escolha final, ou nas duas?\n\n"
                "**Dados.** Bases do Datathon (vagas, prospecções e candidatos, anonimizados pela Decision). O ETL transforma os três "
                "JSONs aninhados em tabelas e descarta nomes, contatos, comentários e remuneração.\n\n"
                "**Etapa 1 · Busca.** Para cada vaga, os prospectados disputam com uma amostra fixa de candidatos da base que não foram "
                f"prospectados para ela. Métricas: AUC e recall no top {TOP} (~1% da lista).\n\n"
                "**Etapa 2 · Decisão.** Para cada vaga, a lista de prospectados é ordenada; mede-se a AUC entre contratados e não "
                "contratados, hit@3/hit@5 e MRR. O aleatório é o valor esperado calculado, não um sorteio.\n\n"
                "**Modelos.** TF-IDF; embeddings em inglês (`all-MiniLM-L6-v2`, o modelo salvo no projeto original do grupo) e "
                "multilíngue (`paraphrase-multilingual-MiniLM-L12-v2`); HistGradientBoosting com similaridade e requisitos, validado "
                "com GroupKFold por vaga.\n\n"
                "**Limitações.** Os prospectados refletem o processo de busca dos hunters, inclusive por palavras-chave, o que favorece "
                "o TF-IDF na etapa 1. A base guarda só a situação final; \"não contratado\" inclui quem desistiu ou ainda estava em "
                "processo. Aderência de texto é apoio à triagem, nunca filtro automático.")
    gr.HTML('<p class="muted" style="margin-top:20px">Mariana Barbalho · <a href="https://maribarbalho.github.io/">Portfólio</a> · '
            '<a href="https://github.com/maribarbalho/match-candidatos-vagas">Código no GitHub</a></p>')

if __name__ == "__main__":
    demo.launch(css=CSS, js="() => { document.body.classList.add('dark'); }",
                theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"))
