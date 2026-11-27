# Match candidato × vaga: da busca à decisão

**[Abrir o app no Hugging Face ↗](https://huggingface.co/spaces/marianabarbalho/match-candidatos-vagas)**

Datathon da pós-graduação em Data Analytics (FIAP), com a base de recrutamento da **Decision**, uma empresa de bodyshop de TI. O desafio pedia IA para tornar o "match" entre candidatos e vagas mais rápido e preciso.

Em vez de assumir que a similaridade entre currículo e vaga resolve o problema, este projeto **mede** onde ela ajuda, usando o histórico real de decisões da empresa:

1. **Busca:** a similaridade encontra, na base inteira, os candidatos que os hunters escolheram para a vaga?
2. **Decisão:** entre os escolhidos, ela indica quem foi contratado?

## Resultados

| Pergunta | Melhor modelo | Resultado | Acaso |
|---|---|---|---|
| **Busca** (prospectado × 4.000 candidatos da base) | TF-IDF | **AUC 0,884** (IC 95%: 0,874–0,893) | 0,500 |
| | TF-IDF, prospectados no top 40 (~1%) | **28,4%** | 1,0% |
| | Embeddings multilíngues | AUC 0,749 | 0,500 |
| **Decisão** (contratado × não contratado, na mesma vaga) | TF-IDF | **AUC 0,539** (IC 95%: 0,513–0,565) | 0,500 |
| | TF-IDF, contratado entre os 3 primeiros | 47,3% das vagas | 41,0% |
| | Embeddings e modelos supervisionados | AUC 0,50–0,51, dentro da margem do acaso | 0,500 |

**Leitura:** a similaridade de texto reproduz bem a busca dos hunters e pode priorizar quem procurar na base, mas quase não diz quem será contratado. A decisão final depende do que o currículo não mostra (entrevista, pretensão, disponibilidade, fit com o cliente). Para a Decision, a IA rende mais na **triagem inicial** do que como "nota" de contratação. Nas vagas de TI, termos técnicos exatos (SAP, ABAP, Java, AWS) pesam mais que o sentido geral do texto, por isso o TF-IDF supera os embeddings.

### Funil de recrutamento

- 53.759 prospecções em 11.279 vagas; **16,8 prospecções por contratação**.
- Só **23% das vagas** têm contratação registrada.
- Vagas SAP convertem **2,7%** das prospecções, contra **6,1%** nas demais. Vagas PJ, **3,0%**, contra **12,8%** em CLT Full.
- Mediana de **9 dias** entre candidatura e última atualização nas contratações.

## Como foi feito

| Etapa | O que faz | Onde |
|---|---|---|
| ETL | Três JSONs aninhados (vagas, prospecções, candidatos) viram tabelas. Ficam fora nomes, contatos, comentários dos recrutadores e remuneração | `src/match/etl.py` |
| Funil | Etapa alcançada por prospect, taxas por nível, tipo de contrato, SAP e inglês | `src/match/funil.py` |
| Features | Similaridades (TF-IDF, embeddings), termos em comum, diferenças de requisito (inglês, formação, nível) | `src/match/features.py` |
| Avaliação | Métricas de ranking por vaga (AUC, hit@k, MRR), aleatório calculado analiticamente, intervalos por bootstrap de vagas | `src/match/evaluate.py` |
| Treino | Rankers comparados e HistGradientBoosting com validação cruzada agrupada por vaga (GroupKFold, 5 folds) | `src/match/train.py` |
| App | Gradio no Hugging Face Spaces: resultados e teste de aderência entre uma vaga e um currículo colados | `app.py` |

**Avaliação de 577 vagas** (com ao menos 5 prospects com currículo e ao menos um contratado): 6.439 pares candidato–vaga e 5.685 candidatos.

**Modelos comparados:** TF-IDF (1 e 2 gramas); `all-MiniLM-L6-v2`, o modelo salvo no projeto original do grupo, só em inglês; `paraphrase-multilingual-MiniLM-L12-v2`; e três versões supervisionadas.

### Privacidade

A base foi anonimizada pela própria Decision com nomes e contatos fictícios, mesmo assim:

- os dados brutos não estão no repositório;
- o app não mostra nenhum dado de candidato;
- o modelo publicado (`models/tfidf_vagas.joblib`) tem vocabulário aprendido **só com o texto das vagas**, e todo termo com nome de pessoa encontrado nas bases foi removido. Ficaram apenas palavras que também nomeiam lugares, como "São José";
- a versão do app tem praticamente o mesmo desempenho na busca (AUC 0,880 contra 0,884).

## Como reproduzir

Os dados do Datathon não são públicos. Coloque `vagas.json`, `prospects.json` e `applicants.json` em `data/raw/`.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

$env:PYTHONPATH = "src"
python -m match.etl        # JSON → Parquet
python -m match.funil      # reports/funil.json
python -m match.train      # embeddings, avaliação e modelo do app (~15 min em CPU na primeira vez)
pytest
python app.py              # http://localhost:7860
```

## Limitações

- A base guarda só a situação final de cada prospect: "não contratado" inclui quem desistiu ou ainda estava em processo.
- Os prospectados refletem como os hunters buscam, inclusive por palavras-chave, o que favorece o TF-IDF na etapa de busca.
- O modelo aprende com decisões passadas, inclusive seus vieses. É apoio à triagem, nunca filtro automático.

## Origem

Evolução do Datathon (fase 5) feito em grupo na pós da FIAP. A primeira versão ([datathon](https://github.com/maribarbalho/datathon)) estruturava as bases. Esta reescreve o ETL, avalia a abordagem de recomendação contra o histórico real e publica um app.

**Ferramentas:** Python, Pandas, scikit-learn, sentence-transformers, Plotly, Gradio, pytest, Hugging Face Spaces.

---

**Mariana Barbalho** · [Portfólio](https://maribarbalho.github.io/) · [LinkedIn](https://www.linkedin.com/in/mariana-barbalho-619254202)
