"""Avalia rankers de candidatos por vaga e treina o modelo final.

Uso: python -m match.train

Tarefa: para cada vaga, ordenar os candidatos que a Decision prospectou. Um bom
ranking coloca quem foi contratado no topo. Rankers comparados:
  1. aleatório (valor esperado)        3. embeddings all-MiniLM-L6-v2 (inglês)
  2. TF-IDF                            4. embeddings paraphrase-multilingual-MiniLM-L12-v2
  5. supervisionado: HistGradientBoosting sobre similaridade + requisitos,
     com validação cruzada agrupada por vaga (nenhuma vaga aparece no treino e no teste)
Saídas: reports/metrics.json e models/ranker.joblib (só features que o app consegue calcular).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold

from .evaluate import aleatorio, busca, ranking
from .features import tabular

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
SEED = 42
MULTI = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MIN_PALAVRAS_CV = 50
MIN_PROSPECTS = 5
FEATURES_APP = ["sim_multi", "gap_ingles", "gap_academico", "gap_nivel", "termos_comuns", "cv_palavras", "vaga_sap"]


def pares_avaliacao() -> pd.DataFrame:
    v = pd.read_parquet(PROCESSED / "vagas.parquet")
    p = pd.read_parquet(PROCESSED / "prospects.parquet")
    c = pd.read_parquet(PROCESSED / "candidatos.parquet")
    c = c[c["texto"].str.split().str.len() >= MIN_PALAVRAS_CV]
    df = (p.merge(v, on="vaga_id", suffixes=("", "_v"))
            .merge(c, on="candidato_id", suffixes=("_vaga", "_cand")))
    df = df.rename(columns={"vaga_sap": "vaga_sap"})
    # Só vagas com pelo menos um contratado e um número mínimo de prospects com CV.
    g = df.groupby("vaga_id")["contratado"].agg(["size", "sum"])
    ok = g[(g["size"] >= MIN_PROSPECTS) & (g["sum"] >= 1) & (g["sum"] < g["size"])].index
    return df[df["vaga_id"].isin(ok)].reset_index(drop=True)


def cos_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=1, keepdims=True).clip(1e-9)
    b = b / np.linalg.norm(b, axis=1, keepdims=True).clip(1e-9)
    return (a * b).sum(axis=1)


def embed(model_name: str, textos: pd.Series, cache: Path) -> np.ndarray:
    if cache.exists():
        return np.load(cache)
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_name, cache_folder=str(MODELS / "hf"), device="cpu")
    m.max_seq_length = 256
    e = m.encode(textos.tolist(), batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    np.save(cache, e.astype(np.float32))
    return e


def similaridades(df: pd.DataFrame) -> pd.DataFrame:
    vag = df.drop_duplicates("vaga_id")[["vaga_id", "texto_vaga"]].reset_index(drop=True)
    cand = df.drop_duplicates("candidato_id")[["candidato_id", "texto_cand"]].reset_index(drop=True)
    vi = dict(zip(vag["vaga_id"], vag.index))
    ci = dict(zip(cand["candidato_id"], cand.index))
    iv, ic = df["vaga_id"].map(vi).to_numpy(), df["candidato_id"].map(ci).to_numpy()

    tf = TfidfVectorizer(strip_accents="unicode", lowercase=True, ngram_range=(1, 2), min_df=3, max_df=0.5,
                         sublinear_tf=True, max_features=200_000)
    tf.fit(pd.concat([vag["texto_vaga"], cand["texto_cand"]]))
    V, C = tf.transform(vag["texto_vaga"]), tf.transform(cand["texto_cand"])
    df["sim_tfidf"] = np.asarray(V[iv].multiply(C[ic]).sum(axis=1)).ravel()

    for nome, modelo in [("en", "models/encoder"), ("multi", MULTI)]:
        src = str(ROOT / modelo) if modelo.startswith("models/") else modelo
        ev = embed(src, vag["texto_vaga"], PROCESSED / f"emb_vagas_{nome}.npy")
        ec = embed(src, cand["texto_cand"], PROCESSED / f"emb_cand_{nome}.npy")
        df[f"sim_{nome}"] = cos_rows(ev[iv], ec[ic])
    return df


def supervisionado(df: pd.DataFrame, feats: list[str]) -> np.ndarray:
    X, y, g = df[feats].to_numpy(float), df["contratado"].to_numpy(int), df["vaga_id"].to_numpy()
    oof = np.zeros(len(df))
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                           l2_regularization=1.0, class_weight="balanced", random_state=SEED)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def etapa_busca(df: pd.DataFrame, n_base: int = 4000, top: int = 40) -> tuple[dict, dict]:
    """O modelo encontra, na base inteira, os candidatos que os hunters prospectaram para a vaga?

    Para cada vaga avaliada, os prospects (positivos) disputam com uma amostra fixa de
    `n_base` candidatos da base que não foram prospectados para ela (negativos).
    Mede-se a AUC e o recall no top-`top` da lista combinada (~1% da base amostrada).
    """
    c = pd.read_parquet(PROCESSED / "candidatos.parquet")
    c = c[c["texto"].str.split().str.len() >= MIN_PALAVRAS_CV]
    base = c.sample(n=n_base, random_state=SEED).reset_index(drop=True)
    vag = df.drop_duplicates("vaga_id")[["vaga_id", "texto_vaga"]].reset_index(drop=True)
    pros = df[["vaga_id", "candidato_id", "texto_cand"]]

    tf = TfidfVectorizer(strip_accents="unicode", lowercase=True, ngram_range=(1, 2), min_df=3, max_df=0.5,
                         sublinear_tf=True, max_features=200_000)
    tf.fit(pd.concat([vag["texto_vaga"], base["texto"], pros["texto_cand"]]))
    Vt, Bt = tf.transform(vag["texto_vaga"]), tf.transform(base["texto"])

    # Versão do app: vocabulário aprendido só com o texto das 14 mil vagas (nenhum texto de currículo
    # vai para o artefato publicado).
    todas_vagas = pd.read_parquet(PROCESSED / "vagas.parquet")["texto"]
    tf_app = TfidfVectorizer(strip_accents="unicode", lowercase=True, ngram_range=(1, 2), min_df=3, max_df=0.5,
                             sublinear_tf=True, max_features=50_000)
    tf_app.fit(todas_vagas[todas_vagas.str.len() > 0])
    # Remove do vocabulário termos com nomes de pessoas (fictícios, mas ainda nomes) e reajusta o IDF.
    excluir = tokens_de_nomes()
    vocab = sorted(t for t in tf_app.vocabulary_ if not (set(t.split()) & excluir))
    tf_app = TfidfVectorizer(strip_accents="unicode", lowercase=True, ngram_range=(1, 2), sublinear_tf=True,
                             vocabulary=vocab)
    tf_app.fit(todas_vagas[todas_vagas.str.len() > 0])
    Va, Ba = tf_app.transform(vag["texto_vaga"]), tf_app.transform(base["texto"])

    ev = np.load(PROCESSED / "emb_vagas_multi.npy")  # mesma ordem de drop_duplicates("vaga_id")
    eb = embed(MULTI, base["texto"], PROCESSED / "emb_base_multi.npy")
    norm = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True).clip(1e-9)  # noqa: E731
    ev, eb = norm(ev), norm(eb)

    res = {"TF-IDF": [], "TF-IDF do app (vocabulário das vagas)": [], "Embeddings multilíngue (MiniLM-L12)": []}
    sims_prospect, sims_base = [], []
    for i, row in vag.iterrows():
        g = df[df["vaga_id"] == row["vaga_id"]]
        fora = ~base["candidato_id"].isin(g["candidato_id"]).to_numpy()
        # TF-IDF (positivos e negativos com o mesmo vocabulário)
        pos_t = (tf.transform(g["texto_cand"]) @ Vt[i].T).toarray().ravel()
        neg_t = (Bt[fora] @ Vt[i].T).toarray().ravel()
        res["TF-IDF"].append(busca(pos_t, neg_t, top))
        pos_a = (tf_app.transform(g["texto_cand"]) @ Va[i].T).toarray().ravel()
        neg_a = (Ba[fora] @ Va[i].T).toarray().ravel()
        res["TF-IDF do app (vocabulário das vagas)"].append(busca(pos_a, neg_a, top))
        # Embeddings
        pos_e = g["sim_multi"].to_numpy()
        neg_e = eb[fora] @ ev[i]
        res["Embeddings multilíngue (MiniLM-L12)"].append(busca(pos_e, neg_e, top))
        sims_prospect.append(pos_a)
        sims_base.append(neg_a[:200])

    out = {"Aleatório (esperado)": {"auc_media": 0.5, f"recall@{top}": round(top / (n_base + 9), 4)}}
    for k, v in res.items():
        a = np.array(v)
        out[k] = {"auc_media": float(a[:, 0].mean()), "auc_ic95": _ic_vagas(a[:, 0]),
                  f"recall@{top}": float(a[:, 1].mean())}
    out["_config"] = {"base_amostrada": n_base, "top": top, "vagas": int(len(vag))}
    q = np.linspace(0, 1, 101)
    quantis = {"prospectados": [round(float(x), 4) for x in np.quantile(np.concatenate(sims_prospect), q)],
               "base": [round(float(x), 4) for x in np.quantile(np.concatenate(sims_base), q)]}
    MODELS.mkdir(exist_ok=True)
    joblib.dump(tf_app, MODELS / "tfidf_vagas.joblib", compress=3)
    for k, v in out.items():
        if not k.startswith("_"):
            print(f"  busca · {k:38s} AUC {v['auc_media']:.3f}  recall@{top} {v[f'recall@{top}']:.3f}")
    return out, quantis


def tokens_de_nomes() -> set[str]:
    """Tokens de nomes de pessoas presentes nas bases (candidatos, requisitantes, analistas), exceto
    os que também nomeiam lugares das vagas (ex.: "paulo" em São Paulo, "jose" em São José dos Campos)."""
    import json as _json
    import unicodedata
    norm = lambda t: "".join(c for c in unicodedata.normalize("NFKD", t.lower()) if not unicodedata.combining(c))  # noqa: E731
    nomes = set()
    raw = ROOT / "data" / "raw"
    for a in _json.loads((raw / "applicants.json").read_text(encoding="utf-8")).values():
        nomes.update(norm(a.get("infos_basicas", {}).get("nome", "")).split())
    lugares = set()
    for v in _json.loads((raw / "vagas.json").read_text(encoding="utf-8")).values():
        b, p = v.get("informacoes_basicas", {}), v.get("perfil_vaga", {})
        for campo in ("requisitante", "analista_responsavel", "solicitante_cliente"):
            nomes.update(norm(b.get(campo, "")).split())
        for campo in ("cidade", "estado", "bairro", "regiao", "local_trabalho"):
            lugares.update(norm(p.get(campo, "") or "").split())
    titulos = {"sr", "sra", "srta", "dr", "dra"}  # "sr" também abrevia "sênior": tratado abaixo
    return {t for t in nomes - lugares - titulos if len(t) > 2 and t.isalpha()}


def _ic_vagas(v: np.ndarray) -> list[float]:
    rng = np.random.default_rng(SEED)
    m = rng.choice(v, size=(2000, len(v)), replace=True).mean(axis=1)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def main() -> None:
    t0 = time.time()
    REPORTS.mkdir(exist_ok=True)
    df = pares_avaliacao()
    print(f"Pares de avaliação: {len(df):,} em {df['vaga_id'].nunique():,} vagas")
    df = similaridades(df)
    tab = tabular(df)
    df = pd.concat([df, tab], axis=1)

    df["score_sup_texto"] = supervisionado(df, ["sim_tfidf", "sim_multi", "sim_en", "termos_comuns"])
    df["score_sup_app"] = supervisionado(df, FEATURES_APP)
    df["score_sup_todas"] = supervisionado(df, FEATURES_APP + ["sim_tfidf", "sim_en"])

    rankers = {
        "Aleatório (esperado)": aleatorio(df),
        "TF-IDF": ranking(df, "sim_tfidf"),
        "Embeddings em inglês (all-MiniLM-L6-v2)": ranking(df, "sim_en"),
        "Embeddings multilíngue (MiniLM-L12)": ranking(df, "sim_multi"),
        "Supervisionado: só similaridades": ranking(df, "score_sup_texto"),
        "Supervisionado: features do app": ranking(df, "score_sup_app"),
        "Supervisionado: todas as features": ranking(df, "score_sup_todas"),
    }
    for k, v in rankers.items():
        print(f"  {k:42s} AUC {v['auc_media']:.3f}  hit@3 {v['hit@3']:.3f}  hit@5 {v['hit@5']:.3f}  MRR {v['mrr']:.3f}")

    # Diagnóstico: quanto a similaridade separa contratados dos demais, no geral.
    sep = df.groupby("contratado")[["sim_tfidf", "sim_en", "sim_multi", "termos_comuns"]].median().round(4)

    busca_res, quantis = etapa_busca(df)

    tam = df.groupby("vaga_id").agg(n=("contratado", "size"), h=("contratado", "sum"))
    metrics = {
        "avaliacao": {
            "pares": int(len(df)), "vagas": int(df["vaga_id"].nunique()),
            "candidatos": int(df["candidato_id"].nunique()),
            "prospects_por_vaga_mediana": float(tam["n"].median()),
            "contratados_por_vaga_mediana": float(tam["h"].median()),
            "criterios": f"vagas com >= {MIN_PROSPECTS} prospects com CV (>= {MIN_PALAVRAS_CV} palavras) e ao menos um contratado",
        },
        "rankers": rankers,
        "busca": busca_res,
        "mediana_similaridade": {str(k): v for k, v in sep.to_dict("index").items()},
        "quantis_tfidf_app": quantis,
        "tempo_s": round(time.time() - t0, 1),
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8", newline="\n")
    print(f"Pronto em {metrics['tempo_s']} s")


if __name__ == "__main__":
    main()
