"""Métricas de ranking por vaga: o contratado aparece no topo?"""

from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd


def _por_vaga(df: pd.DataFrame, score: str) -> pd.DataFrame:
    rows = []
    for vid, g in df.groupby("vaga_id", sort=False):
        y = g["contratado"].to_numpy(bool)
        s = g[score].to_numpy(float)
        n, h = len(y), int(y.sum())
        if h == 0 or h == n:
            continue
        # Empates recebem a posição média (ranking justo para scores repetidos).
        rank = pd.Series(-s).rank(method="average").to_numpy()
        pos, neg = s[y], s[~y]
        auc = ((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg))
        rows.append({
            "vaga_id": vid, "n": n, "contratados": h, "auc": auc,
            "hit3": float((rank[y] <= 3).any()), "hit5": float((rank[y] <= 5).any()),
            "rr": 1.0 / rank[y].min(),
        })
    return pd.DataFrame(rows)


def _ic(valores: np.ndarray, n_boot: int = 2000, seed: int = 42) -> list[float]:
    """Intervalo de 95% por bootstrap, reamostrando vagas."""
    rng = np.random.default_rng(seed)
    medias = rng.choice(valores, size=(n_boot, len(valores)), replace=True).mean(axis=1)
    return [float(np.quantile(medias, 0.025)), float(np.quantile(medias, 0.975))]


def ranking(df: pd.DataFrame, score: str) -> dict:
    r = _por_vaga(df, score)
    return {"vagas": int(len(r)), "auc_media": float(r["auc"].mean()), "auc_ic95": _ic(r["auc"].to_numpy()),
            "hit@3": float(r["hit3"].mean()), "hit@3_ic95": _ic(r["hit3"].to_numpy()),
            "hit@5": float(r["hit5"].mean()), "mrr": float(r["rr"].mean())}


def busca(pos: np.ndarray, neg: np.ndarray, top: int) -> tuple[float, float]:
    """Etapa de busca: AUC dos candidatos escolhidos contra a base e recall no top-k da lista combinada."""
    auc = ((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg))
    corte = np.sort(np.concatenate([pos, neg]))[::-1][top - 1]
    return float(auc), float((pos >= corte).mean())


def aleatorio(df: pd.DataFrame) -> dict:
    """Valores esperados de um ranking aleatório (calculados, não sorteados)."""
    rows = []
    for _, g in df.groupby("vaga_id", sort=False):
        n, h = len(g), int(g["contratado"].sum())
        if h == 0 or h == n:
            continue
        hit = lambda k: 1 - comb(n - h, min(k, n)) / comb(n, min(k, n))  # noqa: E731
        # E[1/posição do primeiro contratado]
        rr = sum((1 / i) * (comb(n - i, h - 1) / comb(n, h)) for i in range(1, n - h + 2))
        rows.append({"auc": 0.5, "hit3": hit(3), "hit5": hit(5), "rr": rr})
    r = pd.DataFrame(rows)
    return {"vagas": int(len(r)), "auc_media": 0.5, "hit@3": float(r["hit3"].mean()),
            "hit@5": float(r["hit5"].mean()), "mrr": float(r["rr"].mean())}
