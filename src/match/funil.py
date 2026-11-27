"""Funil de recrutamento da Decision: onde os candidatos param e quanto tempo leva.

Uso: python -m match.funil  →  reports/funil.json (só agregados, nenhum dado individual)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"


def _taxa(df: pd.DataFrame, by: str, min_n: int = 300) -> list[dict]:
    g = (df.groupby(by, dropna=True)
           .agg(prospects=("contratado", "size"), contratados=("contratado", "sum"), desistencias=("desistiu", "sum"))
           .query("prospects >= @min_n"))
    g["taxa_contratacao"] = (g["contratados"] / g["prospects"]).round(4)
    g["taxa_desistencia"] = (g["desistencias"] / g["prospects"]).round(4)
    return g.sort_values("taxa_contratacao", ascending=False).reset_index().to_dict("records")


def main() -> dict:
    p = pd.read_parquet(PROCESSED / "prospects.parquet")
    v = pd.read_parquet(PROCESSED / "vagas.parquet")
    df = p.merge(v[["vaga_id", "nivel_profissional", "tipo_contratacao", "vaga_sap", "nivel_ingles"]], on="vaga_id", how="left")

    # Situação atual = etapa mais avançada que o candidato alcançou.
    ordem = ["1. Inscrito", "2. Triagem RH", "3. Encaminhado ao cliente", "4. Entrevistas", "5. Contratado"]
    etapa = p["etapa"].where(p["etapa"].isin(ordem))
    alcancou = {e: int((etapa >= e).sum()) for e in ordem}

    por_vaga = p.groupby("vaga_id").agg(prospects=("contratado", "size"), contratados=("contratado", "sum"))
    contratados = p[p["contratado"]].copy()
    contratados["dias"] = (contratados["ultima_atualizacao"] - contratados["data_candidatura"]).dt.days
    dias = contratados["dias"][contratados["dias"].between(0, 365)]

    situacoes = p["situacao"].value_counts().rename_axis("situacao").reset_index(name="prospects").to_dict("records")
    out = {
        "prospects": int(len(p)),
        "vagas_com_prospects": int(p["vaga_id"].nunique()),
        "vagas_total": int(len(v)),
        "candidatos_unicos": int(p["candidato_id"].nunique()),
        "contratados": int(p["contratado"].sum()),
        "desistencias": int(p["desistiu"].sum()),
        "funil_alcancou": alcancou,
        "situacoes": situacoes,
        "vagas_com_contratacao_pct": round(float((por_vaga["contratados"] > 0).mean()), 4),
        "prospects_por_vaga_mediana": float(por_vaga["prospects"].median()),
        "prospects_por_contratacao": round(len(p) / max(int(p["contratado"].sum()), 1), 1),
        "dias_ate_contratacao": {"mediana": float(dias.median()), "p25": float(dias.quantile(.25)),
                                 "p75": float(dias.quantile(.75)), "n": int(dias.size)},
        "por_nivel": _taxa(df, "nivel_profissional"),
        "por_contratacao": _taxa(df, "tipo_contratacao"),
        "por_sap": _taxa(df.assign(sap=df["vaga_sap"].map({True: "Vaga SAP", False: "Vaga não SAP"})), "sap"),
        "por_ingles": _taxa(df, "nivel_ingles"),
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "funil.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str) + "\n",
                                        encoding="utf-8", newline="\n")
    print(json.dumps({k: out[k] for k in ["prospects", "contratados", "desistencias", "funil_alcancou",
                                          "vagas_com_contratacao_pct", "prospects_por_contratacao",
                                          "dias_ate_contratacao"]}, ensure_ascii=False, indent=1))
    return out


if __name__ == "__main__":
    main()
