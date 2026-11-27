"""Features de aderência entre vaga e candidato (usadas no treino e no app)."""

from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

INGLES = {"Nenhum": 0, "Básico": 1, "Intermediário": 2, "Avançado": 3, "Fluente": 4}
ACADEMICO = {
    "Ensino Fundamental Incompleto": 0, "Ensino Fundamental Completo": 1, "Ensino Médio Incompleto": 2,
    "Ensino Médio Completo": 3, "Ensino Técnico Incompleto": 3, "Ensino Técnico Completo": 4,
    "Ensino Superior Incompleto": 4, "Ensino Superior Cursando": 4, "Ensino Superior Completo": 5,
    "Pós Graduação Incompleto": 5, "Pós Graduação Cursando": 5, "Pós Graduação Completo": 6,
    "Mestrado Incompleto": 6, "Mestrado Cursando": 6, "Mestrado Completo": 7,
    "Doutorado Incompleto": 7, "Doutorado Cursando": 7, "Doutorado Completo": 8,
}
NIVEL = {"Estagiário": 0, "Auxiliar": 1, "Assistente": 1, "Técnico de Nível Médio": 2, "Trainee": 1,
         "Júnior": 2, "Analista": 3, "Pleno": 3, "Sênior": 4, "Especialista": 5, "Líder": 5,
         "Coordenador": 5, "Supervisor": 5, "Gerente": 6}

STOP = set("""a o e de da do das dos em para por com sem um uma no na nos nas ao aos à às que se
como mais ou ser ter seu sua seus suas the and of to in for with on at by is are be as an or
experiência conhecimento conhecimentos atividades trabalho empresa área""".split())


def _norm(t: str) -> str:
    t = unicodedata.normalize("NFKD", t.lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def termos(texto: str) -> set[str]:
    """Termos técnicos candidatos: palavras de 2+ caracteres (inclui siglas como sql, sap, aws)."""
    return {w for w in re.findall(r"[a-z0-9+#.]{2,}", _norm(texto or "")) if w not in STOP and not w.isdigit()}


def termos_em_comum(vaga: str, cv: str, limite: int = 12) -> list[str]:
    v, c = termos(vaga), termos(cv)
    comuns = sorted(v & c, key=lambda w: (-len(w), w))
    return comuns[:limite]


def gap(req, tem, escala: dict) -> float:
    """Quanto o candidato fica abaixo do requisito (0 = atende). NaN se algum lado não informou."""
    r, t = escala.get(req), escala.get(tem)
    if r is None or t is None:
        return np.nan
    return float(max(r - t, 0))


def tabular(pares: pd.DataFrame) -> pd.DataFrame:
    """Features não textuais a partir de colunas *_vaga e *_cand."""
    out = pd.DataFrame(index=pares.index)
    out["gap_ingles"] = [gap(r, t, INGLES) for r, t in zip(pares["nivel_ingles_vaga"], pares["nivel_ingles_cand"])]
    out["gap_academico"] = [gap(r, t, ACADEMICO) for r, t in zip(pares["nivel_academico_vaga"], pares["nivel_academico_cand"])]
    out["gap_nivel"] = [
        (NIVEL.get(t, np.nan) - NIVEL.get(r, np.nan)) if r in NIVEL and t in NIVEL else np.nan
        for r, t in zip(pares["nivel_profissional_vaga"], pares["nivel_profissional_cand"])
    ]
    out["cv_palavras"] = pares["texto_cand"].str.split().str.len().fillna(0)
    out["termos_comuns"] = [len(termos(v) & termos(c)) for v, c in zip(pares["texto_vaga"], pares["texto_cand"])]
    out["vaga_sap"] = pares["vaga_sap"].astype(float)
    return out
