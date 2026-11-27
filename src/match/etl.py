"""ETL das bases da Decision (Datathon FIAP): três JSONs aninhados → tabelas Parquet.

Uso: python -m match.etl

Entrada (fora do Git): data/raw/{vagas,prospects,applicants}.json
Saída (fora do Git):   data/processed/{vagas,prospects,candidatos}.parquet

Só entram as colunas usadas na análise e no modelo. Nomes, telefones, e-mails,
comentários dos recrutadores e remuneração ficam de fora, mesmo anonimizados.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

# Situações finais de sucesso no processo.
CONTRATADO = {
    "Contratado pela Decision", "Contratado como Hunting", "Aprovado", "Proposta Aceita",
    "Documentação PJ", "Documentação CLT", "Documentação Cooperado", "Encaminhar Proposta",
}
# Etapa alcançada por situação (para o funil).
ETAPA = {
    "Inscrito": "1. Inscrito", "Prospect": "1. Inscrito",
    "Em avaliação pelo RH": "2. Triagem RH", "Não Aprovado pelo RH": "2. Triagem RH",
    "Encaminhado ao Requisitante": "3. Encaminhado ao cliente", "Não Aprovado pelo Requisitante": "3. Encaminhado ao cliente",
    "Entrevista Técnica": "4. Entrevistas", "Entrevista com Cliente": "4. Entrevistas", "Não Aprovado pelo Cliente": "4. Entrevistas",
    **{s: "5. Contratado" for s in CONTRATADO},
}
DESISTENCIA = {"Desistiu", "Desistiu da Contratação", "Sem interesse nesta vaga", "Recusado"}


def _load(name: str) -> dict:
    return json.loads((RAW / f"{name}.json").read_text(encoding="utf-8"))


def _date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.replace({"": None, "00-00-0000": None, "0000-00-00": None}),
                          format="%d-%m-%Y", errors="coerce")


def clean_text(text: str | None) -> str:
    """Normaliza espaços e remove caracteres de controle; mantém acentos."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"[•▪●·•▪]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def vagas() -> pd.DataFrame:
    rows = []
    for vid, v in _load("vagas").items():
        b, p = v.get("informacoes_basicas", {}), v.get("perfil_vaga", {})
        rows.append({
            "vaga_id": int(vid),
            "titulo": clean_text(b.get("titulo_vaga")),
            "data_requisicao": b.get("data_requicisao", ""),
            "vaga_sap": b.get("vaga_sap") == "Sim",
            "tipo_contratacao": b.get("tipo_contratacao") or None,
            "nivel_profissional": p.get("nivel profissional") or None,
            "nivel_academico": p.get("nivel_academico") or None,
            "nivel_ingles": p.get("nivel_ingles") or None,
            "nivel_espanhol": p.get("nivel_espanhol") or None,
            "area": (p.get("areas_atuacao") or "").strip("- ") or None,
            "estado": p.get("estado") or None,
            "texto": clean_text(" ".join(filter(None, [
                b.get("titulo_vaga"), p.get("principais_atividades"), p.get("competencia_tecnicas_e_comportamentais"),
            ]))),
        })
    df = pd.DataFrame(rows)
    df["data_requisicao"] = _date(df["data_requisicao"])
    return df


def prospects() -> pd.DataFrame:
    rows = []
    for vid, v in _load("prospects").items():
        for x in v.get("prospects", []):
            rows.append({
                "vaga_id": int(vid),
                "candidato_id": int(x["codigo"]),
                "situacao": x.get("situacao_candidado", "").strip(),
                "data_candidatura": x.get("data_candidatura", ""),
                "ultima_atualizacao": x.get("ultima_atualizacao", ""),
            })
    df = pd.DataFrame(rows)
    for c in ["data_candidatura", "ultima_atualizacao"]:
        df[c] = _date(df[c])
    df["etapa"] = df["situacao"].map(ETAPA).fillna("Outros")
    df["contratado"] = df["situacao"].isin(CONTRATADO)
    df["desistiu"] = df["situacao"].isin(DESISTENCIA)
    # O mesmo candidato pode aparecer duas vezes na mesma vaga; fica o registro mais recente.
    return (df.sort_values("ultima_atualizacao")
              .drop_duplicates(["vaga_id", "candidato_id"], keep="last")
              .reset_index(drop=True))


def candidatos() -> pd.DataFrame:
    rows = []
    for cid, a in _load("applicants").items():
        prof = a.get("informacoes_profissionais", {})
        form = a.get("formacao_e_idiomas", {})
        rows.append({
            "candidato_id": int(cid),
            "titulo_profissional": clean_text(prof.get("titulo_profissional")) or None,
            "area": clean_text(prof.get("area_atuacao")) or None,
            "nivel_profissional": prof.get("nivel_profissional") or None,
            "nivel_academico": form.get("nivel_academico") or None,
            "nivel_ingles": form.get("nivel_ingles") or None,
            "nivel_espanhol": form.get("nivel_espanhol") or None,
            "texto": clean_text(" ".join(filter(None, [
                prof.get("titulo_profissional"), prof.get("conhecimentos_tecnicos"),
                prof.get("certificacoes"), a.get("cv_pt"),
            ]))),
        })
    return pd.DataFrame(rows)


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    for name, fn in [("vagas", vagas), ("prospects", prospects), ("candidatos", candidatos)]:
        df = fn()
        df.to_parquet(PROCESSED / f"{name}.parquet", index=False)
        print(f"  {name}: {len(df):,} linhas")


if __name__ == "__main__":
    main()
