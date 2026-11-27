import numpy as np
import pandas as pd
import pytest

from match.etl import CONTRATADO, ETAPA, clean_text
from match.evaluate import aleatorio, busca, ranking
from match.features import INGLES, gap, termos, termos_em_comum


def _vaga(scores, hired, vid=1):
    return pd.DataFrame({"vaga_id": vid, "score": scores, "contratado": hired})


def test_ranking_perfeito_e_invertido():
    perfeito = ranking(_vaga([0.9, 0.5, 0.1, 0.0], [True, False, False, False]), "score")
    assert perfeito["auc_media"] == 1 and perfeito["hit@3"] == 1 and perfeito["mrr"] == 1
    invertido = ranking(_vaga([0.0, 0.5, 0.6, 0.9], [True, False, False, False]), "score")
    assert invertido["auc_media"] == 0 and invertido["hit@3"] == 0 and invertido["mrr"] == pytest.approx(0.25)


def test_empates_ficam_no_meio():
    r = ranking(_vaga([0.5, 0.5, 0.5, 0.5], [True, False, False, False]), "score")
    assert r["auc_media"] == pytest.approx(0.5)


def test_aleatorio_calculado():
    # n = 4 candidatos, 1 contratado: P(contratado entre os 3 primeiros) = 3/4; E[1/posição] = (1+1/2+1/3+1/4)/4.
    r = aleatorio(_vaga([0, 0, 0, 0], [True, False, False, False]))
    assert r["hit@3"] == pytest.approx(0.75)
    assert r["mrr"] == pytest.approx((1 + 1 / 2 + 1 / 3 + 1 / 4) / 4)


def test_aleatorio_bate_com_simulacao():
    rng = np.random.default_rng(0)
    df = pd.concat([_vaga(np.zeros(n), [True] * h + [False] * (n - h), vid=i)
                    for i, (n, h) in enumerate([(8, 1), (12, 2), (5, 1), (20, 3)])])
    esperado = aleatorio(df)
    sims = [ranking(df.assign(score=rng.random(len(df))), "score")["hit@3"] for _ in range(3000)]
    assert np.mean(sims) == pytest.approx(esperado["hit@3"], abs=0.01)


def test_busca_auc_e_recall():
    pos = np.array([0.9, 0.8, 0.2])
    neg = np.array([0.7, 0.5, 0.1, 0.0])
    auc, recall = busca(pos, neg, top=2)
    assert auc == pytest.approx((4 + 4 + 2) / 12)
    assert recall == pytest.approx(2 / 3)  # 0.9 e 0.8 ficam no top 2


def test_ranking_traz_intervalo_de_confianca():
    df = pd.concat([_vaga([0.9, 0.1, 0.2], [True, False, False], vid=i) for i in range(10)])
    r = ranking(df, "score")
    assert r["auc_ic95"][0] <= r["auc_media"] <= r["auc_ic95"][1]


def test_vagas_sem_contraste_sao_ignoradas():
    df = pd.concat([_vaga([1, 0], [True, True], vid=1), _vaga([1, 0], [True, False], vid=2)])
    assert ranking(df, "score")["vagas"] == 1


def test_gap_de_requisito():
    assert gap("Avançado", "Básico", INGLES) == 2
    assert gap("Básico", "Fluente", INGLES) == 0
    assert np.isnan(gap("Avançado", None, INGLES))


def test_termos_normaliza_acentos_e_siglas():
    assert {"sql", "python", "programacao"} <= termos("Programação em Python e SQL")
    assert "sql" in termos_em_comum("Experiência com SQL Server", "sql server e power bi")


def test_etl_limpa_texto_e_mapeia_situacoes():
    assert clean_text("  Python ▪ SQL\n\n•  Spark ") == "Python SQL Spark"
    assert all(ETAPA[s] == "5. Contratado" for s in CONTRATADO)
