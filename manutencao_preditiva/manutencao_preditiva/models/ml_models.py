"""
Módulo de Machine Learning para Manutenção Preditiva
=====================================================
Implementa dois modelos principais baseados nos algoritmos do NASA CMAPS:

1. Isolation Forest  — detecção de anomalias (não supervisionado)
2. Random Forest     — predição de RUL (Remaining Useful Life)

Referência: Saxena et al. "Damage Propagation Modeling for Aircraft Engine
Run-to-Failure Simulation", NASA Ames 2008.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


# ─── Colunas de sensor por tipo de equipamento ────────────────────────────────

SENSORES_HEMODIALISE = [
    "temperatura_sangue", "pressao_arterial", "pressao_venosa",
    "fluxo_sanguineo", "condutividade", "temperatura_dialisado",
    "fluxo_dialisado", "vibracao", "temperatura_motor",
]

SENSORES_RESPIRADOR = [
    "pressao_pico", "pressao_plateau", "peep", "volume_corrente",
    "frequencia_resp", "fio2", "fluxo_inspiratorio",
    "temperatura_circuito", "umidade", "vibracao", "temperatura_compressor",
]


# ─── Isolation Forest — Detecção de Anomalias ────────────────────────────────

class DetectorAnomalias:
    """
    Wrapper do Isolation Forest (Liu et al., 2008) para detecção de anomalias
    em séries temporais de sensores hospitalares.

    contamination: proporção esperada de anomalias no dataset (5% padrão).
    """

    def __init__(self, contamination=0.05, random_state=42):
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self.feature_cols = None
        self.treinado = False

    def treinar(self, df: pd.DataFrame, feature_cols: list):
        self.feature_cols = feature_cols
        X = df[feature_cols].dropna()
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.treinado = True

    def prever(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.treinado:
            raise RuntimeError("Modelo não treinado. Chame treinar() primeiro.")

        resultado = df.copy()
        X = df[self.feature_cols].fillna(df[self.feature_cols].median())
        X_scaled = self.scaler.transform(X)

        # predict() retorna +1 (normal) ou -1 (anomalia)
        resultado["anomalia"] = self.model.predict(X_scaled)
        resultado["anomalia"] = resultado["anomalia"].map({1: False, -1: True})

        # Score de anomalia: quanto mais negativo, mais anômalo
        scores = self.model.score_samples(X_scaled)
        # Normaliza para [0, 1] onde 1 = anomalia máxima
        min_s, max_s = scores.min(), scores.max()
        if max_s > min_s:
            resultado["score_anomalia"] = 1 - (scores - min_s) / (max_s - min_s)
        else:
            resultado["score_anomalia"] = 0.0

        return resultado


# ─── Random Forest — Predição de RUL ─────────────────────────────────────────

class PreditorRUL:
    """
    Preditor de Remaining Useful Life (RUL) usando Random Forest Regressor.
    Treinado com dados históricos completos; prediz ciclos restantes
    até a falha esperada.
    """

    def __init__(self, random_state=42):
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        )
        self.feature_cols = None
        self.treinado = False
        self.metricas = {}

    def treinar(self, df: pd.DataFrame, feature_cols: list):
        self.feature_cols = feature_cols
        dados = df[feature_cols + ["rul"]].dropna()

        X = dados[feature_cols]
        y = dados["rul"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_train_sc = self.scaler.fit_transform(X_train)
        X_test_sc = self.scaler.transform(X_test)

        self.model.fit(X_train_sc, y_train)

        y_pred = self.model.predict(X_test_sc)
        self.metricas = {
            "MAE": round(mean_absolute_error(y_test, y_pred), 2),
            "R²": round(r2_score(y_test, y_pred), 4),
        }
        self.treinado = True

    def prever(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.treinado:
            raise RuntimeError("Modelo não treinado. Chame treinar() primeiro.")

        resultado = df.copy()
        X = df[self.feature_cols].fillna(df[self.feature_cols].median())
        X_scaled = self.scaler.transform(X)
        resultado["rul_predito"] = np.maximum(0, self.model.predict(X_scaled).round(0))
        return resultado

    def importancia_features(self) -> pd.DataFrame:
        imp = pd.DataFrame({
            "sensor": self.feature_cols,
            "importancia": self.model.feature_importances_,
        }).sort_values("importancia", ascending=False)
        imp["importancia_pct"] = (imp["importancia"] * 100).round(2)
        return imp


# ─── Pipeline Completo ────────────────────────────────────────────────────────

class PipelinePreditivo:
    """
    Pipeline completo que combina detecção de anomalias + predição de RUL.
    Encapsula o fluxo de treinamento e inferência para ambos os tipos.
    """

    def __init__(self, tipo: str):
        assert tipo in ("Hemodiálise", "Respirador"), \
            "tipo deve ser 'Hemodiálise' ou 'Respirador'"
        self.tipo = tipo
        self.sensores = (
            SENSORES_HEMODIALISE if tipo == "Hemodiálise" else SENSORES_RESPIRADOR
        )
        self.detector = DetectorAnomalias()
        self.preditor = PreditorRUL()

    def treinar(self, df: pd.DataFrame):
        subset = df[df["tipo"] == self.tipo]
        self.detector.treinar(subset, self.sensores)
        self.preditor.treinar(subset, self.sensores)

    def inferir(self, df: pd.DataFrame) -> pd.DataFrame:
        subset = df[df["tipo"] == self.tipo].copy()
        subset = self.detector.prever(subset)
        subset = self.preditor.prever(subset)

        # Urgência: combina RUL + score de anomalia
        subset["urgencia"] = _calcular_urgencia(subset)
        return subset

    @property
    def metricas_rul(self):
        return self.preditor.metricas


def _calcular_urgencia(df: pd.DataFrame) -> pd.Series:
    """
    Score de urgência combinando RUL predito e score de anomalia.
    Resultado: "Normal" | "Atenção" | "Alerta" | "Crítico"
    """
    def classificar(row):
        rul = row.get("rul_predito", 999)
        score = row.get("score_anomalia", 0)
        if rul < 15 or score > 0.85:
            return "Crítico"
        elif rul < 40 or score > 0.65:
            return "Alerta"
        elif rul < 70 or score > 0.45:
            return "Atenção"
        return "Normal"

    return df.apply(classificar, axis=1)
