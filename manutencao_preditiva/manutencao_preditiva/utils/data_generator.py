"""
Gerador de dados simulados baseado no formato NASA C-MAPSS
(Commercial Modular Aero-Propulsion System Simulation)
Adaptado para máquinas de hemodiálise e respiradores pulmonares.

Referência dos datasets reais:
- NASA CMAPS: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
- Predictive Maintenance: https://www.kaggle.com/c/predictive-maintenance/data

Colunas do NASA CMAPS:
  unit_id, cycle, op_setting1-3, sensor1-21
  RUL (Remaining Useful Life) calculado a partir do ciclo máximo
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def _degradation_curve(ciclo, max_ciclo, ruido=0.05):
    """
    Curva de degradação sigmoidal — imita o comportamento real
    onde o equipamento opera normalmente até entrar em zona crítica.
    """
    progresso = ciclo / max_ciclo
    base = 1 / (1 + np.exp(-10 * (progresso - 0.7)))
    return base + np.random.normal(0, ruido, size=len(ciclo) if hasattr(ciclo, '__len__') else 1)


def gerar_dados_hemodialise(n_maquinas=10, seed=42):
    """
    Gera dataset simulando sensores de máquinas de hemodiálise.
    Inspirado no formato NASA CMAPS com adaptação para parâmetros clínicos.

    Sensores simulados:
      - temperatura_sangue    : temperatura do circuito sanguíneo (°C)
      - pressao_arterial      : pressão na linha arterial (mmHg)
      - pressao_venosa        : pressão na linha venosa (mmHg)
      - fluxo_sanguineo       : fluxo do sangue (mL/min)
      - condutividade         : condutividade do dialisado (mS/cm)
      - temperatura_dialisado : temperatura do dialisado (°C)
      - fluxo_dialisado       : fluxo do dialisado (mL/min)
      - vibração              : vibração da bomba (mm/s)
      - temperatura_motor     : temperatura do motor da bomba (°C)
    """
    np.random.seed(seed)
    registros = []
    agora = datetime.now()

    # Ciclos de vida variáveis por máquina (inspirado no NASA CMAPS FD001-FD004)
    ciclos_vida = {i: np.random.randint(150, 350) for i in range(1, n_maquinas + 1)}

    for maquina_id in range(1, n_maquinas + 1):
        max_ciclo = ciclos_vida[maquina_id]
        # Simula máquina em diferentes estágios de uso
        ciclo_atual = np.random.randint(20, max_ciclo)
        ciclos = np.arange(1, ciclo_atual + 1)
        rul = max_ciclo - ciclos  # Remaining Useful Life

        deg = _degradation_curve(ciclos, max_ciclo, ruido=0.03)

        for i, ciclo in enumerate(ciclos):
            d = deg[i]
            timestamp = agora - timedelta(hours=int(ciclo_atual - ciclo) * 4)

            # Parâmetros normais com degradação progressiva
            temp_sangue = np.random.normal(37.0 + d * 1.5, 0.3)
            pressao_art = np.random.normal(-(150 + d * 30), 8)   # negativa (sucção)
            pressao_ven = np.random.normal(180 + d * 40, 10)
            fluxo_sang = np.random.normal(300 - d * 50, 15)
            condutividade = np.random.normal(14.0 - d * 0.8, 0.2)
            temp_dialisado = np.random.normal(37.0 + d * 2.0, 0.4)
            fluxo_dialisado = np.random.normal(500 - d * 80, 20)
            vibracao = np.random.normal(0.5 + d * 3.5, 0.2)
            temp_motor = np.random.normal(45 + d * 25, 2.0)

            # Status baseado no RUL (como no NASA CMAPS)
            rul_val = rul[i]
            if rul_val < 20:
                status = "Crítico"
            elif rul_val < 50:
                status = "Alerta"
            else:
                status = "Normal"

            registros.append({
                "maquina_id": f"HD-{maquina_id:02d}",
                "tipo": "Hemodiálise",
                "ciclo": int(ciclo),
                "rul": int(rul_val),
                "timestamp": timestamp,
                "temperatura_sangue": round(temp_sangue, 2),
                "pressao_arterial": round(pressao_art, 1),
                "pressao_venosa": round(pressao_ven, 1),
                "fluxo_sanguineo": round(fluxo_sang, 1),
                "condutividade": round(condutividade, 2),
                "temperatura_dialisado": round(temp_dialisado, 2),
                "fluxo_dialisado": round(fluxo_dialisado, 1),
                "vibracao": round(max(0.1, vibracao), 3),
                "temperatura_motor": round(temp_motor, 1),
                "status": status,
            })

    return pd.DataFrame(registros)


def gerar_dados_respirador(n_maquinas=10, seed=99):
    """
    Gera dataset simulando sensores de respiradores pulmonares (ventiladores).
    Formato inspirado no NASA CMAPS adaptado para parâmetros ventilatórios.

    Sensores simulados:
      - pressao_pico          : pressão de pico nas vias aéreas (cmH2O)
      - pressao_plateau       : pressão de plateau (cmH2O)
      - peep                  : pressão expiratória final positiva (cmH2O)
      - volume_corrente       : volume corrente entregue (mL)
      - frequencia_resp       : frequência respiratória (rpm)
      - fio2                  : fração inspirada de O2 (%)
      - fluxo_inspiratorio    : fluxo inspiratório de pico (L/min)
      - temperatura_circuito  : temperatura do circuito (°C)
      - umidade               : umidade relativa no circuito (%)
      - vibracao              : vibração do compressor (mm/s)
      - temperatura_compressor: temperatura do compressor (°C)
    """
    np.random.seed(seed)
    registros = []
    agora = datetime.now()

    ciclos_vida = {i: np.random.randint(120, 300) for i in range(1, n_maquinas + 1)}

    for maquina_id in range(1, n_maquinas + 1):
        max_ciclo = ciclos_vida[maquina_id]
        ciclo_atual = np.random.randint(15, max_ciclo)
        ciclos = np.arange(1, ciclo_atual + 1)
        rul = max_ciclo - ciclos

        deg = _degradation_curve(ciclos, max_ciclo, ruido=0.04)

        for i, ciclo in enumerate(ciclos):
            d = deg[i]
            timestamp = agora - timedelta(hours=int(ciclo_atual - ciclo) * 6)

            pressao_pico = np.random.normal(20 + d * 12, 1.5)
            pressao_plateau = np.random.normal(15 + d * 8, 1.0)
            peep = np.random.normal(5 - d * 1.5, 0.5)
            volume_corrente = np.random.normal(500 - d * 80, 20)
            freq_resp = np.random.normal(16 + d * 6, 1.5)
            fio2 = np.random.normal(40 + d * 20, 3.0)
            fluxo_insp = np.random.normal(60 - d * 15, 3.0)
            temp_circuito = np.random.normal(37 + d * 3, 0.5)
            umidade = np.random.normal(95 - d * 20, 2.0)
            vibracao = np.random.normal(0.3 + d * 2.5, 0.15)
            temp_compressor = np.random.normal(55 + d * 30, 3.0)

            rul_val = rul[i]
            if rul_val < 20:
                status = "Crítico"
            elif rul_val < 50:
                status = "Alerta"
            else:
                status = "Normal"

            registros.append({
                "maquina_id": f"RESP-{maquina_id:02d}",
                "tipo": "Respirador",
                "ciclo": int(ciclo),
                "rul": int(rul_val),
                "timestamp": timestamp,
                "pressao_pico": round(max(5, pressao_pico), 1),
                "pressao_plateau": round(max(3, pressao_plateau), 1),
                "peep": round(max(0, peep), 1),
                "volume_corrente": round(max(100, volume_corrente), 0),
                "frequencia_resp": round(max(8, freq_resp), 1),
                "fio2": round(min(100, max(21, fio2)), 1),
                "fluxo_inspiratorio": round(max(10, fluxo_insp), 1),
                "temperatura_circuito": round(temp_circuito, 2),
                "umidade": round(min(100, max(0, umidade)), 1),
                "vibracao": round(max(0.05, vibracao), 3),
                "temperatura_compressor": round(temp_compressor, 1),
                "status": status,
            })

    return pd.DataFrame(registros)


def snapshot_atual(df):
    """Retorna apenas o estado mais recente de cada máquina (último ciclo)."""
    return df.sort_values("ciclo").groupby("maquina_id").last().reset_index()


def historico_maquina(df, maquina_id):
    """Retorna o histórico completo de uma máquina específica."""
    return df[df["maquina_id"] == maquina_id].sort_values("ciclo")
