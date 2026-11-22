# Sistema de Manutenção Preditiva com IA
## Projeto Integrador – Engenharia da Computação III – FAESA 2026

### Sobre o Projeto

Sistema inteligente de manutenção preditiva aplicado a:
- 10 máquinas de hemodiálise
- 10 respiradores pulmonares

Utiliza algoritmos de Machine Learning baseados no formato do dataset **NASA C-MAPSS**:
- **Isolation Forest** — detecção de anomalias (não supervisionado)
- **Random Forest Regressor** — predição de RUL (Remaining Useful Life)

### Datasets de Referência

Os dados são simulados seguindo o formato real dos datasets:
- NASA C-MAPSS: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
- Predictive Maintenance: https://www.kaggle.com/c/predictive-maintenance/data

Para usar os dados reais do Kaggle, baixe os arquivos CSV e adapte
a função `carregar_dados()` em `app.py` para lê-los diretamente.

### Estrutura do Projeto

```
manutencao_preditiva/
├── app.py                   ← Dashboard principal (Streamlit)
├── requirements.txt         ← Dependências Python
├── README.md
├── utils/
│   └── data_generator.py    ← Gerador de dados no formato NASA C-MAPSS
└── models/
    └── ml_models.py         ← Isolation Forest + Random Forest + Pipeline
```

### Instalação e Execução

```bash
# 1. Criar ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Rodar o dashboard
streamlit run app.py
```

O dashboard abrirá automaticamente em: http://localhost:8501

### Páginas do Dashboard

| Página | Conteúdo |
|--------|----------|
| 📊 Visão Geral | KPIs, tabela de status de todos os equipamentos |
| 🩸 Hemodiálise | Análise individual por máquina, histórico de sensores |
| 💨 Respiradores | Análise individual por respirador, histórico de sensores |
| 🤖 Modelos de IA | Métricas MAE/R², importância de features, código do modelo |

### Como Usar na Apresentação

1. Execute `streamlit run app.py`
2. Na sidebar, use **"Regenerar Dados"** para mostrar simulação ao vivo
3. Navegue pelas páginas para demonstrar cada funcionalidade
4. Na página **Modelos de IA**, mostre o código e as métricas de performance

### Tecnologias

| Tecnologia | Finalidade |
|-----------|------------|
| Python 3.10+ | Linguagem base |
| Streamlit | Dashboard interativo |
| Scikit-learn | Isolation Forest, Random Forest |
| Pandas / NumPy | Manipulação de dados |
| Plotly | Gráficos interativos |

## 🔗 Acesse o Dashboard

👉 **[Abrir o app ao vivo](https://manutencao-preditiva-indyanny.streamlit.app/)**

Link direto: https://manutencao-preditiva-indyanny.streamlit.app/

### Autora

Indyanny Rodrigues Peixinho — FAESA, Vitória – ES, 2026
