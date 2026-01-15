# 📌 Cadeia de Markov aplicada a Churn (A/R/C) — Streamlit

App em **Streamlit** para modelar **churn** com **Cadeia de Markov** usando 3 estados:
- **A** = Ativo  
- **R** = Em risco  
- **C** = Churn (absorvente)

O projeto permite:
- Fazer **upload de dados** (CSV/XLSX)
- Definir regras de negócio para classificar clientes em **A/R/C**
- Estimar a **matriz de transição P** e a matriz de contagens **Nᵢⱼ**
- Calcular **probabilidade de churn em n meses** via **Pⁿ**
- Explorar **insights e validações** (backtesting, estacionaridade, calibração, etc.)
- Visualizar gráficos e métricas (heatmap, evolução da base, etc.)

---

## 🧠 Como o modelo funciona (visão rápida)

1. Você envia um dataset com:
   - **ID do cliente** (ex.: `customer_id`)
   - **Data** (ex.: `date`)
   - (Opcional) **estado pronto (A/R/C)** — se não tiver, o app cria.

2. O app agrega os dados em um painel **cliente × mês** e classifica o estado:
- **A (Ativo):** houve compra no mês
- **R (Em risco):** não comprou, mas ainda não atingiu a janela de churn
- **C (Churn):** sem compra por tempo suficiente (e permanece em C)

3. Com isso, estima a matriz:
- **Nᵢⱼ:** contagem de transições `i → j`
- **P:** probabilidades `i → j`

---

