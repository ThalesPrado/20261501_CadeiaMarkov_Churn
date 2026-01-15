import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


st.set_page_config(page_title="Cadeia de Markov (Churn)", layout="wide")
st.title("📌 Cadeia de Markov aplicada a Churn (A/R/C)")

# ============================================================
# TABS PRINCIPAIS (APP)
# ============================================================
main_tabs = st.tabs([
    "📚 Teoria",
    "📥 Dados",
    "⚙️ Modelo",
    "🧠 Gráficos e Análises",
])

# ============================================================
# ABA: TEORIA
# ============================================================
with main_tabs[0]:
    st.title("📚 Teoria — Cadeia de Markov aplicada a Churn (A/R/C)")

    st.caption(
        "Nesta aba, focamos apenas na parte teórica: definições, equações e significado de cada símbolo. "
        "Não há pipeline de dados aqui."
    )

    tabs = st.tabs([
        "1) Definições (S, X_t, t, i, j)",
        "2) Propriedade de Markov (p_ij)",
        "3) Matriz de Transição (P) e regras",
        "4) Evolução da distribuição (π_t) e P^n",
        "5) Churn absorvente e cadeia absorvente (Q, R, N)",
        "6) Mini-exemplo (intuição)"
    ])

    # ============================================================
    # TAB 1 — Definições básicas
    # ============================================================
    with tabs[0]:
        st.header("1) Definições básicas: o que é cada coisa")

        st.subheader("Espaço de estados: S")
        st.write(
            "**S** é o conjunto de estados possíveis que um cliente pode estar em um período.\n\n"
            "No nosso caso:"
        )
        st.latex(r"S=\{A, R, C\}")
        st.write(
            "- **A** = Ativo\n"
            "- **R** = Em risco\n"
            "- **C** = Churn"
        )

        st.subheader("Variável de estado no tempo: Xₜ")
        st.write(
            "**Xₜ** representa o estado do cliente no período **t**.\n\n"
            "Quando escrevemos:"
        )
        st.latex(r"X_t \in S")
        st.write(
            "Isso quer dizer: **no período t, o cliente está em algum estado dentro de S**, isto é, "
            "ou A, ou R, ou C.\n\n"
            "✅ Exemplo: se no mês de Março (t=3) o cliente está 'Em risco', então **X₃ = R**."
        )

        st.subheader("O que são t, i e j?")
        st.write(
            "- **t**: índice do tempo (ex.: mês 1, mês 2, mês 3...)\n"
            "- **i**: estado atual (de onde estou saindo)\n"
            "- **j**: próximo estado (para onde vou)\n\n"
            "Ex.: se hoje o cliente está em **i = R**, e no próximo mês vai para **j = C**, então é uma transição **R → C**."
        )

    # ============================================================
    # TAB 2 — Propriedade de Markov
    # ============================================================
    with tabs[1]:
        st.header("2) Propriedade de Markov (memória de 1 passo)")

        st.write("A propriedade de Markov diz que o **próximo estado** depende apenas do **estado atual**, não do histórico completo.")

        st.subheader("Equação")
        st.latex(
            r"\mathbb{P}(X_{t+1}=j \mid X_t=i, X_{t-1},\dots) "
            r"= \mathbb{P}(X_{t+1}=j \mid X_t=i) = p_{ij}"
        )

        st.subheader("O que significa cada símbolo?")
        st.write(
            "- **𝙋( … )** ou **ℙ( … )**: probabilidade\n"
            "- **Xₜ**: estado no período t\n"
            "- **Xₜ₊₁**: estado no próximo período\n"
            "- **|** (barra): “dado que” (condicional)\n"
            "- **i**: estado atual\n"
            "- **j**: próximo estado\n"
            "- **pᵢⱼ**: probabilidade de ir do estado **i** para o estado **j** em 1 passo\n\n"
            "✅ Exemplo: **pᵣ𝚌** é a probabilidade de um cliente em **R** virar **C** no próximo mês."
        )

        st.subheader("Homogeneidade temporal (assumida)")
        st.write("Em muitos modelos, assumimos que essa probabilidade não muda com o tempo (simplificação prática).")
        st.latex(r"p_{ij}\ \text{não depende de}\ t \quad\Rightarrow\quad P\ \text{é constante ao longo do tempo}")
        st.write(
            "📌 Em negócio, isso significa: assumimos que o comportamento médio de transição (A→R, R→C etc.) "
            "é relativamente estável no período analisado."
        )

    # ============================================================
    # TAB 3 — Matriz de transição P
    # ============================================================
    with tabs[2]:
        st.header("3) Matriz de Transição (P) e suas propriedades")

        st.write(
            "A **matriz de transição P** junta todas as probabilidades **pᵢⱼ**.\n\n"
            "Cada linha representa o estado atual (i) e cada coluna representa o próximo estado (j)."
        )

        st.subheader("Definição")
        st.latex(r"P = [p_{ij}]_{i,j\in S}")

        st.subheader("Propriedades essenciais")
        st.write("Como P é uma matriz de probabilidades, ela precisa respeitar:")
        st.latex(r"p_{ij} \ge 0")
        st.latex(r"\sum_{j\in S} p_{ij} = 1 \quad \text{(cada linha soma 1)}")

        st.write(
            "✅ Interpretação: se você está no estado i, você vai para algum estado j — então as probabilidades de saída "
            "de i precisam somar 100%."
        )

        st.subheader("Exemplo de forma de P")
        st.markdown("Uma matriz típica (A/R/C) teria a estrutura:")
        st.latex(
            r"P=\begin{pmatrix}"
            r"p_{AA} & p_{AR} & p_{AC}\\"
            r"p_{RA} & p_{RR} & p_{RC}\\"
            r"p_{CA} & p_{CR} & p_{CC}"
            r"\end{pmatrix}"
        )

    # ============================================================
    # TAB 4 — Distribuição π_t e potência P^n
    # ============================================================
    with tabs[3]:
        st.header("4) Evolução da distribuição de estados (πₜ) e Pⁿ")

        st.subheader("O que é πₜ?")
        st.write(
            "**πₜ** é um vetor que representa a **distribuição de estados** na base no tempo t.\n\n"
            "Ex.: πₜ = [0.70, 0.20, 0.10] significa: 70% Ativos, 20% Em risco, 10% Churn."
        )
        st.write("Dentro da indústria esse vetor seria calculado como a proporção de clientes em cada estado no tempo, por exemplo total de cliente ativos no tempo t dividido por total de clientes, total de clientes em risco no tempo t dividido por total de clientes e assim sucessivamente.")

        st.subheader("Como π evolui?")
        st.write("A distribuição no próximo passo é a distribuição atual multiplicada por P.")
        
        st.write("πₜ se refere como está a base de clientes hoje e P como as pessoas mudam de estado, por exemplo vamos supor que nossa base conte com os seguites valores de πₜ = [0.7,0.2,0.1] isso quer dizer nesse mês t temos 70% de clientes ativos, 20% de clientes em risco e 10% de clientes em churn, depois disso multiplicamos pela matriz de probabilidades que na indústria calculariamos como todos os clientes que estavam num estado i e quantos foram para outro estado no mês seguinte.")

        st.latex(r"\pi_t = [\mathbb{P}(X_t=A),\ \mathbb{P}(X_t=R),\ \mathbb{P}(X_t=C)]")

        st.latex(r"\hat p_{ij} = \frac{N_{ij}}{\sum_{k \in \{A,R,C\}} N_{ik}}")
        
        st.latex(r"N_{ij} = \text{número de clientes que estavam no estado } i \text{ no tempo } t \text{ e foram para } j \text{ no tempo } t+1")

        st.latex(r"N_{i\cdot} = \text{número total de clientes que estavam no estado } i \text{ no tempo } t")

        st.write(
        "Ou seja: para cada estado inicial i (A, R ou C), contamos para onde os clientes foram no mês seguinte "
        "e dividimos pelo total de clientes que estavam naquele estado."
        )

        st.latex(
        r"P = \begin{pmatrix}"
        r"\hat p_{AA} & \hat p_{AR} & \hat p_{AC} \\"
        r"\hat p_{RA} & \hat p_{RR} & \hat p_{RC} \\"
        r"\hat p_{CA} & \hat p_{CR} & \hat p_{CC}"
        r"\end{pmatrix}"
        )
        
        st.latex(r"\pi_{t+1} = \pi_t P")

        st.subheader("O que é Pⁿ?")
        st.latex(r"P^n = \underbrace{P \cdot P \cdot \ldots \cdot P}_{n \text{ vezes}}")
        
        st.latex(r"(P^n)_{i,j} = \mathbb{P}(X_{t+n}=j \mid X_t=i)")

        st.write(
            "**Pⁿ** é a matriz de transição após **n passos**.\n\n"
        )
        st.write("P diz o que acontece de um mês para o outro, P elevado a n diz o que acontece ao longo de n meses, exemplo P6 seria o comportamento acumulado em 6 meses")
        
        st.write("Num pipeline real estimamos P dos dados históricos e depois fazemos potência de matrizes.")

        st.latex(r"\pi_{t+n} = \pi_t P^n")

        st.subheader("Probabilidade de churn em n passos (a partir de um estado)")
        st.write(
        "A pergunta que queremos responder é bem direta:\n\n"
        "**“Se um cliente está hoje em um estado (ex.: Ativo), qual a chance de ele estar em Churn  daqui a n meses?”**"
        )
        st.latex(r"\mathbb{P}(X_{t+n}=C \mid X_t=A) = (P^n)_{A,C}")
        st.write(
    "✅ **Interpretação:** a entrada **linha A, coluna C** da matriz **$P^n$** "
    "é a probabilidade de um cliente que começa **Ativo** estar em **Churn** após **n meses**."
)
        st.markdown("### Por que isso funciona? (intuição de caminhos)")
        st.write(
    "O $P^n$ já considera automaticamente **todos os caminhos possíveis** que um cliente pode seguir ao longo de n meses.\n\n"
    "Exemplos de caminhos que levam ao churn:\n"
    "- A → C\n"
    "- A → R → C\n"
    "- A → A → R → C\n"
    "- A → R → A → R → C\n\n"
    "Ou seja: não é só churn direto — o modelo soma todas as maneiras possíveis de chegar em C."
        )

        st.markdown("### Exemplo numérico (com matriz P simples)")
        st.write(
    "Abaixo usamos uma matriz de exemplo (mensal) apenas para visualizar o conceito. "
    "No seu caso real, a matriz **P** vem do histórico (contagem de transições mês a mês)."
)
        # Matriz exemplo
        P_exemplo = np.array([
    [0.7, 0.2, 0.1],
    [0.3, 0.4, 0.3],
    [0.0, 0.0, 1.0]
    ])

        st.latex(
    r"P=\begin{pmatrix}"
    r"0.7 & 0.2 & 0.1\\"
    r"0.3 & 0.4 & 0.3\\"
    r"0   & 0   & 1"
    r"\end{pmatrix}"
    )

        n = st.slider("Escolha n (meses) para ver o churn a partir de A:", min_value=1, max_value=24,   value=6, step=1)

        Pn = np.linalg.matrix_power(P_exemplo, n)
        prob_churn_A_n = Pn[0, 2]  # (A,C)

        st.write("A probabilidade de churn para um cliente que começa em **A** é:")
        st.latex(rf"(P^{n})_{{A,C}} = {prob_churn_A_n:.4f}")
        st.success(f"👉 Em {n} meses: {prob_churn_A_n*100:.2f}% de chance de churn (começando em A)")

        with st.expander("Ver a matriz Pⁿ completa (exemplo)"):
            st.write("Matriz $P^n$ calculada a partir do exemplo:")
            st.code(Pn, language="text")

    # ============================================================
    # TAB 5 — Cadeia absorvente
    # ============================================================
    with tabs[4]:
        st.header("5) Churn absorvente e cadeia absorvente")

        st.subheader("O que é um estado absorvente?")
        st.write(
            "Um estado **absorvente** é um estado que, uma vez alcançado, você **não sai mais dele**.\n\n"
            "No churn, isso significa: depois que o cliente entra em C, ele permanece em C."
        )
        st.latex(r"p_{CC}=1,\quad p_{CA}=0,\quad p_{CR}=0")

        st.write("Onde Q mede como clientes ativos e em risco se comportam entre si, R mede o funil de perda, 0 os clientes churn e I clientes que permanencem churn.")

        st.subheader("Por que isso é útil?")
        st.write(
            "Porque permite calcular coisas como:\n"
            "- **tempo médio até churn**\n"
            "- **probabilidade de churn em horizonte n** de forma consistente\n\n"
            "E isso casa bem com a definição de churn como “perda final” (não apenas uma pausa)."
        )

        st.subheader("Forma canônica (cadeia absorvente)")
        st.write(
            "Separamos os estados em:\n"
            "- **Transitórios** (podem mudar): T = {A, R}\n"
            "- **Absorvente**: {C}\n\n"
            "A matriz pode ser reorganizada como:"
        )

        st.latex(
            r"P=\begin{pmatrix}"
            r"Q & R\\"
            r"0 & I"
            r"\end{pmatrix}"
        )

        st.subheader("O que são Q, R, 0 e I?")
        st.write(
            "- **Q**: submatriz de transição **entre estados transitórios** (A e R)\n"
            "- **R**: submatriz de transição **dos transitórios para absorventes** (A/R → C)\n"
            "- **0**: bloco de zeros (absorvente não volta para transitório)\n"
            "- **I**: matriz identidade (absorvente permanece nele mesmo)\n\n"
            "📌 No nosso caso, Q é 2×2 e I é 1×1."
        )

        st.subheader("Matriz Fundamental N (não confundir com contagens N_ij)")
        st.write(
            " A matriz fundamental **N** mede **quanto tempo, em média, os clientes permanecem na base antes de churnar** "
            "e **como eles circulam entre os estados vivos (A e R)**."
        )
        st.latex(r"N = (I - Q)^{-1}")

        st.write(
        "Por exemplo:\n"
        "- **N₍A,A₎** = quantos meses, em média, um cliente que começa Ativo passa como Ativo antes de churnar\n"
        "- **N₍A,R₎** = quantos meses, em média, esse cliente passa Em Risco antes de churnar\n"
        "- **N₍R,R₎** = quanto tempo um cliente já em risco tende a continuar em risco antes de churnar"
        )


        st.subheader("Tempo esperado até churn")
        st.write("Aqui temo essa formula dizendo basicamente que quantos meses, em média, um cliente que começa no estado i vai continuar na base antes de churnar.")
        st.latex(r"\mathbb{E}[T\mid X_0=i] = \sum_j N_{ij} \quad \text{(para } i\in\{A,R\}\text{)}")

    # ============================================================
    # TAB 6 — Mini exemplo + P^n
    # ============================================================
    with tabs[5]:
        st.header("6) Mini-exemplo (intuição)")

        st.write(
            "Imagine um cliente com periodicidade mensal. A cada mês ele pode estar em A, R ou C.\n\n"
            "Exemplo de sequência de estados (um cliente):"
        )
        st.latex(r"X_1=A,\ X_2=A,\ X_3=R,\ X_4=R,\ X_5=C")

        st.write(
            "Isso significa:\n"
            "- Mês 1: Ativo\n"
            "- Mês 2: Ativo\n"
            "- Mês 3: Em risco\n"
            "- Mês 4: Em risco\n"
            "- Mês 5: Churn\n\n"
            "As transições observadas seriam:\n"
            "- A→A\n"
            "- A→R\n"
            "- R→R\n"
            "- R→C"
        )

        st.write(
            "Quando você faz isso para **todos os clientes**, você consegue:\n"
            "1) contar quantas vezes ocorre cada transição (contagens **N_ij**)\n"
            "2) estimar probabilidades (matriz **P**) por frequência relativa:\n"
        )
        st.latex(r"\hat{p}_{ij}=\frac{N_{ij}}{\sum_k N_{ik}}")

        st.write(
            "✅ A partir de P, você calcula churn em horizontes (P³, P⁶, P¹² etc.), "
            "e projeta a evolução da base (π_t)."
        )

        st.divider()
        st.subheader("📌 Exemplo numérico: probabilidade de churn em n passos (começando em A)")

        st.write(
            "Agora vamos ver um exemplo com números para entender como aparece o termo "
            r"\((P^n)_{A,C}\) na prática."
        )

        st.markdown("### Matriz de transição (exemplo)")
        st.latex(
            r"P=\begin{pmatrix}"
            r"0.7 & 0.2 & 0.1\\"
            r"0.3 & 0.4 & 0.3\\"
            r"0   & 0   & 1"
            r"\end{pmatrix}"
        )

        st.write(
            "- Se está **Ativo (A)**: 70% fica A, 20% vai para R, 10% vai para C\n"
            "- Se está **Em risco (R)**: 30% volta para A, 40% fica R, 30% vai para C\n"
            "- Se está em **Churn (C)**: fica em C (absorvente)"
        )

        P = np.array([
            [0.7, 0.2, 0.1],
            [0.3, 0.4, 0.3],
            [0.0, 0.0, 1.0]
        ])

        st.markdown("### Churn em 1 passo (n = 1)")
        st.latex(r"\mathbb{P}(X_{t+1}=C\mid X_t=A)=P_{A,C}=0.10")
        st.success("👉 Em 1 mês: 10% de chance de churn (começando em A).")

        st.markdown("### Churn em 2 passos (n = 2): somando caminhos")
        st.write("Em 2 meses, existem 3 caminhos principais para terminar em C:")

        st.markdown("**1) A → A → C**")
        st.latex(r"0.7 \times 0.1 = 0.07")

        st.markdown("**2) A → R → C**")
        st.latex(r"0.2 \times 0.3 = 0.06")

        st.markdown("**3) A → C → C** (churn e permanece churn)")
        st.latex(r"0.1 \times 1 = 0.10")

        st.markdown("**Somando:**")
        st.latex(r"(P^2)_{A,C}=0.07+0.06+0.10=0.23")
        st.success("👉 Em 2 meses: 23% de chance de churn (começando em A).")

        st.markdown("### Conferindo via multiplicação de matrizes: P² = P·P")
        P2 = P @ P
        st.write("O elemento **linha A, coluna C** em \(P^2\) confirma o resultado:")
        st.latex(r"(P^2)_{A,C}=0.23")
        st.code(P2, language="text")

        st.markdown("### Escolha n e veja a probabilidade automaticamente")
        n = st.slider("Escolha n (meses):", min_value=1, max_value=24, value=6, step=1)
        Pn = np.linalg.matrix_power(P, n)
        prob_churn_n = Pn[0, 2]  # A,C

        st.write(f"Probabilidade de churn em **{n}** meses, começando em **A**:")
        st.latex(rf"(P^{n})_{{A,C}} = {prob_churn_n:.4f}")
        st.success(f"👉 Churn em {n} meses (começando em A): {prob_churn_n*100:.2f}%")


# ============================
# ABA: 📥 DADOS (APENAS)
# ============================
# ✅ Pré-requisito: no topo do seu arquivo tenha:
# import pandas as pd

with main_tabs[1]:
    st.header("📥 Dados — Upload, validações e mapeamento")

    st.write(
        "Aqui você faz o **upload** do arquivo e informa as colunas mínimas para o app funcionar.\n\n"
        "**Mínimo obrigatório:**\n"
        "- **ID do cliente** (ex.: `customer_id`)\n"
        "- **Data** (ex.: `date`) — vamos converter para **mês**\n\n"
        "**Opcional:**\n"
        "- **Estado (A/R/C)** já pronto. Se não tiver, vamos criar na aba ⚙️ Modelo (com regras)."
    )

    uploaded_file = st.file_uploader("Envie seu arquivo (CSV ou XLSX)", type=["csv", "xlsx", "xls"])

    if uploaded_file is None:
        st.info("Envie um arquivo para continuar.")
        st.stop()

    # ----------------------------
    # 1) Carregamento
    # ----------------------------
    try:
        file_name = uploaded_file.name.lower()
        if file_name.endswith(".csv"):
            df_raw = pd.read_csv(uploaded_file)
        elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
            df_raw = pd.read_excel(uploaded_file)
        else:
            st.error("Formato inválido. Envie CSV ou XLSX.")
            st.stop()

        st.success(f"Arquivo carregado: **{df_raw.shape[0]:,} linhas** × **{df_raw.shape[1]} colunas**")
        st.dataframe(df_raw.head(25), use_container_width=True)

    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {e}")
        st.stop()

    # ----------------------------
    # 1.1) Coluna de Revenue (se possível)
    # ----------------------------
    st.subheader("0) Enriquecimento: Revenue (Price × Quantity)")

    if ("Price" in df_raw.columns) and ("Quantity" in df_raw.columns):
        df_raw["revenue"] = df_raw["Price"] * df_raw["Quantity"]
        st.success("Coluna **revenue** criada com sucesso: `revenue = Price × Quantity`.")
        with st.expander("Ver amostra de revenue"):
            st.dataframe(df_raw[["Price", "Quantity", "revenue"]].head(10), use_container_width=True)
    else:
        st.warning("Não foi possível criar **revenue** automaticamente (colunas `Price` e/ou `Quantity` não encontradas).")

    st.divider()

    # ----------------------------
    # 2) Diagnóstico rápido
    # ----------------------------
    st.subheader("1) Diagnóstico rápido (qualidade dos dados)")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Linhas", f"{df_raw.shape[0]:,}")
    with c2:
        st.metric("Colunas", f"{df_raw.shape[1]:,}")
    with c3:
        st.metric("Nulos totais", f"{int(df_raw.isna().sum().sum()):,}")

    with st.expander("Ver nulos por coluna"):
        nulls = df_raw.isna().sum().sort_values(ascending=False)
        st.dataframe(nulls.to_frame("nulos"), use_container_width=True)

    st.divider()

    # ----------------------------
    # 3) Mapeamento de colunas
    # ----------------------------
    st.subheader("2) Mapeamento de colunas (o que cada coluna significa)")

    cols = df_raw.columns.tolist()

    # sugestões automáticas
    customer_guess = next((c for c in cols if c.lower() in ["customer_id", "customer id", "customer", "userid", "user_id", "id_cliente", "cliente"]), cols[0])
    date_guess = next((c for c in cols if "date" in c.lower() or "data" in c.lower() or "month" in c.lower() or "mes" in c.lower()), cols[0])
    state_guess = next((c for c in cols if c.lower() in ["state", "estado", "status", "markov_state", "status_markov"]), None)

    customer_col = st.selectbox("Coluna de cliente (ID)", options=cols, index=cols.index(customer_guess))
    date_col = st.selectbox("Coluna de data (evento/mês)", options=cols, index=cols.index(date_guess))

    has_state = st.checkbox("Meu dataset já tem uma coluna de estado (A/R/C)", value=(state_guess is not None))
    state_col = None
    if has_state:
        state_col = st.selectbox("Coluna de estado (A/R/C)", options=cols, index=cols.index(state_guess) if state_guess else 0)

    st.divider()

    # ----------------------------
    # 4) Validações mínimas
    # ----------------------------
    st.subheader("3) Validações mínimas (para o Markov funcionar)")

    problems = []

    # Cliente: nulos
    null_rate_customer = df_raw[customer_col].isna().mean()
    if null_rate_customer > 0.05:
        problems.append(f"Mais de 5% de valores nulos na coluna de cliente `{customer_col}` (≈ {null_rate_customer:.0%}).")

    # Data: parse
    date_parsed = pd.to_datetime(df_raw[date_col], errors="coerce")
    parse_fail = date_parsed.isna().mean()
    if parse_fail > 0.20:
        problems.append(f"A coluna `{date_col}` tem muita falha de parse (≈ {parse_fail:.0%}). Ajuste o formato da data.")

    # Estado: valores válidos
    if has_state:
        valid_states = {"A", "R", "C"}
        states_norm = df_raw[state_col].astype(str).str.strip().str.upper()
        invalid_rate = (~states_norm.isin(valid_states)).mean()
        if invalid_rate > 0.10:
            problems.append(
                f"A coluna `{state_col}` tem muitos valores fora de A/R/C (≈ {invalid_rate:.0%}). "
                "Padronize para 'A', 'R' e 'C'."
            )

    if problems:
        st.warning("⚠️ Encontramos pontos para revisar antes de seguir:")
        for p in problems:
            st.write(f"- {p}")
    else:
        st.success("Validações mínimas OK ✅")

    st.divider()

    # ----------------------------
    # 5) Remoção de Customer ID nulo (opcional)
    # ----------------------------
    st.subheader("4) Tratamento de clientes sem ID")

    null_customers = df_raw[customer_col].isna().sum()
    null_pct = df_raw[customer_col].isna().mean()

    if null_customers > 0:
        st.warning(f"Existem **{null_customers:,} linhas ({null_pct:.1%})** sem Customer ID.")
        drop_nulls = st.checkbox("Excluir linhas sem Customer ID (recomendado)", value=True)

        if drop_nulls:
            df_filtered = df_raw.dropna(subset=[customer_col])
            st.success(f"{null_customers:,} linhas removidas. Base agora tem {df_filtered.shape[0]:,} linhas.")
        else:
            df_filtered = df_raw.copy()
            st.warning("⚠️ Manter clientes sem ID pode inviabilizar o Markov (não dá para montar transições por cliente).")
    else:
        df_filtered = df_raw.copy()
        st.success("Nenhum Customer ID nulo encontrado.")

    st.divider()

    # ----------------------------
    # 6) Preparação final (month como inteiro) + salvar em session_state
    # ----------------------------
    st.subheader("5) Preparar dados (month como número) e salvar para a próxima aba")

    df = df_filtered.copy()

    # padronizações leves
    df[customer_col] = df[customer_col].astype(str).str.strip()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # remove linhas inválidas
    df = df.dropna(subset=[customer_col, date_col])

    # month como número (1-12)
    df["month"] = df[date_col].dt.month

    if has_state:
        df[state_col] = df[state_col].astype(str).str.strip().str.upper()

    # salva para as próximas abas
    st.session_state["df_raw"] = df_raw
    st.session_state["df"] = df
    st.session_state["data_config"] = {
        "customer_col": customer_col,
        "date_col": date_col,
        "month_col": "month",
        "has_state": has_state,
        "state_col": state_col,
        "has_revenue": "revenue" in df.columns
    }

    st.success("✅ Dados carregados e configurados! Agora você pode seguir para a aba ⚙️ Modelo.")
    with st.expander("Ver amostra do dataset preparado"):
        st.dataframe(df.head(25), use_container_width=True)

# ============================
# ABA: ⚙️ MODELO (Markov de Churn) — COMPLETA (com explicação de "Distribuição geral de estados no painel")
# ✅ O que foi ajustado aqui:
# - Adicionei uma explicação clara (negócio) antes da tabela de distribuição A/R/C
# - Expliquei exatamente o que é "contagem" e "proporção" (não é contagem de clientes únicos)
# - Mantive o restante igual ao seu bloco mais recente (com keys únicos, revenue/total_purchases)
# ============================
# Pré-requisitos no topo do arquivo:
# import pandas as pd
# import numpy as np
# import streamlit as st

with main_tabs[2]:
# ============================
# ABA: ⚙️ MODELO — Cadeia de Markov (Churn A/R/C) + VALIDAÇÕES
# ============================
    st.header("⚙️ Modelo — Cadeia de Markov para Churn (A/R/C)")

    # ----------------------------
    # 0) Checar se os dados existem
    # ----------------------------
    if "df" not in st.session_state or "data_config" not in st.session_state:
        st.warning("Primeiro carregue e prepare os dados na aba 📥 Dados.")
        st.stop()

    df = st.session_state["df"].copy()
    cfg = st.session_state["data_config"]

    customer_col = cfg["customer_col"]
    date_col = cfg["date_col"]
    month_col = cfg.get("month_col", "month")
    has_revenue = "revenue" in df.columns

    st.caption(
        "Objetivo: transformar o histórico transacional em uma sequência mensal por cliente, "
        "classificar estados A/R/C e estimar a matriz de transição P."
    )

    # ----------------------------
    # 1) Regras de negócio (A/R/C)
    # ----------------------------
    st.subheader("1) Regras de negócio para definir A/R/C")

    colA, colB, colC = st.columns(3)

    with colA:
        risk_gap_months = st.number_input(
            "Gap para entrar/manter em R (meses sem compra)",
            min_value=1, max_value=12, value=1, step=1,
            help="Se o cliente ficar >= este número de meses sem compra, ele fica em R (Em risco).",
            key="model_risk_gap_months"
        )

    with colB:
        churn_gap_months = st.number_input(
            "Gap para entrar/manter em C (meses sem compra)",
            min_value=2, max_value=24, value=3, step=1,
            help="Se o cliente ficar >= este número de meses sem compra, ele entra em C (Churn).",
            key="model_churn_gap_months"
        )

    with colC:
        use_revenue = st.checkbox(
            "Usar revenue como métrica (Price×Quantity) para identificar compra no mês",
            value=has_revenue,
            help="Se desmarcado, usa contagem de compras/linhas no mês como proxy.",
            key="model_use_revenue"
        )

    if churn_gap_months <= risk_gap_months:
        st.error("O gap de churn (C) precisa ser maior que o gap de risco (R). Ajuste os valores.")
        st.stop()

    st.markdown("### ✅ Regras (bem objetivas)")
    st.write(
        "Definimos um cliente mês a mês:\n"
        "- **A (Ativo)**: houve compra no mês\n"
        "- **R (Em risco)**: não comprou, mas ainda está dentro da janela de churn\n"
        "- **C (Churn)**: não compra há tempo suficiente para ser considerado perdido\n\n"
        "**Como aplicamos isso:**\n"
        "- Calculamos **meses desde a última compra** para cada cliente.\n"
        f"- Se **comprou no mês** → estado **A**.\n"
        f"- Se **não comprou** e gap **≥ {risk_gap_months}** e **< {churn_gap_months}** → estado **R**.\n"
        f"- Se **não comprou** e gap **≥ {churn_gap_months}** → estado **C**.\n"
        "- Depois que entra em **C**, fica em **C** (absorvente)."
    )

    st.latex(r"""
    X_{t}=
    \begin{cases}
    A, & \text{se há compra no mês } t\\
    R, & \text{se não há compra e } g_t \in [r,\ c)\\
    C, & \text{se não há compra e } g_t \ge c
    \end{cases}
    """)
    st.latex(r"g_t=\text{meses desde a última compra},\quad r=\text{risk\_gap},\quad c=\text{churn\_gap}")

    st.divider()

    # ----------------------------
    # 2) Agregação mensal (cliente × mês)
    # ----------------------------
    st.subheader("2) Construção da base mensal (cliente × mês)")

    # Tratamento de revenue negativo (devoluções) — comum no Online Retail
    if use_revenue and "revenue" in df.columns:
        st.markdown("**Tratamento de revenue negativo (devoluções/estornos)**")
        neg_mode = st.selectbox(
            "Como tratar revenue < 0?",
            ["Manter (recomendado para análise financeira)", "Zerar negativos (não considerar devolução)", "Remover linhas negativas (limpar devoluções)"],
            index=0,
            key="model_neg_revenue_mode"
        )

        if neg_mode == "Zerar negativos (não considerar devolução)":
            df["revenue"] = df["revenue"].clip(lower=0)
        elif neg_mode == "Remover linhas negativas (limpar devoluções)":
            df = df[df["revenue"] >= 0].copy()

        metric_col = "revenue"
    else:
        metric_col = "_events"
        df[metric_col] = 1

    # mês real (YYYY-MM) e índice sequencial ano+mês
    df["_month_ts"] = pd.to_datetime(df[date_col], errors="coerce").dt.to_period("M").dt.to_timestamp()
    df = df.dropna(subset=[customer_col, "_month_ts"]).copy()
    df["_month_index"] = df["_month_ts"].dt.year * 12 + df["_month_ts"].dt.month

    # Agregação: 1 linha por cliente-mês
    agg = (
        df.groupby([customer_col, "_month_index"], as_index=False)
          .agg(
              month_ts=("_month_ts", "min"),
              revenue=(metric_col, "sum"),
              total_purchases=(metric_col, "size")
          )
    )

    st.write("Abaixo está a agregação mensal (um registro por cliente por mês).")
    st.dataframe(agg.head(20), use_container_width=True)

    st.info("Nota: usamos um índice sequencial **ano+mês** para não misturar Janeiro/2010 com Janeiro/2011.")

    st.divider()

    # ----------------------------
    # 3) Completar meses sem compra (painel mensal completo)
    # ----------------------------
    st.subheader("3) Completar meses sem compra (painel mensal completo)")

    min_m = int(agg["_month_index"].min())
    max_m = int(agg["_month_index"].max())
    st.write(f"Período detectado: **{agg['month_ts'].min().date()}** até **{agg['month_ts'].max().date()}**")

    customers = agg[customer_col].dropna().astype(str).unique()
    all_months = np.arange(min_m, max_m + 1, dtype=int)

    st.subheader("Opcional: amostragem (para performance)")
    sample_mode = st.checkbox("Rodar em amostra de clientes (para testar mais rápido)", value=False, key="model_sample_mode")
    sample_n = None
    if sample_mode:
        sample_n = st.number_input("Qtd. clientes na amostra", min_value=100, max_value=200000, value=5000, step=100, key="model_sample_n")

    if sample_mode:
        rng = np.random.default_rng(42)
        customers_used = rng.choice(customers, size=min(int(sample_n), len(customers)), replace=False)
        st.warning(f"Rodando com amostra de **{len(customers_used):,}** clientes (de {len(customers):,}).")
    else:
        customers_used = customers
        st.success(f"Rodando com **todos os clientes**: {len(customers_used):,}")

    panel = pd.MultiIndex.from_product([customers_used, all_months], names=[customer_col, "_month_index"]).to_frame(index=False)

    panel = panel.merge(
        agg[[customer_col, "_month_index", "month_ts", "revenue", "total_purchases"]],
        on=[customer_col, "_month_index"],
        how="left"
    )

    # reconstruir month_ts onde faltante
    year = panel["_month_index"] // 12
    month = panel["_month_index"] % 12
    month = month.replace(0, 12)
    panel["month_ts"] = panel["month_ts"].fillna(pd.to_datetime(dict(year=year, month=month, day=1)))

    panel["revenue"] = panel["revenue"].fillna(0.0)
    panel["total_purchases"] = panel["total_purchases"].fillna(0).astype(int)

    st.write("Painel completo (inclui meses sem compra com revenue=0 e total_purchases=0):")
    st.dataframe(panel.head(20), use_container_width=True)

    st.divider()

    # ----------------------------
    # 4) Definir estados A/R/C
    # ----------------------------
    st.subheader("4) Definição de estado por mês (A/R/C)")

    panel = panel.sort_values([customer_col, "_month_index"]).reset_index(drop=True)

    # compra no mês?
    if use_revenue:
        panel["_had_purchase"] = panel["revenue"] > 0
    else:
        panel["_had_purchase"] = panel["total_purchases"] > 0

    # começar o "relógio" do cliente no primeiro mês em que ele aparece comprando
    first_purchase = (
        panel[panel["_had_purchase"]]
        .groupby(customer_col)["_month_index"].min()
        .rename("_first_purchase_month")
        .reset_index()
    )
    panel = panel.merge(first_purchase, on=customer_col, how="left")
    panel = panel[panel["_month_index"] >= panel["_first_purchase_month"]].copy()

    # meses desde a última compra
    panel["_last_purchase_month"] = np.where(panel["_had_purchase"], panel["_month_index"], np.nan)
    panel["_last_purchase_month"] = panel.groupby(customer_col)["_last_purchase_month"].ffill()
    panel["_months_since_purchase"] = panel["_month_index"] - panel["_last_purchase_month"]

    # estado
    panel["state"] = "R"
    panel.loc[panel["_had_purchase"], "state"] = "A"
    panel.loc[(~panel["_had_purchase"]) & (panel["_months_since_purchase"] >= risk_gap_months), "state"] = "R"
    panel.loc[(~panel["_had_purchase"]) & (panel["_months_since_purchase"] >= churn_gap_months), "state"] = "C"

    # churn absorvente: depois de C, sempre C
    panel["_ever_churned"] = panel.groupby(customer_col)["state"].transform(lambda s: (s == "C").cummax())
    panel.loc[panel["_ever_churned"], "state"] = "C"

    # explicação business do que é essa tabela
    st.markdown("### 📌 O que essa distribuição significa? (para negócios)")
    st.write(
        "Aqui nós olhamos o painel **cliente-mês**.\n\n"
        "- **contagem** = número de linhas cliente-mês em cada estado (não é cliente único)\n"
        "- **proporção** = contagem do estado ÷ total de linhas cliente-mês\n\n"
        "Isso responde: **em média, ao longo do tempo, como a base está se comportando?**\n"
        "Cliente pode aparecer em estados diferentes em meses diferentes (A hoje, R amanhã, C depois)."
    )

    dist = panel["state"].value_counts(normalize=True).rename("proporção").to_frame()
    dist["contagem"] = panel["state"].value_counts()
    st.write("Distribuição geral de estados no painel (cliente-mês):")
    st.dataframe(dist, use_container_width=True)

    with st.expander("Ver amostra com colunas de diagnóstico"):
        st.dataframe(
            panel[[customer_col, "month_ts", "_month_index", "revenue", "total_purchases", "_months_since_purchase", "state"]].head(50),
            use_container_width=True
        )

    st.divider()

    # ----------------------------
    # 5) Transições e matriz N_ij
    # ----------------------------
    st.subheader("5) Transições mensais e matriz de contagens Nᵢⱼ")

    panel["next_state"] = panel.groupby(customer_col)["state"].shift(-1)
    trans = panel.dropna(subset=["next_state"]).copy()

    states = ["A", "R", "C"]
    trans["state"] = pd.Categorical(trans["state"], categories=states, ordered=True)
    trans["next_state"] = pd.Categorical(trans["next_state"], categories=states, ordered=True)

    Nij = (
        trans.groupby(["state", "next_state"])
             .size()
             .unstack(fill_value=0)
             .reindex(index=states, columns=states, fill_value=0)
    )

    st.write("Matriz de contagens **Nᵢⱼ** (quantas transições i→j observamos):")
    st.dataframe(Nij, use_container_width=True)

    st.divider()

    # ----------------------------
    # 6) Estimar matriz P
    # ----------------------------
    st.subheader("6) Estimação da matriz de transição P")

    row_sums = Nij.sum(axis=1).replace(0, np.nan)
    P = Nij.div(row_sums, axis=0).fillna(0.0)

    force_absorb = st.checkbox("Forçar churn como absorvente (C→C = 1)", value=True, key="model_force_absorb")
    if force_absorb:
        P.loc["C", :] = 0.0
        P.loc["C", "C"] = 1.0

    st.write("Matriz **P** (probabilidades i→j):")
    st.dataframe(P.style.format("{:.4f}"), use_container_width=True)

    # Salvar para outras abas
    st.session_state["panel_monthly"] = panel
    st.session_state["Nij"] = Nij
    st.session_state["P"] = P
    st.session_state["states"] = states
    st.session_state["model_params"] = {
        "risk_gap_months": int(risk_gap_months),
        "churn_gap_months": int(churn_gap_months),
        "use_revenue": bool(use_revenue),
        "metric_used": "revenue" if use_revenue else "total_purchases"
    }

    st.success("✅ Modelo estimado! Matrizes Nᵢⱼ e P salvas para a aba 📈 Gráficos.")
    st.divider()

    # ----------------------------
    # 7) Preview: churn em n meses via P^n (COM KEY ÚNICO)
    # ----------------------------
    st.subheader("7) Preview: churn em n meses via Pⁿ (rápido)")

    n_preview = st.slider(
        "Escolha n (meses) para calcular P(churn em n meses)",
        min_value=1, max_value=60, value=12, step=1,
        key="model_preview_n_slider"  # <- evita StreamlitDuplicateElementId
    )

    P_np = P.to_numpy(dtype=float)
    Pn = np.linalg.matrix_power(P_np, int(n_preview))
    idx = {s: i for i, s in enumerate(states)}

    prob_A = Pn[idx["A"], idx["C"]]
    prob_R = Pn[idx["R"], idx["C"]]

    st.write(f"Probabilidade de estar em **Churn (C)** em **{n_preview} meses**:")
    st.success(f"Começando em **A**: {prob_A*100:.2f}%")
    st.success(f"Começando em **R**: {prob_R*100:.2f}%")

    # ============================================================
    # 8) Validação do modelo (Markov/memória, backtesting, estabilidade, log-loss)
    # ============================================================
    st.divider()
    st.header("✅ Validação do Modelo (qualidade e confiabilidade)")

    import matplotlib.pyplot as plt

    # ----------------------------
    # Helpers
    # ----------------------------
    def build_P_from_panel(panel_df: pd.DataFrame, states=("A","R","C"), force_absorb=True) -> pd.DataFrame:
        dfv = panel_df.copy()
        dfv["next_state"] = dfv.groupby(customer_col)["state"].shift(-1)
        transv = dfv.dropna(subset=["next_state"]).copy()

        transv["state"] = pd.Categorical(transv["state"], categories=list(states), ordered=True)
        transv["next_state"] = pd.Categorical(transv["next_state"], categories=list(states), ordered=True)

        Nijv = (
            transv.groupby(["state", "next_state"]).size()
                 .unstack(fill_value=0)
                 .reindex(index=states, columns=states, fill_value=0)
        )
        row_sums_v = Nijv.sum(axis=1).replace(0, np.nan)
        Pv = Nijv.div(row_sums_v, axis=0).fillna(0.0)

        if force_absorb and "C" in Pv.index and "C" in Pv.columns:
            Pv.loc["C", :] = 0.0
            Pv.loc["C", "C"] = 1.0
        return Pv

    def month_dist(panel_df: pd.DataFrame, month_value, states=("A","R","C")) -> np.ndarray:
        d = (
            panel_df[panel_df["month_ts"] == month_value]["state"]
            .value_counts(normalize=True)
            .reindex(states).fillna(0.0)
        )
        return d.to_numpy()

    def mae(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean(np.abs(a - b)))

    def log_loss(y_true: np.ndarray, p_pred: np.ndarray, eps: float = 1e-15) -> float:
        p = np.clip(p_pred, eps, 1 - eps)
        return float(-(y_true*np.log(p) + (1-y_true)*np.log(1-p)).mean())

    def brier_score(y_true: np.ndarray, p_pred: np.ndarray) -> float:
        return float(np.mean((p_pred - y_true)**2))

    def confusion_counts(y_true: np.ndarray, p_pred: np.ndarray, thr: float):
        y_hat = (p_pred >= thr).astype(int)
        tp = int(((y_hat==1) & (y_true==1)).sum())
        fp = int(((y_hat==1) & (y_true==0)).sum())
        tn = int(((y_hat==0) & (y_true==0)).sum())
        fn = int(((y_hat==0) & (y_true==1)).sum())
        return tp, fp, tn, fn

    def l1_matrix_norm(Pa: pd.DataFrame, Pb: pd.DataFrame) -> float:
        A = Pa.to_numpy(dtype=float)
        B = Pb.to_numpy(dtype=float)
        return float(np.mean(np.abs(A - B)))

    # padronizar month_ts
    panel_val = panel.copy()
    panel_val["month_ts"] = pd.to_datetime(panel_val["month_ts"]).dt.to_period("M").dt.to_timestamp()
    panel_val = panel_val.sort_values([customer_col, "month_ts"]).reset_index(drop=True)
    states_tuple = tuple(states)

    # ----------------------------
    # 8.1 Backtesting (Out-of-Time)
    # ----------------------------
    st.subheader("1) Backtesting (Out-of-Time) — previsão de distribuição do mês seguinte")
    st.info(
        "✅ **Pergunta de negócio:** o modelo consegue prever como a base vai se distribuir (A/R/C) no próximo mês?\n\n"
        "Como fazemos:\n"
        "- Treinamos P em uma janela de meses\n"
        "- Aplicamos em um mês base π(t)\n"
        "- Prevemos π(t+1)=π(t)·P e comparamos com o real"
    )

    months = np.sort(panel_val["month_ts"].unique())
    if len(months) < 4:
        st.warning("Poucos meses no painel para backtesting. Ideal: 4+ meses.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            train_start = st.selectbox("Mês inicial (treino)", options=list(months), index=0, key="bt_train_start")
        with c2:
            train_end = st.selectbox("Mês final (treino)", options=list(months), index=min(2, len(months)-2), key="bt_train_end")
        with c3:
            apply_month = st.selectbox("Mês base π(t) (aplicar P)", options=list(months), index=min(3, len(months)-2), key="bt_apply_month")

        month_to_idx = {m:i for i,m in enumerate(months)}
        apply_idx = month_to_idx[apply_month]
        target_month = months[apply_idx + 1] if apply_idx + 1 < len(months) else None

        if target_month is None:
            st.warning("Não há mês seguinte para comparar.")
        else:
            panel_train = panel_val[(panel_val["month_ts"] >= train_start) & (panel_val["month_ts"] <= train_end)].copy()
            P_bt = build_P_from_panel(panel_train, states=states_tuple, force_absorb=True)

            pi_apply = month_dist(panel_val, apply_month, states=states_tuple)
            pi_real = month_dist(panel_val, target_month, states=states_tuple)
            pi_pred = pi_apply @ P_bt.to_numpy(dtype=float)

            df_cmp = pd.DataFrame({
                "estado": list(states_tuple),
                "previsto (π̂)": pi_pred,
                "real (π)": pi_real,
                "abs_erro": np.abs(pi_pred - pi_real)
            })
            st.dataframe(df_cmp.style.format({"previsto (π̂)":"{:.4f}","real (π)":"{:.4f}","abs_erro":"{:.4f}"}), use_container_width=True)

            st.metric("MAE total (shares)", f"{mae(pi_pred, pi_real):.4f}")

            # gráfico
            fig, ax = plt.subplots()
            x = np.arange(len(states_tuple))
            ax.bar(x - 0.2, pi_real, width=0.4, label="Real")
            ax.bar(x + 0.2, pi_pred, width=0.4, label="Previsto")
            ax.set_xticks(x)
            ax.set_xticklabels(states_tuple)
            ax.set_ylabel("Proporção")
            ax.set_title("Backtest — distribuição real vs prevista (mês alvo)")
            ax.legend()
            st.pyplot(fig)

    st.divider()

    # ----------------------------
    # 8.2 Teste de Markov (memória) — aproximado
    # ----------------------------
    st.subheader("2) Teste da Propriedade de Markov (memória) — aproximado")
    st.info(
    "✅ **Pergunta de negócio:** o estado atual (A/R/C) é suficiente para prever o próximo passo?\n\n"
    "A hipótese de Markov diz que, para prever o futuro, **só importa o estado atual**. "
    "Este teste verifica se clientes que estão no mesmo estado **hoje** (curr), mas vieram de estados diferentes "
    "no mês anterior (prev), têm **o mesmo comportamento no mês seguinte**.\n\n"
    "➡️ Se o passado não muda o comportamento, seu modelo Markov (1ª ordem) é consistente.\n"
    "➡️ Se o passado muda muito, o sistema tem 'memória' e o modelo simples pode perder precisão."
    )

    st.write("### Como interpretar a tabela")
    st.markdown(
    "- **prev (Estado anterior):** onde o cliente estava no mês passado.\n"
    "- **curr (Estado atual):** onde o cliente está agora.\n"
    "- **div_L1_media (Divergência):** mede o quanto o comportamento do grupo (prev→curr) "
    "difere do comportamento 'médio' do estado **curr**.\n"
    "  - **próximo de 0:** ótimo (o passado quase não importa).\n"
    "  - **alto:** indica 'memória' (o passado influencia).\n"
    "- **amostra:** tamanho do grupo analisado (grupos pequenos geram divergências menos confiáveis)."
    )

    tmp = panel_val.copy()
    tmp["prev_state"] = tmp.groupby(customer_col)["state"].shift(1)
    tmp["next_state"] = tmp.groupby(customer_col)["state"].shift(-1)
    tri = tmp.dropna(subset=["prev_state", "next_state"]).copy()

    if tri.empty:
        st.warning("Não há sequência suficiente para testar memória (precisa de 3+ meses por cliente).")
    else:
        P1 = (
            tri.groupby(["state", "next_state"]).size()
            .unstack(fill_value=0)
            .reindex(index=states_tuple, columns=states_tuple, fill_value=0)
        )
        P1 = P1.div(P1.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

        g = tri.groupby(["prev_state", "state", "next_state"]).size().rename("n").reset_index()

        divergences = []
        for (prev_s, curr_s), sub in g.groupby(["prev_state", "state"]):
            vec_counts = sub.set_index("next_state")["n"].reindex(states_tuple).fillna(0.0).to_numpy()
            if vec_counts.sum() == 0:
                continue
            p2 = vec_counts / vec_counts.sum()
            p1 = P1.loc[curr_s].to_numpy(dtype=float)
            div = np.mean(np.abs(p2 - p1))
            divergences.append({"prev": prev_s, "curr": curr_s, "div_L1_media": div, "amostra": int(vec_counts.sum())})

        div_df = pd.DataFrame(divergences).sort_values("div_L1_media", ascending=False)
        st.dataframe(div_df.head(20), use_container_width=True)

    st.divider()

    # ----------------------------
    # 8.3 Estacionaridade (P muda no tempo?)
    # ----------------------------
    st.subheader("3) Estacionaridade — estabilidade temporal da matriz P")
    st.info(
    "📈 **Pergunta de negócio:** Podemos confiar que este funil representa bem o comportamento dos clientes ao longo do tempo?\n\n"
    "Aqui avaliamos se as probabilidades de um cliente avançar, ficar ou sair do funil "
    "são estáveis mês a mês.\n\n"
    "Se elas variam muito, o negócio não está em regime estável — "
    "e previsões feitas com um único modelo podem estar distorcidas."
    )

    tmp2 = panel_val.copy()
    tmp2["next_state"] = tmp2.groupby(customer_col)["state"].shift(-1)
    trans2 = tmp2.dropna(subset=["next_state"]).copy()

    if trans2.empty:
        st.warning("Não há transições suficientes para estimar P por período.")
    else:
        mats = []
        for m, dfm in trans2.groupby("month_ts"):
            Nij_m = (
                dfm.groupby(["state","next_state"]).size()
                .unstack(fill_value=0)
                .reindex(index=states_tuple, columns=states_tuple, fill_value=0)
            )
            rs = Nij_m.sum(axis=1).replace(0, np.nan)
            Pm = Nij_m.div(rs, axis=0).fillna(0.0)
            Pm.loc["C", :] = 0.0
            Pm.loc["C", "C"] = 1.0
            mats.append((m, Pm))

        diffs = [{"month": m, "L1_medio_vs_global": l1_matrix_norm(P, Pm)} for m, Pm in mats]
        diff_df = pd.DataFrame(diffs).sort_values("month")
        st.dataframe(diff_df, use_container_width=True)

        fig, ax = plt.subplots()
        ax.plot(diff_df["month"], diff_df["L1_medio_vs_global"])
        ax.set_title("Instabilidade temporal — diferença média (L1) de P por mês vs P global")
        ax.set_xlabel("Mês")
        ax.set_ylabel("Diferença média (quanto maior, menos estável)")
        st.pyplot(fig)

    st.divider()

    # ----------------------------
    # 8.4 Calibração: Confusion Matrix + Log-Loss
    # ----------------------------
    # ----------------------------

    st.subheader("4) Calibração probabilística — Matriz de Confusão e Log-Loss")
    st.info(
    "🎯 **Pergunta de negócio:** Quando o modelo diz que um cliente tem alto risco de churn, "
    "isso realmente se confirma?\n\n"
    "Aqui avaliamos se as probabilidades geradas pelo modelo são **confiáveis para tomada de decisão**.\n\n"
    "• **Matriz de confusão**: mostra quantos clientes o modelo manda para ação (ex.: retenção) "
    "e quantos realmente cancelam.\n"
    "• **Log-Loss**: mede o quão boas são as probabilidades — "
    "quanto menor, mais podemos confiar no número (ex.: 70% realmente significa ~70%).\n\n"
    "Se a calibração for ruim, o modelo até pode acertar quem cancela, "
    "mas errará no **quanto** devemos investir para reter cada cliente."
    "O Brier Score mede o quão erradas estão as probabilidades que o modelo fornece."
    )

    eval_df = panel_val.copy()
    eval_df["next_state"] = eval_df.groupby(customer_col)["state"].shift(-1)
    eval_df = eval_df.dropna(subset=["next_state"]).copy()

    # prob churn próxima etapa = P[estado_atual, C]
    eval_df["p_churn_next"] = eval_df["state"].map(P["C"].to_dict())
    eval_df["y_churn_next"] = (eval_df["next_state"] == "C").astype(int)

    y_true = eval_df["y_churn_next"].to_numpy(dtype=int)
    p_pred = eval_df["p_churn_next"].to_numpy(dtype=float)

    thr = st.slider(
        "Threshold para classificar churn (ex.: 0.5)",
        min_value=0.05, max_value=0.95, value=0.50, step=0.05,
        key="tab_model_validation_confusion_threshold"  # <- key única
    )

    def confusion_counts(y_true: np.ndarray, p_pred: np.ndarray, thr: float):
        y_hat = (p_pred >= thr).astype(int)
        tp = int(((y_hat==1) & (y_true==1)).sum())
        fp = int(((y_hat==1) & (y_true==0)).sum())
        tn = int(((y_hat==0) & (y_true==0)).sum())
        fn = int(((y_hat==0) & (y_true==1)).sum())
        return tp, fp, tn, fn

    def log_loss(y_true: np.ndarray, p_pred: np.ndarray, eps: float = 1e-15) -> float:
        p = np.clip(p_pred, eps, 1 - eps)
        return float(-(y_true*np.log(p) + (1-y_true)*np.log(1-p)).mean())

    def brier_score(y_true: np.ndarray, p_pred: np.ndarray) -> float:
        return float(np.mean((p_pred - y_true)**2))

    tp, fp, tn, fn = confusion_counts(y_true, p_pred, thr)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

    ll = log_loss(y_true, p_pred)
    bs = brier_score(y_true, p_pred)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc*100:.2f}%")
    c2.metric("Precision", f"{precision*100:.2f}%")
    c3.metric("Recall", f"{recall*100:.2f}%")
    c4.metric("Log-Loss", f"{ll:.4f}")

    st.caption(f"Brier Score (menor é melhor): {bs:.4f}")

    cm = pd.DataFrame(
        [[tn, fp],
        [fn, tp]],
        index=["Real: Não Churn", "Real: Churn"],
        columns=["Prev: Não Churn", "Prev: Churn"]
    )
    st.dataframe(cm, use_container_width=True)

    st.write(
    "💡 **Como ler isso (executivo):**\n"
    "- **Recall alto**: você captura a maioria dos churners (bom para evitar perda, mas pode gerar falsos positivos)\n"
    "- **Precision alta**: quando você age, geralmente está certo (ações mais eficientes)\n"
    "- **Log-Loss baixo**: probabilidades confiáveis (modelo bem calibrado)"
)

# ============================
# ABA: 📈 GRÁFICOS — Análises Markov (Steady State, Tempo até Churn, LTV, P^n, etc.)
# ============================
# ✅ Pré-requisitos no topo do arquivo:
# import pandas as pd
# import numpy as np
# import streamlit as st
# import matplotlib.pyplot as plt

with main_tabs[3]:
    st.header("📈 Gráficos & Insights — Markov aplicado a Churn (A/R/C)")

    # ----------------------------
    # 0) Checar se o modelo existe
    # ----------------------------
    required = ["P", "states", "panel_monthly", "data_config", "model_params"]
    if any(k not in st.session_state for k in required):
        st.warning("Primeiro carregue os dados (📥 Dados) e rode o modelo (⚙️ Modelo).")
        st.stop()

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    P = st.session_state["P"].copy()
    states = st.session_state["states"]
    panel = st.session_state["panel_monthly"].copy()
    cfg = st.session_state["data_config"]
    params = st.session_state["model_params"]

    customer_col = cfg["customer_col"]

    # Garantir ordem
    P = P.reindex(index=states, columns=states).astype(float)
    P_np = P.to_numpy(dtype=float)

    st.caption(
        "Nesta aba você vê **o que o modelo responde para o negócio**: "
        "risco de churn por horizonte, tempo de vida, LTV, projeções e impacto de ações."
    )

    # ----------------------------
    # Helpers
    # ----------------------------
    def is_absorbing(P_df, absorb_state="C") -> bool:
        if absorb_state not in P_df.index or absorb_state not in P_df.columns:
            return False
        row = P_df.loc[absorb_state, :]
        return np.isclose(row.drop(absorb_state).sum(), 0.0) and np.isclose(row[absorb_state], 1.0)

    def safe_matrix_power(P_np, n: int):
        return np.linalg.matrix_power(P_np, int(n))

    def get_Q(P_df, transient=("A", "R")):
        return P_df.loc[list(transient), list(transient)].to_numpy(dtype=float)

    def fundamental_matrix(Q):
        I = np.eye(Q.shape[0])
        return np.linalg.inv(I - Q)

    def expected_time_to_absorption(Q):
        N = fundamental_matrix(Q)
        ones = np.ones((Q.shape[0], 1))
        t = (N @ ones).flatten()
        return t, N

    def month_state_distribution(panel_df):
        tmp = (
            panel_df.groupby(["month_ts", "state"])
                    .size()
                    .rename("count")
                    .reset_index()
        )
        total = tmp.groupby("month_ts")["count"].transform("sum")
        tmp["share"] = tmp["count"] / total
        pivot = tmp.pivot(index="month_ts", columns="state", values="share").fillna(0.0)
        pivot = pivot.reindex(columns=states, fill_value=0.0)
        return pivot

    def reward_by_state(panel_df, remove_negative=True):
        df2 = panel_df.copy()
        if "revenue" in df2.columns:
            if remove_negative:
                df2 = df2[df2["revenue"] >= 0].copy()
            # ⚠️ bom para negócio: churn por definição não gera receita
            # então forçamos reward(C)=0 para não confundir o usuário
            rewards = df2.groupby("state")["revenue"].mean().reindex(states).fillna(0.0)
            if "C" in rewards.index:
                rewards.loc["C"] = 0.0
            return rewards
        else:
            rewards = df2.groupby("state")["total_purchases"].mean().reindex(states).fillna(0.0)
            if "C" in rewards.index:
                rewards.loc["C"] = 0.0
            return rewards

    idx = {s: i for i, s in enumerate(states)}
    absorbing = is_absorbing(P, "C")

    # ============================================================
    # 1) Matriz P + Heatmap
    # ============================================================
    st.subheader("1) Matriz P — Probabilidade de mudar de estado (mês seguinte)")

    st.info(
        "✅ **Pergunta de negócio:** onde está o maior vazamento do funil?\n"
        "- **A→R** alto = clientes perdem recorrência rápido (problema de engajamento).\n"
        "- **R→A** baixo = reativação fraca.\n"
        "- **R→C** alto = churn vira inevitável sem intervenção."
    )

    st.dataframe(P.style.format("{:.4f}"), use_container_width=True)

    fig, ax = plt.subplots()
    ax.imshow(P.values, aspect="auto")
    ax.set_xticks(range(len(states)))
    ax.set_yticks(range(len(states)))
    ax.set_xticklabels(states)
    ax.set_yticklabels(states)
    ax.set_title("Heatmap — Matriz P (estado atual → próximo estado)")
    for i in range(len(states)):
        for j in range(len(states)):
            ax.text(j, i, f"{P.values[i, j]:.2f}", ha="center", va="center")
    st.pyplot(fig)

    # KPI rápido (executivo)
    c1, c2, c3 = st.columns(3)
    c1.metric("Retenção 1 mês (A→A)", f"{P.loc['A','A']*100:.2f}%" if "A" in P.index else "n/a")
    c2.metric("Deterioração 1 mês (A→R)", f"{P.loc['A','R']*100:.2f}%" if "A" in P.index else "n/a")
    c3.metric("Churn 1 mês (R→C)", f"{P.loc['R','C']*100:.2f}%" if ("R" in P.index and "C" in P.columns) else "n/a")

    st.divider()

    # ============================================================
    # 2) Evolução histórica π_t (stacked area)
    # ============================================================
    st.subheader("2) Evolução histórica da base — saúde da carteira (A/R/C)")

    st.info(
    "📊 **Pergunta de negócio:** a base de clientes está ficando mais saudável ou mais frágil?\n\n"
    "Cada linha mostra a fração da base em cada estado ao longo do tempo:\n"
    "• **A (Ativos saudáveis)**\n"
    "• **R (Em risco)**\n"
    "• **C (Churn)**\n\n"
    "Tendências importantes:\n"
    "• **R subindo** → mais clientes entrando em risco\n"
    "• **C subindo** → churn acumulando\n"
    "• **A caindo** → enfraquecimento da carteira"
    )

    dist_month = month_state_distribution(panel)

    fig, ax = plt.subplots()

    for s in states:
        ax.plot(dist_month.index, dist_month[s], label=s, linewidth=2)

    ax.set_title("Distribuição da base por mês")
    ax.set_ylabel("Proporção da base")
    ax.set_xlabel("Mês")
    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)

    with st.expander("Ver tabela (shares por mês)"):
        st.dataframe(dist_month, use_container_width=True)

    st.divider()

    # ============================================================
    # 3) Churn acumulado por horizonte (P^n)
    # ============================================================
    st.subheader("3) Churn acumulado por horizonte — P(churn em n meses)")

    st.info(
        "✅ **Pergunta de negócio:** se eu pegar um cliente hoje, qual a chance dele churnar em 3/6/12 meses?\n"
        "Isso ajuda a definir metas por horizonte e a priorizar ações."
    )

    start_state = st.selectbox("Estado inicial para análise", options=["A", "R"], index=0, key="graphs_start_state")
    horizon = st.slider("Horizonte máximo (meses)", 1, 60, 24, 1, key="graphs_horizon")

    churn_idx = idx.get("C", None)

    probs = []
    for n in range(1, horizon + 1):
        Pn = safe_matrix_power(P_np, n)
        probs.append(Pn[idx[start_state], churn_idx])

    curve_df = pd.DataFrame({"n": np.arange(1, horizon + 1), "P(churn até n)": probs}).set_index("n")

    fig, ax = plt.subplots()
    ax.plot(curve_df.index, curve_df["P(churn até n)"])
    ax.set_title(f"Churn acumulado a partir de {start_state}")
    ax.set_xlabel("Meses (n)")
    ax.set_ylabel("Probabilidade acumulada")
    st.pyplot(fig)

    # Resumo 3/6/12 (executivo)
    for k in [3, 6, 12]:
        if k <= horizon:
            st.write(f"📌 **Churn acumulado em {k} meses** (começando em {start_state}): **{curve_df.loc[k,'P(churn até n)']*100:.2f}%**")

    st.divider()

    # ============================================================
    # 4) Projeção futura da base (π0 P^n) - forecast
    # ============================================================
    st.subheader("4) Projeção da base — se nada mudar (π₀ Pⁿ)")

    st.info(
        "✅ **Pergunta de negócio:** se continuarmos operando igual, como a base tende a evoluir?\n"
        "Isso é útil para mostrar o 'custo de não agir' para diretoria."
    )

    last_month = panel["month_ts"].max()
    pi0 = (
        panel[panel["month_ts"] == last_month]["state"]
        .value_counts(normalize=True)
        .reindex(states).fillna(0.0)
        .to_numpy()
    )

    sim_h = st.slider("Projetar até (meses)", 6, 120, 36, 6, key="graphs_forecast_h")

    pis = []
    for n in range(0, sim_h + 1):
        Pn = safe_matrix_power(P_np, n)
        pis.append(pi0 @ Pn)

    pi_df = pd.DataFrame(pis, columns=states)
    pi_df.index.name = "n_meses"

    fig, ax = plt.subplots()
    for s in states:
        ax.plot(pi_df.index, pi_df[s], label=s)
    ax.legend()
    ax.set_title("Projeção da distribuição da base (π₀ Pⁿ)")
    ax.set_xlabel("Meses à frente")
    ax.set_ylabel("Proporção")
    st.pyplot(fig)

    st.divider()

    # ============================================================
    # 5) Tempo médio até churn
    # ============================================================
    st.subheader("5) Tempo médio até churn — janela de intervenção")

    st.info(
        "✅ **Pergunta de negócio:** quanto tempo temos, em média, para agir antes do churn?\n"
        "Se o tempo em R é curto, campanhas precisam ser rápidas."
    )

    if absorbing:
        Q = get_Q(P, transient=("A", "R"))
        t_vec, Nfund = expected_time_to_absorption(Q)
        t_df = pd.DataFrame({"Estado inicial": ["A", "R"], "Tempo médio até churn (meses)": t_vec})
        st.dataframe(t_df, use_container_width=True)

        fig, ax = plt.subplots()
        ax.bar(t_df["Estado inicial"], t_df["Tempo médio até churn (meses)"])
        ax.set_title("Tempo médio até churn por estado inicial")
        ax.set_ylabel("Meses")
        st.pyplot(fig)

        with st.expander("Ver matriz fundamental N"):
            st.dataframe(pd.DataFrame(Nfund, index=["A","R"], columns=["A","R"]).style.format("{:.4f}"), use_container_width=True)
    else:
        st.warning("Tempo até churn faz mais sentido quando C é absorvente.")

    st.divider()

    # ============================================================
    # 6) LTV Markov + teto de investimento
    # ============================================================
    st.subheader("6) LTV Markov — quanto vale um cliente hoje?")

    st.info(
        "✅ **Pergunta de negócio:** qual o valor esperado de receita futura de um cliente a partir do estado atual?\n"
        "E quanto vale salvar um cliente em risco (diferença entre LTV(A) e LTV(R))?"
    )

    remove_negative = st.checkbox("Ignorar revenue negativo (devoluções) no reward", True, key="graphs_rm_neg")
    rewards = reward_by_state(panel, remove_negative=remove_negative)

    st.write("Ganho médio mensal estimado por estado (reward):")
    st.dataframe(rewards.to_frame("ganho_medio_mensal").style.format("{:.2f}"), use_container_width=True)

    # desconto: explicar em linguagem de negócio
    st.markdown("**Taxa de desconto (γ):** quanto menor, mais conservador (dinheiro futuro vale menos).")
    discount = st.slider(
        "γ (1.00 = sem desconto; 0.98 ≈ 2% ao mês)",
        0.80, 1.00, 0.98, 0.01,
        key="graphs_gamma"
    )

    if absorbing:
        Q = get_Q(P, transient=("A","R"))
        r = rewards.reindex(["A","R"]).to_numpy(dtype=float).reshape(-1,1)

        I = np.eye(Q.shape[0])
        V = np.linalg.inv(I - discount * Q) @ r
        V = V.flatten()

        ltv_df = pd.DataFrame({"Estado inicial": ["A","R"], "LTV esperado (a partir de hoje)": V})
        st.dataframe(ltv_df, use_container_width=True)

        # insight executivo
        ltv_gap = float(V[0] - V[1])
        st.success(f"📌 **Quanto vale salvar um cliente em risco (A vs R):** ~ {ltv_gap:,.2f} de revenue futuro (antes de margem).")

        fig, ax = plt.subplots()
        ax.bar(ltv_df["Estado inicial"], ltv_df["LTV esperado (a partir de hoje)"])
        ax.set_title("LTV esperado por estado (Markov)")
        ax.set_ylabel("Revenue esperado (descontado)")
        st.pyplot(fig)
    else:
        st.warning("LTV até churn funciona melhor quando C é absorvente.")

    st.divider()

    # ============================================================
    # 7) Foto executiva (último mês) - clientes únicos
    # ============================================================
    st.subheader("7) Foto executiva — clientes únicos por estado (último mês)")

    st.info(
        "✅ **Pergunta de negócio:** quantos clientes estão hoje em cada estado para dimensionar operação?\n"
        "Ex.: quantos clientes precisam de campanha de reativação (R)?"
    )

    snap = panel[panel["month_ts"] == last_month].copy()
    uniq = snap.groupby("state")[customer_col].nunique().reindex(states).fillna(0).astype(int)

    st.dataframe(uniq.to_frame("clientes_unicos").T, use_container_width=True)
