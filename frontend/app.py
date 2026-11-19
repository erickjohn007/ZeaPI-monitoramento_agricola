"""ZeaPI — Plataforma de Inteligência Agroclimática e Risco Hídrico do Piauí."""

import os
import json
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# ==============================================================================
# 1. CONFIGURAÇÃO GLOBAL DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="ZeaPI · Inteligência Agroclimática do Piauí",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Caminhos absolutos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
PARQUET_PATH = os.path.join(PROCESSED_DIR, "agroclima_consolidado.parquet")
META_PATH = os.path.join(PROCESSED_DIR, "metadados_modelo.json")
GEOJSON_PATH = os.path.join(PROCESSED_DIR, "municipios_piaui.geojson")

# ==============================================================================
# 2. DESIGN SYSTEM & ENQUADRAMENTO FRONT-END PROFISSIONAL
# ==============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --primary-green: #14532D;
        --accent-green: #15803D;
        --soft-green: #DCFCE7;
        --accent-amber: #D97706;
        --slate-900: #0F172A;
        --slate-700: #334155;
        --slate-500: #64748B;
        --slate-200: #E2E8F0;
        --bg-canvas: #F1F5F9;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background-color: var(--bg-canvas);
    }

    /* Header transparente preservando o botão de recolher/abrir sidebar */
    header[data-testid="stHeader"] {
        background: transparent !important;
        pointer-events: none;
    }
    header[data-testid="stHeader"] button,
    div[data-testid="stSidebarCollapsedControl"] button,
    button[data-testid="stBaseButton-headerNoPadding"] {
        pointer-events: auto !important;
        background-color: #FFFFFF !important;
        border: 1px solid var(--slate-200) !important;
        border-radius: 8px !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08) !important;
        color: var(--primary-green) !important;
    }
    div[data-testid="stAppDeployButton"] {
        display: none !important;
    }

    /* Grid principal ocupando 100% da largura disponível ao abrir ou fechar a sidebar */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
        width: 100% !important;
    }

    /* Barra lateral: aplica largura apenas quando está ABERTA (aria-expanded="true") */
    section[data-testid="stSidebar"][aria-expanded="true"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid var(--slate-200);
        min-width: 300px !important;
        max-width: 320px !important;
    }
    section[data-testid="stSidebar"][aria-expanded="false"] {
        min-width: 0px !important;
        width: 0px !important;
        margin-left: 0px !important;
        border-right: none !important;
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 2rem !important;
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
    }

    /* Tags dos Multiselects em Verde Floresta */
    span[data-baseweb="tag"],
    div[data-baseweb="tag"] {
        background-color: var(--primary-green) !important;
        color: #FFFFFF !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        font-size: 0.74rem !important;
        max-width: 100%;
    }
    span[data-baseweb="tag"] span,
    div[data-baseweb="tag"] span {
        color: #FFFFFF !important;
    }
    span[data-baseweb="tag"] svg,
    div[data-baseweb="tag"] svg {
        fill: #FFFFFF !important;
    }

    /* Cards de KPIs Executivos com altura uniforme e tipografia fluida */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid var(--slate-200);
        border-top: 4px solid var(--accent-green);
        padding: 16px 18px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.74rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: var(--slate-500) !important;
        margin-bottom: 4px;
    }
    div[data-testid="stMetricValue"] {
        font-size: clamp(1.35rem, 1.8vw, 1.8rem) !important;
        font-weight: 800 !important;
        color: var(--slate-900) !important;
        letter-spacing: -0.02em;
        line-height: 1.15;
    }

    /* Moldura limpa para os gráficos Plotly */
    div[data-testid="stPlotlyChart"] {
        background-color: #FFFFFF;
        border: 1px solid var(--slate-200);
        border-radius: 10px;
        padding: 12px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }

    div[role="radiogroup"] {
        background-color: #FFFFFF;
        padding: 10px 14px;
        border-radius: 10px;
        border: 1px solid var(--slate-200);
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 3. CAMADA DE DADOS E CACHE
# ==============================================================================
@st.cache_data
def carregar_geojson_piaui():
    """Obtém a malha municipal oficial do Piauí (IBGE)."""
    if os.path.exists(GEOJSON_PATH):
        with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    url = "https://servicodados.ibge.gov.br/api/v3/malhas/estados/22?formato=application/vnd.geo+json&qualidade=minima&intrarregiao=municipio"
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    geojson = resp.json()

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(GEOJSON_PATH, "w", encoding="utf-8") as f:
        json.dump(geojson, f)
    return geojson


@st.cache_data
def carregar_dados():
    """Carrega base processada do pipeline ou dados sintéticos de demonstração."""
    if os.path.exists(PARQUET_PATH) and os.path.exists(META_PATH):
        df = pd.read_parquet(PARQUET_PATH)
        df["cod_ibge"] = df["cod_ibge"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return df, meta, False

    np.random.seed(42)
    anos = list(range(2000, 2025))
    municipios_exemplo = [
        ("2211001", "Teresina"), ("2211209", "Uruçuí"),
        ("2201903", "Bom Jesus"), ("2200905", "Baixa Grande do Ribeiro"),
        ("2203206", "Corrente"), ("2208007", "Picos"),
        ("2203909", "Floriano"), ("2207702", "Parnaíba"),
    ]
    registros = []
    for cod, nome in municipios_exemplo:
        base_prod = 4800 if nome in ["Uruçuí", "Bom Jesus", "Baixa Grande do Ribeiro"] else 1400
        for ano in anos:
            chuva = max(380, np.random.normal(880, 180))
            temp = np.random.normal(27.5, 1.1)
            area_p = np.random.uniform(5000, 65000) if base_prod > 3000 else np.random.uniform(800, 6000)
            perda = max(2.0, min(45.0, (950 - chuva) * 0.04 + np.random.normal(5, 3)))
            area_c = area_p * (1 - perda / 100)
            rend = max(450, base_prod * (chuva / 900) ** 0.75 + np.random.normal(0, 250))
            prod = (area_c * rend) / 1000
            risco = "Alto Risco" if (chuva < 650 or perda > 25) else ("Risco Moderado" if chuva <= 900 else "Baixo Risco")
            registros.append({
                "cod_ibge": cod, "municipio": nome, "ano": ano,
                "area_plantada_ha": area_p, "area_colhida_ha": area_c,
                "producao_ton": prod, "rendimento_kg_ha": rend,
                "taxa_perda_pct": perda, "precipitacao_anual_mm": chuva,
                "temp_media_c": temp, "temp_max_c": temp + 6.2,
                "nivel_risco_climatico": risco
            })
    meta_demo = {
        "metricas_modelo": {
            "rmse_kg_ha": 312.45,
            "r2_score": 0.8742,
            "algoritmo": "Random Forest Regressor"
        },
        "importancia_variaveis": {
            "precipitacao_anual_mm": 0.44,
            "area_plantada_ha": 0.21,
            "temp_max_c": 0.14
        }
    }
    return pd.DataFrame(registros), meta_demo, True


df, meta, modo_demo = carregar_dados()

anos_disponiveis = sorted([int(a) for a in df["ano"].unique()])
municipios_disponiveis = sorted(df["municipio"].unique().tolist())
riscos_opcoes = ["Baixo Risco", "Risco Moderado", "Alto Risco"]

# ==============================================================================
# 4. ESTADO GLOBAL (FILTROS + NAVEGAÇÃO DE ABAS + CONTROLE DO TOUR)
# ==============================================================================
ABAS_NOMES = [
    "Série Histórica & Sensibilidade Hídrica",
    "Mapa de Risco Municipal (IBGE)",
    "Simulador de Choques Climáticos",
    "Diagnóstico do Modelo Preditivo"
]

if "aba_ativa" not in st.session_state:
    st.session_state["aba_ativa"] = ABAS_NOMES[0]

if "tour_ativo" not in st.session_state:
    st.session_state["tour_ativo"] = False

if "tour_step" not in st.session_state:
    st.session_state["tour_step"] = 0

if "tour_just_closed" not in st.session_state:
    st.session_state["tour_just_closed"] = False

if "filtros_inicializados" not in st.session_state:
    qp = st.query_params

    modo_url = qp.get("modo", "Intervalo Contínuo")
    st.session_state["f_modo"] = modo_url if modo_url in ["Intervalo Contínuo", "Safras Específicas"] else "Intervalo Contínuo"

    try:
        ini_url = int(qp.get("ini", min(anos_disponiveis)))
        fim_url = int(qp.get("fim", max(anos_disponiveis)))
        if ini_url not in anos_disponiveis or fim_url not in anos_disponiveis or ini_url > fim_url:
            ini_url, fim_url = min(anos_disponiveis), max(anos_disponiveis)
    except ValueError:
        ini_url, fim_url = min(anos_disponiveis), max(anos_disponiveis)
    st.session_state["f_intervalo"] = (ini_url, fim_url)

    anos_url_str = qp.get("anos", "")
    if anos_url_str:
        st.session_state["f_anos_esp"] = [int(x) for x in anos_url_str.split(",") if x.isdigit() and int(x) in anos_disponiveis]
    else:
        st.session_state["f_anos_esp"] = [a for a in [2010, 2017] if a in anos_disponiveis] or anos_disponiveis[-2:]

    mun_url_str = qp.get("mun", "")
    st.session_state["f_municipios"] = [m for m in mun_url_str.split("|") if m in municipios_disponiveis] if mun_url_str else []

    risco_url_str = qp.get("risco", "")
    if "risco" in qp:
        st.session_state["f_riscos"] = [r for r in risco_url_str.split("|") if r in riscos_opcoes]
    else:
        st.session_state["f_riscos"] = riscos_opcoes.copy()

    st.session_state["filtros_inicializados"] = True


def sincronizar_url():
    """Atualiza a URL de forma segura no on_change."""
    modo = st.session_state.get("f_modo", "Intervalo Contínuo")
    st.query_params["modo"] = modo

    if modo == "Intervalo Contínuo":
        ini, fim = st.session_state.get("f_intervalo", (min(anos_disponiveis), max(anos_disponiveis)))
        st.query_params["ini"] = str(ini)
        st.query_params["fim"] = str(fim)
        st.query_params.pop("anos", None)
    else:
        default_esp = [a for a in [2010, 2017] if a in anos_disponiveis] or anos_disponiveis[-2:]
        anos_esp = st.session_state.get("f_anos_esp", default_esp)
        st.query_params["anos"] = ",".join(str(a) for a in sorted(anos_esp))
        st.query_params.pop("ini", None)
        st.query_params.pop("fim", None)

    municipios = st.session_state.get("f_municipios", [])
    if municipios:
        st.query_params["mun"] = "|".join(municipios)
    else:
        st.query_params.pop("mun", None)

    riscos = st.session_state.get("f_riscos", riscos_opcoes)
    st.query_params["risco"] = "|".join(riscos)


def resetar_filtros():
    """Restaura todos os filtros para 2000-2024 e limpa a URL."""
    st.session_state["f_modo"] = "Intervalo Contínuo"
    st.session_state["f_intervalo"] = (min(anos_disponiveis), max(anos_disponiveis))
    st.session_state["f_anos_esp"] = [a for a in [2010, 2017] if a in anos_disponiveis] or anos_disponiveis[-2:]
    st.session_state["f_municipios"] = []
    st.session_state["f_riscos"] = riscos_opcoes.copy()
    st.query_params.clear()


TOUR_CENAS = [
    {
        "parte": "Parte 1 de 7 · Cabeçalho",
        "titulo": "Resumo do que você está vendo",
        "texto": "Esta barra mostra o nome do sistema e quais cidades e anos estão selecionados agora. Se você atualizar a página, suas escolhas continuam salvas.",
        "dica": "💡 Clique na setinha no canto superior esquerdo para esconder os filtros e ver os gráficos em tela cheia.",
        "alvo": "header",
        "lado": "bottom",
        "aba_idx": 0
    },
    {
        "parte": "Parte 2 de 7 · Filtros",
        "titulo": "Escolha anos, cidades e nível de risco",
        "texto": "Aqui você escolhe o período (de 2000 a 2024), compara uma ou várias cidades do Piauí ao mesmo tempo e filtra apenas os anos com falta de chuva.",
        "dica": "💡 Use 'Safras Específicas' para comparar um ano de seca forte (2012) com um ano de chuva boa (2020).",
        "alvo": "sidebar",
        "lado": "right",
        "aba_idx": 0
    },
    {
        "parte": "Parte 3 de 7 · Números Principais",
        "titulo": "Os 4 resultados mais importantes",
        "texto": "Estes cartões mostram quanto milho foi colhido no total, a produtividade média por hectare, a média de chuvas no ano e quanto da plantação foi perdida.",
        "dica": "💡 Sempre que você muda um filtro na barra lateral, esses 4 números se atualizam na hora.",
        "alvo": "kpis",
        "lado": "bottom",
        "aba_idx": 0
    },
    {
        "parte": "Parte 4 de 7 · Gráficos Históricos",
        "titulo": "A relação entre chuva e colheita",
        "texto": "Na esquerda, as barras azuis mostram a chuva e a linha verde mostra a colheita ano a ano. Na direita, cada bolinha é uma cidade: quanto maior a bolha, maior a plantação.",
        "dica": "💡 Passe o mouse sobre qualquer bolinha ou barra para ver o nome da cidade e os números exatos.",
        "alvo": "conteudo_aba",
        "lado": "top",
        "aba_idx": 0
    },
    {
        "parte": "Parte 5 de 7 · Mapa do Piauí",
        "titulo": "Veja a situação de cada município no mapa",
        "texto": "Este mapa mostra todas as 224 cidades do Piauí. Use os botões acima dele para trocar as cores e ver onde houve mais perda de safra, chuva ou produção.",
        "dica": "💡 Use a rodinha do mouse para aproximar o mapa e passe o cursor sobre qualquer cidade.",
        "alvo": "mapa_aba",
        "lado": "left_corner",
        "aba_idx": 1
    },
    {
        "parte": "Parte 6 de 7 · Simulador de Clima",
        "titulo": "Teste o que acontece se faltar chuva",
        "texto": "Arraste os controles para simular uma seca ou aumento de calor. O sistema calcula na hora quantas toneladas de milho seriam perdidas na lavoura.",
        "dica": "💡 Ajuda produtores e empresas a planejarem investimentos antes que a seca cause prejuízo.",
        "alvo": "conteudo_aba",
        "lado": "top",
        "aba_idx": 2
    },
    {
        "parte": "Parte 7 de 7 · Inteligência Artificial",
        "titulo": "O que mais influencia na colheita?",
        "texto": "Aqui mostramos como a Inteligência Artificial do sistema funciona e quais fatores mais pesam no resultado final, como o tamanho da área plantada e o volume de chuvas.",
        "dica": "💡 Pronto! Clique em 'Finalizar ✓' para voltar à tela normal e explorar livremente.",
        "alvo": "conteudo_aba",
        "lado": "top",
        "aba_idx": 3
    }
]


def encerrar_tour():
    """Encerra o tour e retorna o dashboard limpo para a página inicial normal."""
    st.session_state["tour_ativo"] = False
    st.session_state["tour_step"] = 0
    st.session_state["aba_ativa"] = ABAS_NOMES[0]
    st.session_state["nav_radio"] = ABAS_NOMES[0]
    st.session_state["tour_just_closed"] = True


def ir_para_passo_tour(novo_passo: int):
    """Atualiza o passo do tour e força a aba correspondente ou finaliza voltando ao normal."""
    if 0 <= novo_passo < len(TOUR_CENAS):
        st.session_state["tour_ativo"] = True
        st.session_state["tour_step"] = novo_passo
        aba_alvo = ABAS_NOMES[TOUR_CENAS[novo_passo]["aba_idx"]]
        st.session_state["aba_ativa"] = aba_alvo
        st.session_state["nav_radio"] = aba_alvo
    else:
        encerrar_tour()


# ==============================================================================
# 5. BARRA LATERAL RECOLHÍVEL (RESPOSTA INSTANTÂNEA)
# ==============================================================================
with st.sidebar:
    st.markdown(
        "<div style='font-size: 0.78rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.06em; color: #14532D; margin-bottom: 14px; padding-bottom: 8px; border-bottom: 2px solid #E2E8F0;'>Painel de Filtros</div>",
        unsafe_allow_html=True
    )

    if modo_demo:
        st.info("Base demonstrativa ativa. Rode `pipeline_spark.py` para carregar os 224 municípios.")

    modo_tempo = st.radio(
        "Modo de Seleção Temporal",
        options=["Intervalo Contínuo", "Safras Específicas"],
        key="f_modo",
        on_change=sincronizar_url,
        horizontal=True,
        help="Escolha entre um intervalo contínuo (ex: 2000 a 2024) ou anos isolados para comparação direta."
    )

    if modo_tempo == "Intervalo Contínuo":
        if "f_intervalo" not in st.session_state:
            st.session_state["f_intervalo"] = (min(anos_disponiveis), max(anos_disponiveis))
        intervalo_anos = st.slider(
            "Período Histórico (Safras)",
            min_value=min(anos_disponiveis),
            max_value=max(anos_disponiveis),
            key="f_intervalo",
            on_change=sincronizar_url
        )
        anos_selecionados = [a for a in anos_disponiveis if intervalo_anos[0] <= a <= intervalo_anos[1]]
        label_safras = f"{intervalo_anos[0]}–{intervalo_anos[1]}"
    else:
        if "f_anos_esp" not in st.session_state:
            st.session_state["f_anos_esp"] = [a for a in [2010, 2017] if a in anos_disponiveis] or anos_disponiveis[-2:]
        anos_esp = st.multiselect(
            "Selecione os Anos (Safras)",
            options=anos_disponiveis,
            key="f_anos_esp",
            on_change=sincronizar_url,
            placeholder="Escolha um ou mais anos..."
        )
        if not anos_esp:
            anos_selecionados = anos_disponiveis
            label_safras = f"{min(anos_disponiveis)}–{max(anos_disponiveis)} (Todos)"
        else:
            anos_selecionados = anos_esp
            if len(anos_selecionados) <= 4:
                label_safras = ", ".join(str(a) for a in sorted(anos_selecionados))
            else:
                label_safras = f"{len(anos_selecionados)} safras selecionadas"

    st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

    if "f_municipios" not in st.session_state:
        st.session_state["f_municipios"] = []
    municipios_sel = st.multiselect(
        "Recorte Territorial (Municípios)",
        options=municipios_disponiveis,
        key="f_municipios",
        on_change=sincronizar_url,
        placeholder="Todos os municípios (Estado)",
        help="Selecione quantos municípios desejar comparar. Deixe vazio para visualizar todo o Piauí."
    )

    if not municipios_sel:
        label_territorio = "Todos os Municípios (Piauí)"
    elif len(municipios_sel) == 1:
        label_territorio = municipios_sel[0]
    elif len(municipios_sel) <= 3:
        label_territorio = ", ".join(municipios_sel)
    else:
        label_territorio = f"{len(municipios_sel)} municípios selecionados"

    st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

    if "f_riscos" not in st.session_state:
        st.session_state["f_riscos"] = riscos_opcoes.copy()
    risco_sel = st.multiselect(
        "Perfil de Risco Climático",
        options=riscos_opcoes,
        key="f_riscos",
        on_change=sincronizar_url,
        help="Classificação baseada no déficit hídrico e na taxa histórica de perda de área."
    )

    st.button("Restaurar Filtros Padrão", on_click=resetar_filtros, use_container_width=True)

    st.markdown(
        """
        <div style="margin-top: 20px; padding: 14px; background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; font-size: 0.73rem; color: #475569; line-height: 1.55;">
            <strong style="color: #0F172A; display: block; margin-bottom: 4px;">Arquitetura &amp; Fontes</strong>
            • <b>Safra:</b> IBGE / PAM (2000–2024)<br>
            • <b>Clima:</b> NASA POWER API<br>
            • <b>Modelo:</b> Random Forest Regressor
        </div>
        """,
        unsafe_allow_html=True
    )

# ==============================================================================
# 6. FILTRAGEM CACHEADA DOS DADOS
# ==============================================================================
@st.cache_data(show_spinner=False)
def filtrar_dataset(anos_tuple, municipios_tuple, riscos_tuple):
    mask = (df["ano"].isin(anos_tuple)) & (df["nivel_risco_climatico"].isin(riscos_tuple))
    if municipios_tuple:
        mask = mask & (df["municipio"].isin(municipios_tuple))
    return df[mask]


df_filtrado = filtrar_dataset(
    tuple(sorted(anos_selecionados)),
    tuple(sorted(municipios_sel)),
    tuple(sorted(risco_sel))
)

# ==============================================================================
# 7. HEADER CARD UNIFICADO + BOTÃO DO TOUR INTERATIVO
# ==============================================================================
h_info, h_tour = st.columns([5.1, 1.2], gap="small")

with h_info:
    st.markdown(
        f"""
        <div id="zeapi-header-anchor" style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 16px 22px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04); display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px;">
            <div style="display: flex; align-items: center; gap: 14px;">
                <svg width="46" height="46" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg" style="flex-shrink: 0;">
                    <defs>
                        <linearGradient id="zeapiCardBg" x1="0" y1="0" x2="64" y2="64" gradientUnits="userSpaceOnUse">
                            <stop offset="0%" stop-color="#14532D"/>
                            <stop offset="100%" stop-color="#052E16"/>
                        </linearGradient>
                    </defs>
                    <rect width="64" height="64" rx="14" fill="url(#zeapiCardBg)"/>
                    <path d="M18 44C18 30 26 20 38 18C38 30 30 40 18 44Z" fill="#22C55E" fill-opacity="0.9"/>
                    <path d="M41 17C41 17 49 26 49 33C49 37.4183 45.4183 41 41 41C36.5817 41 33 37.4183 33 33C33 26 41 17 41 17Z" fill="#38BDF8" fill-opacity="0.95"/>
                    <rect x="20" y="42" width="5" height="8" rx="2" fill="#FACC15"/>
                    <rect x="28" y="37" width="5" height="13" rx="2" fill="#FACC15"/>
                    <rect x="36" y="32" width="5" height="18" rx="2" fill="#FACC15"/>
                    <rect x="44" y="27" width="5" height="23" rx="2" fill="#FDE047"/>
                </svg>
                <div>
                    <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
                        <span style="font-size: 1.5rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1; color: #0F172A;">
                            Zea<span style="color: #D97706;">PI</span>
                        </span>
                        <span style="background-color: #F0FDF4; border: 1px solid #BBF7D0; color: #14532D; font-size: 0.68rem; font-weight: 700; padding: 3px 8px; border-radius: 5px; text-transform: uppercase; letter-spacing: 0.04em;">
                            Observatório do Milho · PI
                        </span>
                    </div>
                    <div style="font-size: 0.88rem; font-weight: 500; color: #475569; margin-top: 4px;">
                        Monitoramento de Produtividade Agrícola &amp; Risco Hídrico
                    </div>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 7px 13px; border-radius: 8px; font-size: 0.79rem; color: #334155;">
                    Recorte: <strong style="color: #0F172A;">{label_territorio}</strong>
                </div>
                <div style="background-color: #F0FDF4; border: 1px solid #BBF7D0; padding: 7px 13px; border-radius: 8px; font-size: 0.79rem; color: #14532D;">
                    Safras: <strong>{label_safras}</strong>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with h_tour:
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    if not st.session_state["tour_ativo"]:
        if st.button("▶ Tour Interativo", type="primary", help="Guia visual passo a passo de todas as funções da tela.", use_container_width=True):
            ir_para_passo_tour(0)
            st.rerun()
    else:
        if st.button("✕ Fechar Tour", type="secondary", use_container_width=True):
            encerrar_tour()
            st.rerun()

# ==============================================================================
# 8. GATILHOS NATIVOS INVISÍVEIS DO TOUR (INÍCIO NA 1ª ENTRADA + PASSOS)
# ==============================================================================
step_atual = st.session_state["tour_step"]
st.markdown(
    """
    <style>
        div[data-testid="stHorizontalBlock"]:has(#zeapi-hidden-tour-triggers) {
            position: fixed !important;
            top: -300px !important;
            left: -300px !important;
            opacity: 0 !important;
            pointer-events: none !important;
            height: 0 !important;
            overflow: hidden !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)
c_trig0, c_trig1, c_trig2, c_trig3 = st.columns(4)
with c_trig0:
    st.markdown("<div id='zeapi-hidden-tour-triggers'></div>", unsafe_allow_html=True)
    if st.button("ZEAPI_TOUR_AUTOSTART", key="btn_tour_autostart"):
        ir_para_passo_tour(0)
        st.rerun()
with c_trig1:
    if st.button("ZEAPI_TOUR_PREV", key="btn_tour_prev"):
        ir_para_passo_tour(step_atual - 1)
        st.rerun()
with c_trig2:
    if st.button("ZEAPI_TOUR_NEXT", key="btn_tour_next"):
        ir_para_passo_tour(step_atual + 1)
        st.rerun()
with c_trig3:
    if st.button("ZEAPI_TOUR_CLOSE", key="btn_tour_close"):
        encerrar_tour()
        st.rerun()

# ==============================================================================
# 9. KPIS PRINCIPAIS
# ==============================================================================
col1, col2, col3, col4 = st.columns(4, gap="medium")
prod_total_mt = df_filtrado["producao_ton"].sum() / 1_000_000
rend_medio_tha = (df_filtrado["rendimento_kg_ha"].mean() / 1000) if not df_filtrado.empty else 0
chuva_media = df_filtrado["precipitacao_anual_mm"].mean() if not df_filtrado.empty else 0
perda_media = df_filtrado["taxa_perda_pct"].mean() if not df_filtrado.empty else 0

col1.metric("Produção Acumulada", f"{prod_total_mt:,.2f} Mi t", help="Volume total colhido no período e recorte selecionados.")
col2.metric("Rendimento Médio", f"{rend_medio_tha:,.2f} t/ha", help="Produtividade média por hectare colhido.")
col3.metric("Precipitação Média", f"{chuva_media:,.0f} mm/ano", help="Média anual de chuvas acumuladas (NASA POWER).")
col4.metric("Perda Média de Área", f"{perda_media:,.1f}%", help="Percentual médio de área plantada não colhida por estresse climático.")

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

# ==============================================================================
# 10. BARRA DE NAVEGAÇÃO CONTROLADA POR ESTADO
# ==============================================================================
def on_nav_change():
    st.session_state["aba_ativa"] = st.session_state["nav_radio"]


if "nav_radio" not in st.session_state:
    st.session_state["nav_radio"] = st.session_state["aba_ativa"]

aba_selecionada = st.radio(
    "Navegação de Módulos",
    options=ABAS_NOMES,
    key="nav_radio",
    on_change=on_nav_change,
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("<div id='zeapi-aba-conteudo-anchor'></div>", unsafe_allow_html=True)

# --- MÓDULO 1: SÉRIE HISTÓRICA ---
if aba_selecionada == ABAS_NOMES[0]:
    c_esq, c_dir = st.columns([1.15, 1], gap="medium")

    with c_esq:
        df_anual = df_filtrado.groupby("ano", as_index=False).agg({
            "rendimento_kg_ha": "mean",
            "precipitacao_anual_mm": "mean",
            "producao_ton": "sum"
        }).sort_values("ano")

        eixo_x = df_anual["ano"].astype(str) if modo_tempo == "Safras Específicas" else df_anual["ano"]

        fig_dual = go.Figure()
        fig_dual.add_trace(go.Bar(
            x=eixo_x,
            y=df_anual["precipitacao_anual_mm"],
            name="Precipitação (mm)",
            marker_color="rgba(14, 165, 233, 0.35)",
            yaxis="y2"
        ))
        fig_dual.add_trace(go.Scatter(
            x=eixo_x,
            y=df_anual["rendimento_kg_ha"],
            name="Rendimento (kg/ha)",
            mode="lines+markers",
            line=dict(color="#14532D", width=3),
            marker=dict(size=7, color="#D97706")
        ))
        fig_dual.update_layout(
            title=dict(text="Evolução por Safra: Rendimento vs. Precipitação", font=dict(size=14, color="#0F172A", family="Inter"), y=0.96),
            xaxis=dict(type="category" if modo_tempo == "Safras Específicas" else "linear", automargin=True),
            yaxis=dict(title="Rendimento (kg/ha)", showgrid=True, gridcolor="#F1F5F9", zeroline=False, automargin=True),
            yaxis2=dict(title="Precipitação (mm)", overlaying="y", side="right", showgrid=False, zeroline=False, automargin=True),
            legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1),
            height=430,
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            margin=dict(l=16, r=16, t=68, b=16)
        )
        st.plotly_chart(fig_dual, use_container_width=True)

    with c_dir:
        fig_disp = px.scatter(
            df_filtrado,
            x="precipitacao_anual_mm",
            y="rendimento_kg_ha",
            color="nivel_risco_climatico",
            size="area_plantada_ha",
            hover_name="municipio",
            hover_data=["ano", "taxa_perda_pct"],
            color_discrete_map={
                "Baixo Risco": "#15803D",
                "Risco Moderado": "#D97706",
                "Alto Risco": "#B91C1C"
            },
            labels={
                "precipitacao_anual_mm": "Precipitação Anual (mm)",
                "rendimento_kg_ha": "Rendimento (kg/ha)",
                "nivel_risco_climatico": "Risco"
            },
            height=430
        )
        fig_disp.update_layout(
            title=dict(text="Sensibilidade Hídrica por Município/Safra", font=dict(size=14, color="#0F172A", family="Inter"), y=0.96),
            legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1, title=None),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            xaxis=dict(showgrid=True, gridcolor="#F1F5F9", zeroline=False, automargin=True),
            yaxis=dict(showgrid=True, gridcolor="#F1F5F9", zeroline=False, automargin=True),
            margin=dict(l=16, r=16, t=68, b=16)
        )
        st.plotly_chart(fig_disp, use_container_width=True)

# --- MÓDULO 2: MAPA DE RISCO ---
elif aba_selecionada == ABAS_NOMES[1]:
    @st.fragment
    def renderizar_mapa(df_base):
        c_map_ctrl, c_map_info = st.columns([3.2, 1], gap="medium")
        with c_map_ctrl:
            metrica_mapa = st.radio(
                "Selecione a variável cartográfica:",
                options=["taxa_perda_pct", "precipitacao_anual_mm", "rendimento_kg_ha", "producao_ton"],
                format_func=lambda x: {
                    "taxa_perda_pct": "Taxa de Perda de Safra (%)",
                    "precipitacao_anual_mm": "Precipitação Média (mm)",
                    "rendimento_kg_ha": "Rendimento Médio (kg/ha)",
                    "producao_ton": "Produção Acumulada (t)"
                }[x],
                horizontal=True
            )
        with c_map_info:
            st.markdown(
                f"""
                <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 12px 16px; text-align: right; height: 100%; display: flex; flex-direction: column; justify-content: center;">
                    <span style="font-size: 0.70rem; text-transform: uppercase; color: #64748B; font-weight: 700;">Cobertura Espacial</span>
                    <span style="font-size: 1.15rem; font-weight: 800; color: #0F172A;">{df_base['cod_ibge'].nunique()} municípios</span>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        df_mapa = df_base.groupby(["cod_ibge", "municipio"], as_index=False).agg({
            "rendimento_kg_ha": "mean",
            "precipitacao_anual_mm": "mean",
            "taxa_perda_pct": "mean",
            "producao_ton": "sum"
        })

        paletas_mapa = {
            "taxa_perda_pct": "YlOrRd",
            "precipitacao_anual_mm": "Blues",
            "rendimento_kg_ha": "YlGn",
            "producao_ton": "Greens"
        }

        try:
            geojson_pi = carregar_geojson_piaui()
            fig_map = px.choropleth_map(
                df_mapa,
                geojson=geojson_pi,
                locations="cod_ibge",
                featureidkey="properties.codarea",
                color=metrica_mapa,
                hover_name="municipio",
                hover_data={
                    "cod_ibge": False,
                    "taxa_perda_pct": ":.1f",
                    "precipitacao_anual_mm": ":.0f",
                    "rendimento_kg_ha": ":.0f",
                    "producao_ton": ":,.0f"
                },
                color_continuous_scale=paletas_mapa[metrica_mapa],
                map_style="carto-positron",
                zoom=5.6,
                center={"lat": -6.95, "lon": -42.95},
                opacity=0.88,
                height=540,
                labels={
                    "taxa_perda_pct": "Perda (%)",
                    "precipitacao_anual_mm": "Chuva (mm)",
                    "rendimento_kg_ha": "Rendimento (kg/ha)",
                    "producao_ton": "Produção (t)"
                }
            )
            fig_map.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                coloraxis_colorbar=dict(
                    bgcolor="rgba(255,255,255,0.92)",
                    bordercolor="#E2E8F0",
                    borderwidth=1,
                    thickness=14,
                    len=0.72,
                    x=0.98
                )
            )
            st.plotly_chart(fig_map, use_container_width=True)
        except Exception as e:
            st.error(f"Erro ao carregar a malha geográfica do IBGE: {e}")

    renderizar_mapa(df_filtrado)

# --- MÓDULO 3: SIMULADOR CLIMÁTICO ---
elif aba_selecionada == ABAS_NOMES[2]:
    @st.fragment
    def renderizar_simulador(df_base):
        st.markdown(
            """
            <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 18px 22px; margin-bottom: 16px;">
                <div style="font-size: 1.05rem; font-weight: 700; color: #0F172A;">Simulador de Estresse Climático (Cenários El Niño / La Niña)</div>
                <div style="font-size: 0.85rem; color: #64748B; margin-top: 2px;">
                    Modele variações na precipitação e na temperatura média para estimar o impacto produtivo sobre a linha de base histórica.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        s1, s2, s3 = st.columns(3, gap="large")
        delta_chuva = s1.slider("Anomalia de Precipitação (%)", min_value=-50, max_value=40, value=-20, step=5)
        delta_temp = s2.slider("Variação de Temperatura Média (°C)", min_value=-1.0, max_value=4.0, value=1.5, step=0.25)
        area_sim = s3.number_input("Área Plantada de Referência (ha)", min_value=100, max_value=500000, value=50000, step=1000)

        chuva_base = df_base["precipitacao_anual_mm"].mean() if not df_base.empty else 850.0
        rend_base = df_base["rendimento_kg_ha"].mean() if not df_base.empty else 3200.0

        chuva_sim = chuva_base * (1 + delta_chuva / 100.0)
        fator_hidrico = (chuva_sim / chuva_base) ** 0.78
        fator_termico = max(0.55, 1.0 - max(0, delta_temp) * 0.065)

        rend_projetado = rend_base * fator_hidrico * fator_termico
        prod_base_ton = (area_sim * rend_base) / 1000.0
        prod_proj_ton = (area_sim * rend_projetado) / 1000.0
        diff_ton = prod_proj_ton - prod_base_ton
        diff_pct = ((rend_projetado - rend_base) / rend_base) * 100

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3, gap="medium")
        r1.metric("Precipitação Projetada", f"{chuva_sim:,.0f} mm", f"{delta_chuva:+d}% vs. média")
        r2.metric("Rendimento Estimado", f"{rend_projetado:,.0f} kg/ha", f"{diff_pct:+.1f}%")
        r3.metric("Produção Total Projetada", f"{prod_proj_ton:,.0f} t", f"{diff_ton:+,.0f} t")

    renderizar_simulador(df_filtrado)

# --- MÓDULO 4: DIAGNÓSTICO DO MODELO ---
elif aba_selecionada == ABAS_NOMES[3]:
    m1, m2, m3 = st.columns(3, gap="medium")
    m1.metric("Arquitetura Preditiva", "Random Forest")
    m2.metric("Coeficiente de Explicação (R²)", f"{meta['metricas_modelo']['r2_score']:.2%}")
    m3.metric("Erro Médio (RMSE)", f"{meta['metricas_modelo']['rmse_kg_ha']} kg/ha")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    df_imp = pd.DataFrame({
        "Variável": list(meta["importancia_variaveis"].keys()),
        "Importância": list(meta["importancia_variaveis"].values())
    }).sort_values("Importância", ascending=True)

    fig_imp = px.bar(
        df_imp,
        x="Importância",
        y="Variável",
        orientation="h",
        text_auto=".0%",
        height=360
    )
    fig_imp.update_traces(marker_color="#14532D", textposition="outside")
    fig_imp.update_layout(
        title=dict(text="Importância das Variáveis na Predição de Produtividade", font=dict(size=14, color="#0F172A", family="Inter"), y=0.96),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        xaxis=dict(showgrid=True, gridcolor="#F1F5F9", tickformat=".0%", range=[0, max(df_imp["Importância"]) * 1.18], automargin=True),
        yaxis=dict(title=None, automargin=True),
        margin=dict(l=16, r=24, t=58, b=16)
    )
    st.plotly_chart(fig_imp, use_container_width=True)


# ==============================================================================
# 11. CONTROLE DE PRIMEIRA ENTRADA (sessionStorage) + OVERLAY DO TOUR
# ==============================================================================
if not st.session_state["tour_ativo"]:
    just_closed_flag = "true" if st.session_state.get("tour_just_closed") else "false"
    if st.session_state.get("tour_just_closed"):
        st.session_state["tour_just_closed"] = False

    components.html(
        f"""
        <script>
        (function() {{
            const doc = window.parent.document;
            const win = window.parent;
            const justClosed = {just_closed_flag};

            // Remove qualquer resquício visual do tour
            const oldRoot = doc.getElementById('zeapi-spot-overlay');
            if (oldRoot) oldRoot.remove();
            const oldStyle = doc.getElementById('zeapi-spot-style');
            if (oldStyle) oldStyle.remove();

            if (justClosed) {{
                const header = doc.getElementById('zeapi-header-anchor');
                if (header && header.scrollIntoView) {{
                    header.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
                }} else {{
                    win.scrollTo({{ top: 0, behavior: 'smooth' }});
                }}
                return;
            }}

            // Dispara o tour OBRIGATORIAMENTE apenas na primeira entrada da aba/sessão
            if (!win.sessionStorage.getItem('zeapi_tour_primeira_entrada')) {{
                win.sessionStorage.setItem('zeapi_tour_primeira_entrada', '1');
                setTimeout(() => {{
                    const buttons = Array.from(doc.querySelectorAll('button'));
                    const startBtn = buttons.find(b => b.innerText && b.innerText.trim() === 'ZEAPI_TOUR_AUTOSTART');
                    if (startBtn) startBtn.click();
                }}, 220);
            }}
        }})();
        </script>
        """,
        height=0,
        width=0
    )
else:
    cena_atual = TOUR_CENAS[st.session_state["tour_step"]]
    cena_json = json.dumps(cena_atual)
    step_num = st.session_state["tour_step"]
    total_steps = len(TOUR_CENAS)

    components.html(
        f"""
        <script>
        (function() {{
            const doc = window.parent.document;
            const win = window.parent;
            win.sessionStorage.setItem('zeapi_tour_primeira_entrada', '1');

            const cena = {cena_json};
            const stepNum = {step_num};
            const totalSteps = {total_steps};

            const oldRoot = doc.getElementById('zeapi-spot-overlay');
            if (oldRoot) oldRoot.remove();
            const oldStyle = doc.getElementById('zeapi-spot-style');
            if (oldStyle) oldStyle.remove();

            // Escurece a barra lateral nos passos 1, 3, 4, 5, 6 e 7; acende somente no passo 2
            const sidebarDimRule = (cena.alvo !== 'sidebar') ? `
                section[data-testid="stSidebar"] {{
                    filter: brightness(0.42) !important;
                    transition: filter 0.3s ease !important;
                }}
            ` : `
                section[data-testid="stSidebar"] {{
                    filter: none !important;
                    transition: filter 0.3s ease !important;
                }}
            `;

            const style = doc.createElement('style');
            style.id = 'zeapi-spot-style';
            style.textContent = `
                ${{sidebarDimRule}}
                #zeapi-spot-overlay {{
                    position: fixed;
                    inset: 0;
                    z-index: 9999999;
                    pointer-events: none;
                    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                }}
                #zeapi-spot-ring {{
                    position: fixed;
                    border-radius: 14px;
                    box-shadow: 0 0 0 9999px rgba(15, 23, 42, 0.62), 0 0 0 3px #22C55E, 0 0 26px rgba(34, 197, 94, 0.5);
                    transition: all 0.35s ease;
                    pointer-events: none;
                }}
                #zeapi-popover-box {{
                    position: fixed;
                    width: 410px;
                    max-width: calc(100vw - 28px);
                    background: rgba(15, 23, 42, 0.98);
                    color: #F8FAFC;
                    border: 1px solid rgba(148, 163, 184, 0.35);
                    border-top: 4px solid #22C55E;
                    border-radius: 14px;
                    box-shadow: 0 24px 48px -12px rgba(0, 0, 0, 0.7);
                    padding: 16px 18px;
                    pointer-events: auto;
                    z-index: 10000000;
                    transition: top 0.35s ease, left 0.35s ease;
                }}
                #zeapi-popover-arrow {{
                    position: absolute;
                    width: 14px;
                    height: 14px;
                    background: #0F172A;
                    border-left: 1px solid rgba(148, 163, 184, 0.35);
                    border-top: 1px solid rgba(148, 163, 184, 0.35);
                }}
                .zp-nav-btn {{
                    border: 1px solid rgba(148, 163, 184, 0.35);
                    background: rgba(255, 255, 255, 0.08);
                    color: #F8FAFC;
                    font-size: 12px;
                    font-weight: 600;
                    padding: 6px 12px;
                    border-radius: 7px;
                    cursor: pointer;
                }}
                .zp-nav-btn:hover:not(:disabled) {{ background: rgba(255, 255, 255, 0.18); }}
                .zp-nav-btn:disabled {{ opacity: 0.35; cursor: not-allowed; }}
                .zp-nav-primary {{
                    background: #16A34A;
                    border-color: #22C55E;
                    color: #FFFFFF;
                }}
                .zp-nav-primary:hover {{ background: #15803D; }}
            `;
            doc.head.appendChild(style);

            const overlay = doc.createElement('div');
            overlay.id = 'zeapi-spot-overlay';
            overlay.innerHTML = `
                <div id="zeapi-spot-ring"></div>
                <div id="zeapi-popover-box">
                    <div id="zeapi-popover-arrow"></div>
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                        <span style="font-size: 11px; font-weight: 800; color: #86EFAC; text-transform: uppercase; letter-spacing: 0.05em;">
                            ${{cena.parte}}
                        </span>
                        <button id="zp-btn-close" class="zp-nav-btn" style="padding:2px 8px;" title="Fechar Tour">✕</button>
                    </div>
                    <div style="font-size: 15.5px; font-weight: 800; color: #FFFFFF; margin-bottom: 6px;">
                        ${{cena.titulo}}
                    </div>
                    <div style="font-size: 13px; color: #E2E8F0; line-height: 1.5; margin-bottom: 10px;">
                        ${{cena.texto}}
                    </div>
                    <div style="background: rgba(217, 119, 6, 0.18); border-left: 3px solid #F59E0B; padding: 7px 10px; border-radius: 6px; font-size: 12px; color: #FDE68A; line-height: 1.4; margin-bottom: 12px;">
                        ${{cena.dica}}
                    </div>
                    <div style="display:flex; justify-content:space-between; align-items:center; gap:8px;">
                        <span style="font-size:11.5px; color:#94A3B8;">Passo ${{stepNum + 1}} de ${{totalSteps}}</span>
                        <div style="display:flex; gap:6px;">
                            <button id="zp-btn-prev" class="zp-nav-btn" ${{stepNum === 0 ? 'disabled' : ''}}>← Anterior</button>
                            <button id="zp-btn-next" class="zp-nav-btn zp-nav-primary">${{stepNum === totalSteps - 1 ? 'Finalizar ✓' : 'Próximo →'}}</button>
                        </div>
                    </div>
                </div>
            `;
            doc.body.appendChild(overlay);

            function clickStreamlitButton(label) {{
                const buttons = Array.from(doc.querySelectorAll('button'));
                const target = buttons.find(b => b.innerText && b.innerText.trim() === label);
                if (target) target.click();
            }}

            doc.getElementById('zp-btn-prev').onclick = () => clickStreamlitButton('ZEAPI_TOUR_PREV');
            doc.getElementById('zp-btn-next').onclick = () => {{
                if (stepNum === totalSteps - 1) {{
                    overlay.remove();
                    style.remove();
                }}
                clickStreamlitButton('ZEAPI_TOUR_NEXT');
            }};
            doc.getElementById('zp-btn-close').onclick = () => {{
                overlay.remove();
                style.remove();
                clickStreamlitButton('ZEAPI_TOUR_CLOSE');
            }};

            function resolveTargetRect() {{
                const vw = win.innerWidth;
                const vh = win.innerHeight;

                if (cena.alvo === 'header') {{
                    const el = doc.getElementById('zeapi-header-anchor');
                    if (el) return el.getBoundingClientRect();
                }}
                if (cena.alvo === 'sidebar') {{
                    const sb = doc.querySelector('section[data-testid="stSidebar"]');
                    if (sb) return sb.getBoundingClientRect();
                }}
                if (cena.alvo === 'kpis') {{
                    const metrics = doc.querySelectorAll('div[data-testid="stMetric"]');
                    if (metrics.length) {{
                        const row = metrics[0].closest('div[data-testid="stHorizontalBlock"]') || metrics[0];
                        return row.getBoundingClientRect();
                    }}
                }}
                if (cena.alvo === 'mapa_aba') {{
                    const anchor = doc.getElementById('zeapi-aba-conteudo-anchor');
                    const mainBlock = doc.querySelector('section.main .block-container') || doc.querySelector('.block-container');
                    if (anchor && mainBlock) {{
                        const aRect = anchor.getBoundingClientRect();
                        const mRect = mainBlock.getBoundingClientRect();
                        const top = Math.max(42, aRect.bottom + 4);
                        const height = Math.max(420, Math.min(vh - top - 16, 635));
                        return {{
                            top: top,
                            left: mRect.left + 16,
                            width: mRect.width - 32,
                            height: height
                        }};
                    }}
                }}
                if (cena.alvo === 'conteudo_aba') {{
                    const anchor = doc.getElementById('zeapi-aba-conteudo-anchor');
                    const mainBlock = doc.querySelector('section.main .block-container') || doc.querySelector('.block-container');
                    if (anchor && mainBlock) {{
                        const aRect = anchor.getBoundingClientRect();
                        const mRect = mainBlock.getBoundingClientRect();
                        const top = Math.max(72, aRect.bottom + 8);
                        const height = Math.max(280, Math.min(vh - top - 24, 510));
                        return {{
                            top: top,
                            left: mRect.left + 16,
                            width: mRect.width - 32,
                            height: height
                        }};
                    }}
                }}
                return {{ top: 100, left: 340, width: vw - 380, height: 320 }};
            }}

            function alignElements() {{
                const rect = resolveTargetRect();
                const vw = win.innerWidth;
                const vh = win.innerHeight;
                const pad = 6;

                const sTop = Math.max(4, rect.top - pad);
                const sLeft = Math.max(4, rect.left - pad);
                const sWidth = Math.min(vw - 8, rect.width + pad * 2);
                const sHeight = Math.min(vh - 8, rect.height + pad * 2);

                const ring = doc.getElementById('zeapi-spot-ring');
                const pop = doc.getElementById('zeapi-popover-box');
                const arrow = doc.getElementById('zeapi-popover-arrow');
                if (!ring || !pop || !arrow) return;

                ring.style.top = sTop + 'px';
                ring.style.left = sLeft + 'px';
                ring.style.width = sWidth + 'px';
                ring.style.height = sHeight + 'px';

                const popRect = pop.getBoundingClientRect();
                const popW = popRect.width || 410;
                const popH = popRect.height || 235;
                const gap = 16;

                let pTop = 24;
                let pLeft = vw - popW - 28;
                let sideUsed = 'corner';

                if (cena.lado === 'right' && (sLeft + sWidth + popW + gap + 12 <= vw)) {{
                    sideUsed = 'right';
                    pLeft = sLeft + sWidth + gap;
                    pTop = Math.min(vh - popH - 18, Math.max(24, sTop + 60));
                }} else if (cena.lado === 'bottom' && (sTop + sHeight + popH + gap + 12 <= vh)) {{
                    sideUsed = 'bottom';
                    pTop = sTop + sHeight + gap;
                    pLeft = Math.min(vw - popW - 20, Math.max(20, sLeft + (sWidth / 2) - (popW / 2)));
                }} else if (cena.lado === 'left_corner') {{
                    sideUsed = 'left_corner';
                    pTop = Math.min(vh - popH - 22, Math.max(24, sTop + sHeight - popH - 18));
                    pLeft = Math.max(20, sLeft + 18);
                }} else if (sTop - popH - gap >= 20) {{
                    sideUsed = 'top';
                    pTop = sTop - popH - gap;
                    pLeft = Math.min(vw - popW - 20, Math.max(20, sLeft + sWidth - popW - 12));
                }} else {{
                    sideUsed = 'corner';
                    pTop = Math.min(vh - popH - 20, Math.max(20, sTop + sHeight - popH - 16));
                    pLeft = Math.min(vw - popW - 20, Math.max(20, sLeft + sWidth - popW - 16));
                }}

                pop.style.top = pTop + 'px';
                pop.style.left = pLeft + 'px';

                arrow.style.display = 'block';
                arrow.style.top = 'auto';
                arrow.style.bottom = 'auto';
                arrow.style.left = 'auto';
                arrow.style.right = 'auto';

                if (sideUsed === 'right') {{
                    arrow.style.left = '-7px';
                    arrow.style.top = '40px';
                    arrow.style.transform = 'rotate(-45deg)';
                }} else if (sideUsed === 'bottom') {{
                    const arrowX = Math.min(popW - 28, Math.max(22, (sLeft + sWidth / 2) - pLeft - 7));
                    arrow.style.top = '-7px';
                    arrow.style.left = arrowX + 'px';
                    arrow.style.transform = 'rotate(45deg)';
                }} else if (sideUsed === 'top') {{
                    const arrowX = Math.min(popW - 28, Math.max(22, (sLeft + sWidth * 0.72) - pLeft - 7));
                    arrow.style.bottom = '-7px';
                    arrow.style.left = arrowX + 'px';
                    arrow.style.transform = 'rotate(225deg)';
                }} else {{
                    arrow.style.display = 'none';
                }}
            }}

            if (cena.alvo === 'conteudo_aba' || cena.alvo === 'mapa_aba') {{
                const anchor = doc.getElementById('zeapi-aba-conteudo-anchor');
                if (anchor && anchor.scrollIntoView) {{
                    anchor.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
                }}
            }}

            setTimeout(alignElements, 140);
            setTimeout(alignElements, 520);
            setTimeout(alignElements, 950);
            win.addEventListener('resize', alignElements);
            win.addEventListener('scroll', alignElements, true);
        }})();
        </script>
        """,
        height=0,
        width=0
    )