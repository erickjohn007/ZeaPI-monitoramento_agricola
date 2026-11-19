"""
Pipeline ETL e Treinamento Preditivo Agroclimático do Piauí.
================================================================================
Módulo de Processamento de Dados (Backend)

Este módulo realiza a ingestão, limpeza, enriquecimento, unificação e modelagem
estatística dos dados de safra agrícola (IBGE PAM) e clima (NASA POWER) para
todos os 224 municípios do estado do Piauí entre 2000 e 2024.

Fluxo do Pipeline:
  1. Leitura tolerante de CSVs (detecção de BOM UTF-8 e separadores ; ou ,).
  2. Normalização de dados numéricos (vírgula decimal brasileira vs ponto americano).
  3. Imputação de indicadores derivados (rendimento em kg/ha e taxa de perda de área).
  4. Agregação e fusão relacional (Inner Join por chave composta: [cod_ibge, ano]).
  5. Classificação probabilística de risco climático com regras de negócio agroclimáticas.
  6. Treinamento de modelo Random Forest para predição de produtividade.
  7. Exportação em formato colunar Apache Parquet, metadados JSON e modelo Joblib.
"""

import os
import json
import logging
from typing import List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ==============================================================================
# CONFIGURAÇÃO DE LOGS E AMBIENTE
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("ZeaPI-ETL")

# Resolução de diretórios absolutos a partir da raiz do repositório
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


def ler_csv_limpo(caminho_arquivo: str) -> pd.DataFrame:
    """Detecta a codificação e o delimitador do CSV, limpando os nomes das colunas.

    Lê os primeiros bytes do arquivo para identificar se o separador utilizado
    é ponto-e-vírgula (padrão exportado no Excel/IBGE brasileiro) ou vírgula
    (padrão internacional/NASA POWER). Remove marcas BOM (Byte Order Mark) e
    espaços residuais nos cabeçalhos.

    Args:
        caminho_arquivo: Caminho completo para o arquivo CSV no disco.

    Returns:
        pd.DataFrame: DataFrame com colunas limpas em formato ASCII.

    Raises:
        FileNotFoundError: Se o arquivo especificado não existir.
    """
    if not os.path.exists(caminho_arquivo):
        logger.error("Arquivo não encontrado no caminho: %s", caminho_arquivo)
        raise FileNotFoundError(f"Arquivo ausente: {caminho_arquivo}")

    # Amostragem da primeira linha em formato binário para inspeção do separador
    with open(caminho_arquivo, "rb") as f:
        primeira_linha = f.readline().decode("utf-8-sig", errors="replace")
    separador = ";" if ";" in primeira_linha else ","

    df = pd.read_csv(caminho_arquivo, sep=separador, encoding="utf-8-sig")

    # Normalização dos cabeçalhos: remove acentos e espaços laterais
    df.columns = [
        col.encode("ascii", "ignore").decode("ascii").strip()
        for col in df.columns
    ]

    # Padronização da chave relacional do IBGE
    if "codigo_ibge" in df.columns:
        df = df.rename(columns={"codigo_ibge": "cod_ibge"})

    return df


def converter_numero(serie: pd.Series) -> pd.Series:
    """Converte valores numéricos formatados em string para ponto flutuante.

    Trata formatos de números brasileiros (onde '.' é milhar e ',' é decimal)
    e converte valores ausentes ou corrompidos para NaN via pd.to_numeric.

    Args:
        serie: Coluna do pandas contendo valores numéricos ou strings.

    Returns:
        pd.Series: Série numérica tipada em float64.
    """
    if serie.dtype == object:
        texto = serie.astype(str).str.strip()
        tem_virgula = texto.str.contains(",", na=False)
        # Substitui ponto de milhar por vazio e vírgula decimal por ponto
        texto_limpo = np.where(
            tem_virgula,
            texto.str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
            texto
        )
        return pd.to_numeric(texto_limpo, errors="coerce")
    return pd.to_numeric(serie, errors="coerce")


def processar_dados_ibge(caminho_ibge: str) -> pd.DataFrame:
    """Processa e valida a base de safras do milho (IBGE PAM).

    Aplica filtros de ano (2000 a 2024), sanitiza códigos municipais,
    deriva rendimento médio ausente a partir de produção/área colhida
    e calcula a taxa de perda percentual de safra.

    Args:
        caminho_ibge: Caminho do CSV de dados agrícolas do IBGE.

    Returns:
        pd.DataFrame: Base do IBGE limpa e enriquecida.
    """
    logger.info("Carregando e sanitizando base agrícola do IBGE...")
    df_ibge = ler_csv_limpo(caminho_ibge)

    # Chave IBGE como string sem decimais residuais
    df_ibge["cod_ibge"] = (
        df_ibge["cod_ibge"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
    df_ibge["ano"] = pd.to_numeric(df_ibge["ano"], errors="coerce").fillna(0).astype(int)

    # Limpeza dos indicadores quantitativos
    cols_numericas = ["area_plantada_ha", "area_colhida_ha", "producao_ton", "rendimento_kg_ha"]
    for col in cols_numericas:
        if col in df_ibge.columns:
            df_ibge[col] = converter_numero(df_ibge[col]).fillna(0.0)

    # Regra de negócio 1: Se o rendimento estiver ausente, recalcula: (Produção * 1000) / Área Colhida
    df_ibge["rendimento_kg_ha"] = np.where(
        df_ibge["rendimento_kg_ha"] > 0,
        df_ibge["rendimento_kg_ha"],
        np.where(
            df_ibge["area_colhida_ha"] > 0,
            (df_ibge["producao_ton"] * 1000.0) / df_ibge["area_colhida_ha"],
            0.0
        )
    )

    # Regra de negócio 2: Taxa de perda mede o percentual de área plantada que não chegou a ser colhida
    df_ibge["taxa_perda_pct"] = np.where(
        df_ibge["area_plantada_ha"] > 0,
        np.clip(
            ((df_ibge["area_plantada_ha"] - df_ibge["area_colhida_ha"]) / df_ibge["area_plantada_ha"]) * 100.0,
            0.0,
            100.0
        ),
        0.0
    )

    # Restrição de série temporal do projeto (2000 a 2024)
    df_ibge = df_ibge[(df_ibge["ano"] >= 2000) & (df_ibge["ano"] <= 2024)].copy()
    return df_ibge


def processar_dados_nasa(caminho_nasa: str) -> pd.DataFrame:
    """Processa e agrega a base climatológica de superfície da NASA POWER.

    Padroniza nomes de colunas agrometeorológicas e agrega registros
    pela média anual de cada município.

    Args:
        caminho_nasa: Caminho do CSV climatológico da NASA POWER.

    Returns:
        pd.DataFrame: Base climática agregada por município e ano.
    """
    logger.info("Carregando e agregando base climatológica NASA POWER...")
    df_nasa = ler_csv_limpo(caminho_nasa)

    df_nasa["cod_ibge"] = (
        df_nasa["cod_ibge"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
    df_nasa["ano"] = pd.to_numeric(df_nasa["ano"], errors="coerce").fillna(0).astype(int)

    renomear_map = {
        "chuvas_total_mm": "precipitacao_anual_mm",
        "temp_max_media_c": "temp_max_c",
        "temp_min_media_c": "temp_min_c"
    }
    df_nasa = df_nasa.rename(columns={k: v for k, v in renomear_map.items() if k in df_nasa.columns})

    cols_clima = ["precipitacao_anual_mm", "temp_media_c", "temp_max_c", "temp_min_c", "graus_dia_base10"]
    for col in cols_clima:
        if col in df_nasa.columns:
            df_nasa[col] = converter_numero(df_nasa[col])

    # Agregação anual por município
    cols_existentes = [c for c in cols_clima if c in df_nasa.columns]
    agg_dict = {col: "mean" for col in cols_existentes}
    return df_nasa.groupby(["cod_ibge", "ano"], as_index=False).agg(agg_dict)


def classificar_risco_climatico(df: pd.DataFrame) -> pd.Series:
    """Classifica o perfil de risco agroclimático com base em estresse hídrico e perda.

    Critérios Agronômicos:
      - Alto Risco: Precipitação < 650 mm OU Perda de Safra > 25%.
      - Risco Moderado: Precipitação entre 650 mm e 900 mm OU Perda de Safra entre 10% e 25%.
      - Baixo Risco: Precipitação > 900 mm e Perda de Safra < 10%.

    Args:
        df: DataFrame contendo 'precipitacao_anual_mm' e 'taxa_perda_pct'.

    Returns:
        pd.Series: Rótulo de classificação ('Alto Risco', 'Risco Moderado', 'Baixo Risco').
    """
    condicoes = [
        (df["precipitacao_anual_mm"] < 650) | (df["taxa_perda_pct"] > 25),
        (df["precipitacao_anual_mm"].between(650, 900)) | (df["taxa_perda_pct"].between(10, 25))
    ]
    escolhas = ["Alto Risco", "Risco Moderado"]
    return pd.Series(np.select(condicoes, escolhas, default="Baixo Risco"), index=df.index)


def treinar_modelo_produtividade(
    df: pd.DataFrame,
    features: List[str]
) -> Tuple[RandomForestRegressor, dict]:
    """Treina um modelo Random Forest para predição do rendimento agrícola (kg/ha).

    Realiza holdout validation (80% treino, 20% teste) e extrai métricas
    de interpretabilidade (R², RMSE e Feature Importance).

    Args:
        df: DataFrame consolidado com features e target.
        features: Lista de nomes de variáveis independentes preditivas.

    Returns:
        Tuple contendo o estimador treinado e o dicionário de metadados.
    """
    logger.info("Iniciando treinamento do modelo Random Forest Regressor...")
    df_ml = df.dropna(subset=features + ["rendimento_kg_ha"])

    X = df_ml[features]
    y = df_ml["rendimento_kg_ha"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    modelo = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )
    modelo.fit(X_train, y_train)

    predicoes = modelo.predict(X_test)
    rmse = round(float(np.sqrt(mean_squared_error(y_test, predicoes))), 2)
    r2 = round(float(r2_score(y_test, predicoes)), 4)

    metadados = {
        "metricas_modelo": {
            "rmse_kg_ha": rmse,
            "r2_score": r2,
            "algoritmo": "RandomForestRegressor (100 árvores, max_depth=8)"
        },
        "importancia_variaveis": {
            feat: round(float(imp), 4)
            for feat, imp in zip(features, modelo.feature_importances_)
        },
        "features": features
    }

    logger.info("Modelo avaliado com sucesso: R² = %.2f%% | RMSE = %.2f kg/ha", r2 * 100, rmse)
    return modelo, metadados


def executar_pipeline(caminho_ibge: str, caminho_nasa: str) -> None:
    """Executa o pipeline ETL ponta a ponta e persiste artefatos no disco."""
    logger.info("=== INÍCIO DO PIPELINE DE DADOS ZEAPI ===")

    # 1. Processamento das fontes primárias
    df_ibge = processar_dados_ibge(caminho_ibge)
    df_nasa = processar_dados_nasa(caminho_nasa)

    # 2. Junção relacional interna
    logger.info("Cruzando safras do IBGE com histórico climatológico da NASA...")
    df_final = pd.merge(df_ibge, df_nasa, on=["cod_ibge", "ano"], how="inner")

    # 3. Classificação de Risco
    df_final["nivel_risco_climatico"] = classificar_risco_climatico(df_final)

    # 4. Exportação colunar Parquet
    caminho_parquet = os.path.join(PROCESSED_DIR, "agroclima_consolidado.parquet")
    df_final.to_parquet(caminho_parquet, index=False, engine="pyarrow")
    logger.info("Base consolidada gravada em: %s (Total: %d registros)", caminho_parquet, len(df_final))

    # 5. Modelagem Preditiva
    candidatas = [
        "precipitacao_anual_mm", "temp_media_c", "temp_max_c",
        "temp_min_c", "graus_dia_base10", "area_plantada_ha"
    ]
    features_presentes = [c for c in candidatas if c in df_final.columns]

    modelo, metadados = treinar_modelo_produtividade(df_final, features_presentes)

    # 6. Persistência dos Metadados e do Modelo Treinado (Joblib)
    caminho_meta = os.path.join(PROCESSED_DIR, "metadados_modelo.json")
    with open(caminho_meta, "w", encoding="utf-8") as f:
        json.dump(metadados, f, indent=4, ensure_ascii=False)

    caminho_modelo = os.path.join(PROCESSED_DIR, "modelo_rf.joblib")
    joblib.dump(modelo, caminho_modelo)
    logger.info("Metadados e modelo preditivo salvos em: %s e %s", caminho_meta, caminho_modelo)
    logger.info("=== PIPELINE EXECUTADO COM SUCESSO ===")


if __name__ == "__main__":
    ibge_csv = os.path.join(RAW_DIR, "ibge_milho_piaui_2000_2024.csv")
    nasa_csv = os.path.join(RAW_DIR, "nasa_power_piaui_2000_2024.csv")
    executar_pipeline(ibge_csv, nasa_csv)
