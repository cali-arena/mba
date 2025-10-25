import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# ARQUIVO COMPLETAMENTE NOVO - SEM SCATTER PLOT PROBLEMÁTICO
# VERSÃO: 2025-10-25 - CORREÇÃO CRÍTICA

# Configuração da página
st.set_page_config(
    page_title="🧪 Laboratório & Sintomas (EDA)",
    page_icon="🧪",
    layout="wide"
)

# CSS personalizado
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .stSelectbox > div > div {
        background-color: #262730;
    }
    .stSlider > div > div > div > div {
        background-color: #00d4aa;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown("# 🧪 Laboratório & Sintomas (EDA)")
st.markdown("**Análise Exploratória de Dados Veterinários**")

# Função para carregar dados
@st.cache_data
def load_data():
    """Carrega os dados veterinários"""
    try:
        # Tentar carregar dados reais primeiro
        df = pd.read_csv('data/veterinary_complete_real_dataset.csv')
        st.success("✅ Dados reais carregados com sucesso!")
    except:
        try:
            # Fallback para dados sintéticos
            df = pd.read_csv('data/veterinary_realistic_dataset.csv')
            st.info("ℹ️ Usando dados sintéticos (dados reais não encontrados)")
        except:
            # Criar dados de exemplo se nenhum arquivo for encontrado
            st.warning("⚠️ Criando dados de exemplo...")
            np.random.seed(42)
            n_samples = 1000
            
            df = pd.DataFrame({
                'idade_anos': np.random.normal(5, 3, n_samples),
                'peso_kg': np.random.normal(15, 8, n_samples),
                'especie': np.random.choice(['Cão', 'Gato', 'Coelho'], n_samples),
                'sexo': np.random.choice(['M', 'F'], n_samples),
                'hemoglobina': np.random.normal(12, 3, n_samples),
                'hematocrito': np.random.normal(40, 8, n_samples),
                'leucocitos': np.random.normal(8, 3, n_samples),
                'glicose': np.random.normal(100, 30, n_samples),
                'ureia': np.random.normal(30, 15, n_samples),
                'creatinina': np.random.normal(1.0, 0.5, n_samples),
                'diagnostico': np.random.choice(['Saudável', 'Dermatite', 'Gastrite', 'Infecção'], n_samples)
            })
    
    return df

# Carregar dados
df = load_data()

# Schema das colunas
SCHEMA_COLUNAS = {
    'demograficas': ['idade_anos', 'peso_kg', 'especie', 'sexo', 'raca'],
    'exames': ['hemoglobina', 'hematocrito', 'leucocitos', 'glicose', 'ureia', 'creatinina'],
    'sintomas': ['febre', 'letargia', 'anorexia', 'vomito', 'diarreia'],
    'diagnostico': ['diagnostico']
}

# Sidebar com filtros
st.sidebar.markdown("## 🔍 Filtros")

# Filtro por espécie
especies_disponiveis = ['Todas'] + list(df['especie'].unique()) if 'especie' in df.columns else ['Todas']
especie_filtro = st.sidebar.selectbox("Espécie:", especies_disponiveis)

# Filtro por raça
racas_disponiveis = ['Todas'] + list(df['raca'].unique()) if 'raca' in df.columns else ['Todas']
raca_filtro = st.sidebar.selectbox("Raça:", racas_disponiveis)

# Filtro por sexo
sexos_disponiveis = ['Todos'] + list(df['sexo'].unique()) if 'sexo' in df.columns else ['Todos']
sexo_filtro = st.sidebar.selectbox("Sexo:", sexos_disponiveis)

# Filtro por idade
if 'idade_anos' in df.columns:
    idade_min, idade_max = st.sidebar.slider(
        "Faixa Etária (anos):",
        min_value=float(df['idade_anos'].min()),
        max_value=float(df['idade_anos'].max()),
        value=(float(df['idade_anos'].min()), float(df['idade_anos'].max()))
else:
    idade_min, idade_max = 0, 20

# Filtro por diagnóstico
diagnosticos_disponiveis = ['Todos'] + list(df['diagnostico'].unique()) if 'diagnostico' in df.columns else ['Todos']
diagnostico_filtro = st.sidebar.selectbox("Diagnóstico:", diagnosticos_disponiveis)

# Aplicar filtros
df_filtrado = df.copy()

if especie_filtro != 'Todas':
    df_filtrado = df_filtrado[df_filtrado['especie'] == especie_filtro]

if raca_filtro != 'Todas':
    df_filtrado = df_filtrado[df_filtrado['raca'] == raca_filtro]

if sexo_filtro != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['sexo'] == sexo_filtro]

if 'idade_anos' in df_filtrado.columns:
    df_filtrado = df_filtrado[
        (df_filtrado['idade_anos'] >= idade_min) & 
        (df_filtrado['idade_anos'] <= idade_max)
    ]

if diagnostico_filtro != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['diagnostico'] == diagnostico_filtro]

# Mostrar estatísticas dos filtros
st.markdown(f"**📊 Dados filtrados: {len(df_filtrado)} de {len(df)} registros**")

# Seleção do tipo de análise
st.markdown("## 📈 Tipo de Análise")
analise_tipo = st.selectbox(
    "Escolha o tipo de análise:",
    ["Visão Geral", "Distribuições", "Correlações", "Outliers"]
)

# ============================================================================
# ANÁLISE: VISÃO GERAL
# ============================================================================

if analise_tipo == "Visão Geral":
    st.markdown("## 📊 Visão Geral dos Dados")
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Registros", len(df_filtrado))
    
    with col2:
        if 'especie' in df_filtrado.columns:
            especies_unicas = df_filtrado['especie'].nunique()
            st.metric("Espécies", especies_unicas)
        else:
            st.metric("Espécies", "N/A")
    
    with col3:
        if 'diagnostico' in df_filtrado.columns:
            diagnosticos_unicos = df_filtrado['diagnostico'].nunique()
            st.metric("Diagnósticos", diagnosticos_unicos)
        else:
            st.metric("Diagnósticos", "N/A")
    
    with col4:
        if 'idade_anos' in df_filtrado.columns:
            idade_media = df_filtrado['idade_anos'].mean()
            st.metric("Idade Média", f"{idade_media:.1f} anos")
        else:
            st.metric("Idade Média", "N/A")
    
    # Distribuição por espécie
    if 'especie' in df_filtrado.columns:
        st.markdown("### 🐕 Distribuição por Espécie")
        especie_counts = df_filtrado['especie'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(values=especie_counts.values, names=especie_counts.index, 
                        title="Distribuição por Espécie")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(x=especie_counts.index, y=especie_counts.values,
                        title="Contagem por Espécie")
            st.plotly_chart(fig, use_container_width=True)
    
    # Top diagnósticos
    if 'diagnostico' in df_filtrado.columns:
        st.markdown("### 🏥 Top 10 Diagnósticos")
        diag_counts = df_filtrado['diagnostico'].value_counts().head(10)
        
        fig = px.bar(x=diag_counts.values, y=diag_counts.index,
                    orientation='h', title="Top 10 Diagnósticos")
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# ANÁLISE: DISTRIBUIÇÕES
# ============================================================================

elif analise_tipo == "Distribuições":
    st.markdown("## 📊 Análise de Distribuições")
    
    # Selecionar variável
    colunas_numericas = df_filtrado.select_dtypes(include=[np.number]).columns.tolist()
    
    if not colunas_numericas:
        st.warning("⚠️ Nenhuma variável numérica encontrada.")
        st.stop()
    
    col1, col2 = st.columns(2)
    
    with col1:
        variavel = st.selectbox("Selecione a variável:", colunas_numericas)
    
    with col2:
        tipo_grafico = st.selectbox("Tipo de gráfico:", ["Histograma", "Box Plot"])
    
    # Estatísticas descritivas
    st.markdown(f"### 📈 Estatísticas de {variavel.replace('_', ' ').title()}")
    
    stats_data = df_filtrado[variavel].describe()
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Média", f"{stats_data['mean']:.2f}")
    with col2:
        st.metric("Mediana", f"{stats_data['50%']:.2f}")
    with col3:
        st.metric("Desvio Padrão", f"{stats_data['std']:.2f}")
    with col4:
        st.metric("Coeficiente de Variação", f"{(stats_data['std']/stats_data['mean']*100):.1f}%")
    
    # Gráfico
    if tipo_grafico == "Histograma":
        fig = px.histogram(df_filtrado, x=variavel, 
                          title=f'Distribuição de {variavel.replace("_", " ").title()}',
                          nbins=30)
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    else:  # Box Plot
        fig = px.box(df_filtrado, y=variavel,
                    title=f'Box Plot de {variavel.replace("_", " ").title()}')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Teste de normalidade
    st.markdown("### 🔬 Teste de Normalidade")
    
    if len(df_filtrado[variavel].dropna()) > 3:
        shapiro_stat, shapiro_p = stats.shapiro(df_filtrado[variavel].dropna())
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Estatística Shapiro-Wilk", f"{shapiro_stat:.4f}")
        with col2:
            st.metric("P-valor", f"{shapiro_p:.4f}")
        
        if shapiro_p > 0.05:
            st.success("✅ Dados seguem distribuição normal (p > 0.05)")
        else:
            st.warning("⚠️ Dados não seguem distribuição normal (p ≤ 0.05)")

# ============================================================================
# ANÁLISE: CORRELAÇÕES
# ============================================================================

elif analise_tipo == "Correlações":
    st.markdown("## 🔗 Análise de Correlações")
    
    # Selecionar variáveis numéricas
    colunas_numericas = df_filtrado.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(colunas_numericas) < 2:
        st.warning("⚠️ É necessário pelo menos 2 variáveis numéricas para análise de correlação.")
        st.stop()
    
    # Matriz de correlação
    corr_matrix = df_filtrado[colunas_numericas].corr()
    
    # Heatmap de correlação
    fig = px.imshow(corr_matrix, 
                    text_auto=True, 
                    aspect="auto",
                    title="Matriz de Correlação",
                    color_continuous_scale="RdBu_r")
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Top correlações
    st.markdown("### 🔥 Top 15 Correlações")
    
    # Extrair correlações
    correlacoes = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            var1 = corr_matrix.columns[i]
            var2 = corr_matrix.columns[j]
            corr_value = corr_matrix.iloc[i, j]
            correlacoes.append({
                'Variável 1': var1,
                'Variável 2': var2,
                'Correlação': corr_value
            })
    
    df_correlacoes = pd.DataFrame(correlacoes)
    df_correlacoes['Abs Correlação'] = df_correlacoes['Correlação'].abs()
    df_correlacoes = df_correlacoes.sort_values('Abs Correlação', ascending=False).head(15)
    
    st.dataframe(
        df_correlacoes[['Variável 1', 'Variável 2', 'Correlação']].style.format({'Correlação': '{:.3f}'}),
        use_container_width=True,
        hide_index=True
    )

# ============================================================================
# ANÁLISE: OUTLIERS
# ============================================================================

elif analise_tipo == "Outliers":
    st.markdown("## 🎯 Detecção de Outliers")
    
    st.markdown("""
    Outliers são valores que se desviam significativamente da maioria dos dados.
    Podem indicar:
    - ⚠️ Erros de medição ou registro
    - 🔬 Casos clínicos extremos que requerem atenção
    - 📊 Variabilidade natural em casos específicos
    """)
    
    # Selecionar exame
    exames_disponiveis = [col for col in SCHEMA_COLUNAS['exames'] if col in df.columns]
    
    if not exames_disponiveis:
        st.warning("⚠️ Nenhum exame laboratorial encontrado.")
        st.stop()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        exame_outlier = st.selectbox("Selecione o exame:", exames_disponiveis)
    
    with col2:
        metodo = st.selectbox("Método de detecção:", ["IQR (Interquartile Range)", "Z-Score"])
    
    # Detectar outliers usando IQR
    def identificar_outliers(df, coluna, metodo='iqr'):
        """Identifica outliers usando IQR ou Z-Score"""
        if metodo == 'iqr':
            Q1 = df[coluna].quantile(0.25)
            Q3 = df[coluna].quantile(0.75)
            IQR = Q3 - Q1
            limite_inferior = Q1 - 1.5 * IQR
            limite_superior = Q3 + 1.5 * IQR
            return (df[coluna] < limite_inferior) | (df[coluna] > limite_superior)
        else:  # zscore
            z_scores = np.abs(stats.zscore(df[coluna].dropna()))
            return z_scores > 3
    
    # Detectar outliers
    metodo_param = 'iqr' if 'IQR' in metodo else 'zscore'
    mask_outliers = identificar_outliers(df_filtrado, exame_outlier, metodo=metodo_param)
    
    df_outliers = df_filtrado[mask_outliers].copy()
    df_normais = df_filtrado[~mask_outliers].copy()
    
    # Métricas
    total_registros = len(df_filtrado)
    outliers_count = len(df_outliers)
    percentual_outliers = (outliers_count / total_registros) * 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total de Registros", total_registros)
    
    with col2:
        st.metric("Outliers Detectados", outliers_count)
    
    with col3:
        st.metric("Percentual de Outliers", f"{percentual_outliers:.1f}%")
    
    # Box plot com outliers
    fig = go.Figure()
    
    fig.add_trace(go.Box(
        y=df_normais[exame_outlier],
        name='Valores Normais',
        boxmean='sd'
    ))
    
    # Outliers
    if len(df_outliers) > 0:
        fig.add_trace(go.Scatter(
            y=df_outliers[exame_outlier],
            mode='markers',
            name='Outliers',
            marker=dict(color='red', size=10, symbol='x')
        ))
    
    fig.update_layout(
        title=f'Detecção de Outliers - {exame_outlier.replace("_", " ").title()}',
        yaxis_title=exame_outlier.replace('_', ' ').title(),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabela de outliers
    if len(df_outliers) > 0:
        st.markdown("### 📋 Registros com Outliers")
        
        colunas_mostrar = ['idade_anos', 'especie', 'sexo', exame_outlier, 'diagnostico']
        colunas_disponiveis = [col for col in colunas_mostrar if col in df_outliers.columns]
        
        st.dataframe(
            df_outliers[colunas_disponiveis].head(20),
            use_container_width=True,
            hide_index=True
        )
        
        # Download dos outliers
        csv_outliers = df_outliers.to_csv(index=False)
        st.download_button(
            label="📥 Download Outliers (CSV)",
            data=csv_outliers,
            file_name=f"outliers_{exame_outlier}.csv",
            mime="text/csv"
        )
    else:
        st.success("✅ Nenhum outlier detectado para esta variável!")

# Rodapé
st.markdown("---")
st.markdown("**🔬 Sistema de Análise Veterinária - EDA Completo**")
st.markdown("*Desenvolvido para análise exploratória de dados clínicos veterinários*")
