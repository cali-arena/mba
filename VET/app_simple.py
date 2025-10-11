"""
VetDiagnosisAI - Sistema Inteligente de Apoio ao Diagnóstico Veterinário
Versão Simplificada para Deploy
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# Configuração da página
st.set_page_config(
    page_title="VetDiagnosisAI",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .dataset-link {
        padding: 10px;
        background-color: #f0f2f6;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Header principal
st.markdown('<div class="main-header">🐾 VetDiagnosisAI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Sistema Inteligente de Apoio ao Diagnóstico Veterinário</div>', unsafe_allow_html=True)

# Função para gerar dados sintéticos
@st.cache_data
def gerar_dados_sinteticos():
    """Gera dados sintéticos para demonstração"""
    np.random.seed(42)
    n_samples = 200
    
    # Criar dados sintéticos
    data = {
        'id': range(1, n_samples + 1),
        'especie': np.random.choice(['Cão', 'Gato', 'Ave'], n_samples, p=[0.6, 0.35, 0.05]),
        'raca': np.random.choice(['SRD', 'Pastor', 'Siames', 'Persa', 'Canário'], n_samples),
        'idade_anos': np.random.uniform(0.5, 18, n_samples).round(1),
        'sexo': np.random.choice(['M', 'F'], n_samples),
        'peso_kg': np.random.uniform(1, 50, n_samples).round(1),
        
        # Exames laboratoriais
        'hemoglobina': np.random.normal(12, 2, n_samples).round(1),
        'hematocrito': np.random.normal(40, 5, n_samples).round(1),
        'leucocitos': np.random.normal(8000, 2000, n_samples).round(0),
        'plaquetas': np.random.normal(300000, 50000, n_samples).round(0),
        'glicose': np.random.normal(100, 20, n_samples).round(1),
        'ureia': np.random.normal(30, 10, n_samples).round(1),
        'creatinina': np.random.normal(1.2, 0.3, n_samples).round(2),
        'alt': np.random.normal(40, 15, n_samples).round(1),
        'ast': np.random.normal(35, 12, n_samples).round(1),
        'proteinas_totais': np.random.normal(6.5, 1, n_samples).round(1),
        
        # Sinais vitais
        'temperatura_retal': np.random.normal(38.5, 0.5, n_samples).round(1),
        'pulso': np.random.normal(120, 20, n_samples).round(0),
        'freq_respiratoria': np.random.normal(20, 5, n_samples).round(0),
        
        # Sintomas (0 = não, 1 = sim)
        'febre': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'apatia': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'perda_peso': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'vomito': np.random.choice([0, 1], n_samples, p=[0.75, 0.25]),
        'diarreia': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'tosse': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'letargia': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'poliuria': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'polidipsia': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        
        # Diagnóstico
        'diagnostico': np.random.choice([
            'Normal', 'Infecção Respiratória', 'Doença Renal', 'Diabetes', 
            'Problema Gastrointestinal', 'Dermatite', 'Doença Hepática'
        ], n_samples, p=[0.4, 0.15, 0.1, 0.1, 0.1, 0.1, 0.05])
    }
    
    df = pd.DataFrame(data)
    return df

# Carregar dados
df = gerar_dados_sinteticos()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/veterinarian.png", width=100)
    st.title("🐾 VetDiagnosisAI")
    st.markdown("---")
    
    st.subheader("📊 Status dos Dados")
    st.success(f"✅ Dataset carregado: {len(df)} registros")
    st.info(f"📅 Espécies: {df['especie'].nunique()}")
    st.info(f"🏥 Diagnósticos: {df['diagnostico'].nunique()}")
    
    st.markdown("---")
    
    st.subheader("📋 Navegação")
    pagina = st.selectbox(
        "Escolha uma página:",
        [
            "🏠 Visão Geral",
            "📊 Análise de Dados",
            "🤖 Predição de Diagnóstico",
            "📈 Estatísticas"
        ]
    )

# Conteúdo principal baseado na página selecionada
if pagina == "🏠 Visão Geral":
    st.header("🏠 Visão Geral do Sistema")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total de Registros", len(df))
    
    with col2:
        st.metric("Espécies Únicas", df['especie'].nunique())
    
    with col3:
        st.metric("Diagnósticos Únicos", df['diagnostico'].nunique())
    
    st.markdown("---")
    
    st.subheader("📊 Distribuição por Espécie")
    especie_counts = df['especie'].value_counts()
    st.bar_chart(especie_counts)
    
    st.subheader("🏥 Distribuição de Diagnósticos")
    diag_counts = df['diagnostico'].value_counts()
    st.bar_chart(diag_counts)
    
    # Mostrar amostra dos dados
    st.subheader("📋 Amostra dos Dados")
    st.dataframe(df.head(10))

elif pagina == "📊 Análise de Dados":
    st.header("📊 Análise Exploratória dos Dados")
    
    # Filtros
    st.subheader("🔍 Filtros")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        especie_filtro = st.selectbox("Espécie:", ['Todas'] + list(df['especie'].unique()))
    
    with col2:
        idade_min, idade_max = st.slider("Faixa de Idade:", 0.0, 20.0, (0.0, 20.0))
    
    with col3:
        diag_filtro = st.selectbox("Diagnóstico:", ['Todos'] + list(df['diagnostico'].unique()))
    
    # Aplicar filtros
    df_filtrado = df.copy()
    if especie_filtro != 'Todas':
        df_filtrado = df_filtrado[df_filtrado['especie'] == especie_filtro]
    
    df_filtrado = df_filtrado[
        (df_filtrado['idade_anos'] >= idade_min) & 
        (df_filtrado['idade_anos'] <= idade_max)
    ]
    
    if diag_filtro != 'Todos':
        df_filtrado = df_filtrado[df_filtrado['diagnostico'] == diag_filtro]
    
    st.info(f"📊 Mostrando {len(df_filtrado)} registros após filtros")
    
    # Análises
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Exames Laboratoriais")
        
        exames = ['hemoglobina', 'hematocrito', 'leucocitos', 'glicose', 'ureia', 'creatinina']
        exame_selecionado = st.selectbox("Selecione um exame:", exames)
        
        # Box plot do exame selecionado
        st.bar_chart(df_filtrado.groupby('diagnostico')[exame_selecionado].mean())
    
    with col2:
        st.subheader("🌡️ Sinais Vitais")
        
        sinais = ['temperatura_retal', 'pulso', 'freq_respiratoria']
        sinal_selecionado = st.selectbox("Selecione um sinal vital:", sinais)
        
        st.bar_chart(df_filtrado.groupby('diagnostico')[sinal_selecionado].mean())
    
    # Correlações
    st.subheader("🔗 Correlações entre Variáveis")
    
    # Selecionar colunas numéricas para correlação
    colunas_numericas = df_filtrado.select_dtypes(include=[np.number]).columns.tolist()
    if len(colunas_numericas) > 1:
        correlacao = df_filtrado[colunas_numericas].corr()
        st.dataframe(correlacao)

elif pagina == "🤖 Predição de Diagnóstico":
    st.header("🤖 Predição de Diagnóstico")
    
    st.info("💡 Esta é uma demonstração. Para uma versão completa com ML, veja o sistema completo.")
    
    # Formulário para entrada de dados
    st.subheader("📝 Dados do Paciente")
    
    with st.form("form_diagnostico"):
        col1, col2 = st.columns(2)
        
        with col1:
            especie = st.selectbox("Espécie:", df['especie'].unique())
            raca = st.text_input("Raça:", "SRD")
            idade = st.number_input("Idade (anos):", 0.1, 25.0, 5.0)
            peso = st.number_input("Peso (kg):", 0.1, 100.0, 15.0)
            sexo = st.selectbox("Sexo:", ['M', 'F'])
        
        with col2:
            st.subheader("🔬 Exames Laboratoriais")
            hemoglobina = st.number_input("Hemoglobina:", 5.0, 20.0, 12.0)
            hematocrito = st.number_input("Hematócrito:", 20.0, 60.0, 40.0)
            leucocitos = st.number_input("Leucócitos:", 2000.0, 20000.0, 8000.0)
            glicose = st.number_input("Glicose:", 50.0, 300.0, 100.0)
            ureia = st.number_input("Ureia:", 10.0, 100.0, 30.0)
        
        st.subheader("🌡️ Sinais Vitais")
        col3, col4 = st.columns(3)
        
        with col3:
            temperatura = st.number_input("Temperatura (°C):", 35.0, 42.0, 38.5)
        
        with col4:
            pulso = st.number_input("Pulso (bpm):", 60.0, 200.0, 120.0)
        
        st.subheader("🚨 Sintomas")
        col5, col6 = st.columns(2)
        
        with col5:
            febre = st.checkbox("Febre")
            apatia = st.checkbox("Apatia")
            perda_peso = st.checkbox("Perda de Peso")
        
        with col6:
            vomito = st.checkbox("Vômito")
            diarreia = st.checkbox("Diarreia")
            tosse = st.checkbox("Tosse")
        
        submitted = st.form_submit_button("🔍 Analisar Diagnóstico")
        
        if submitted:
            # Simulação de predição (versão simplificada)
            st.success("✅ Dados recebidos! Processando...")
            
            # Lógica simples baseada em regras
            if febre and tosse:
                predicao = "Infecção Respiratória"
            elif ureia > 50:
                predicao = "Doença Renal"
            elif glicose > 150:
                predicao = "Diabetes"
            elif vomito and diarreia:
                predicao = "Problema Gastrointestinal"
            elif apatia and perda_peso:
                predicao = "Problema Sistêmico"
            else:
                predicao = "Normal"
            
            st.success(f"🎯 **Diagnóstico Predito:** {predicao}")
            
            # Mostrar estatísticas do diagnóstico predito
            if predicao in df['diagnostico'].values:
                casos_similares = df[df['diagnostico'] == predicao]
                st.info(f"📊 Encontramos {len(casos_similares)} casos similares no banco de dados")

elif pagina == "📈 Estatísticas":
    st.header("📈 Estatísticas Detalhadas")
    
    # Estatísticas gerais
    st.subheader("📊 Estatísticas Gerais")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Média de Idade", f"{df['idade_anos'].mean():.1f} anos")
        st.metric("Peso Médio", f"{df['peso_kg'].mean():.1f} kg")
        st.metric("Temperatura Média", f"{df['temperatura_retal'].mean():.1f}°C")
    
    with col2:
        st.metric("Taxa de Febre", f"{(df['febre'].sum() / len(df) * 100):.1f}%")
        st.metric("Taxa de Vômito", f"{(df['vomito'].sum() / len(df) * 100):.1f}%")
        st.metric("Taxa de Diarreia", f"{(df['diarreia'].sum() / len(df) * 100):.1f}%")
    
    # Distribuições
    st.subheader("📊 Distribuições")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Por Espécie")
        especie_stats = df.groupby('especie').agg({
            'idade_anos': 'mean',
            'peso_kg': 'mean',
            'febre': 'sum'
        }).round(2)
        st.dataframe(especie_stats)
    
    with col4:
        st.subheader("Por Diagnóstico")
        diag_stats = df.groupby('diagnostico').size().sort_values(ascending=False)
        st.bar_chart(diag_stats)
    
    # Download dos dados
    st.subheader("📥 Download")
    
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📊 Baixar Dataset Completo (CSV)",
        data=csv,
        file_name='veterinary_data.csv',
        mime='text/csv',
    )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🐾 VetDiagnosisAI - Sistema de Apoio ao Diagnóstico Veterinário</p>
    <p>Desenvolvido para o MBA - Versão Simplificada para Demonstração</p>
</div>
""", unsafe_allow_html=True)
