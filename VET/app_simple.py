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

# Função para carregar datasets reais
@st.cache_data(ttl=3600)  # Cache por 1 hora
def carregar_dataset_completo():
    """Carrega o dataset completo da pasta data"""
    try:
        # Tentar carregar datasets da pasta data
        data_path = Path("data")
        csv_files = list(data_path.glob("*.csv")) if data_path.exists() else []
        
        if csv_files:
            # Priorizar datasets específicos
            datasets_prioritarios = [
                'veterinary_complete_real_dataset.csv',
                'veterinary_master_dataset.csv', 
                'veterinary_realistic_dataset.csv',
                'clinical_veterinary_data.csv',
                'laboratory_complete_panel.csv'
            ]
            
            dataset_escolhido = None
            
            # Procurar por dataset prioritário
            for dataset in datasets_prioritarios:
                if Path(data_path / dataset).exists():
                    dataset_escolhido = data_path / dataset
                    break
            
            # Se não encontrar prioritário, usar o primeiro disponível
            if not dataset_escolhido:
                dataset_escolhido = csv_files[0]
            
            # Carregar o dataset
            df = pd.read_csv(dataset_escolhido)
            
            # Limpar e preparar dados
            df = df.dropna(how='all')  # Remover linhas completamente vazias
            
            # Renomear colunas se necessário
            if 'diagnostico' not in df.columns:
                for col in df.columns:
                    if 'diagnos' in col.lower() or 'outcome' in col.lower():
                        df = df.rename(columns={col: 'diagnostico'})
                        break
            
            # Garantir que temos pelo menos algumas colunas básicas
            colunas_necessarias = ['id', 'especie', 'diagnostico']
            colunas_faltando = [col for col in colunas_necessarias if col not in df.columns]
            
            if colunas_faltando:
                st.warning(f"⚠️ Colunas faltando: {colunas_faltando}")
                
                # Criar colunas básicas se não existirem
                if 'id' not in df.columns:
                    df['id'] = range(1, len(df) + 1)
                if 'especie' not in df.columns:
                    df['especie'] = 'Cão'  # Default
                if 'diagnostico' not in df.columns:
                    df['diagnostico'] = 'Normal'  # Default
            
            st.success(f"📊 Dataset carregado: {len(df)} registros, {len(df.columns)} colunas")
            return df
        
        else:
            st.warning("⚠️ Nenhum arquivo CSV encontrado na pasta data")
            return gerar_dados_sinteticos()
            
    except Exception as e:
        st.error(f"❌ Erro ao carregar dataset: {str(e)}")
        st.info("🔄 Usando dados sintéticos como fallback")
        return gerar_dados_sinteticos()

def gerar_dados_sinteticos():
    """Gera dados sintéticos como fallback"""
    np.random.seed(42)
    n_samples = 500  # Mais dados sintéticos
    
    # Criar dados sintéticos mais realistas
    data = {
        'id': range(1, n_samples + 1),
        'especie': np.random.choice(['Cão', 'Gato', 'Ave', 'Equino'], n_samples, p=[0.5, 0.35, 0.1, 0.05]),
        'raca': np.random.choice(['SRD', 'Pastor Alemão', 'Golden Retriever', 'Siames', 'Persa', 'Canário', 'Puro Sangue'], n_samples),
        'idade_anos': np.random.uniform(0.5, 20, n_samples).round(1),
        'sexo': np.random.choice(['M', 'F'], n_samples),
        'peso_kg': np.random.uniform(1, 80, n_samples).round(1),
        
        # Exames laboratoriais completos
        'hemoglobina': np.random.normal(12, 2, n_samples).round(1),
        'hematocrito': np.random.normal(40, 5, n_samples).round(1),
        'leucocitos': np.random.normal(8000, 2000, n_samples).round(0),
        'plaquetas': np.random.normal(300000, 50000, n_samples).round(0),
        'eritrocitos': np.random.normal(6, 1, n_samples).round(2),
        'neutrofilos': np.random.normal(65, 10, n_samples).round(1),
        'linfocitos': np.random.normal(25, 8, n_samples).round(1),
        'monocitos': np.random.normal(5, 2, n_samples).round(1),
        'eosinofilos': np.random.normal(3, 1.5, n_samples).round(1),
        
        # Bioquímica
        'glicose': np.random.normal(100, 20, n_samples).round(1),
        'ureia': np.random.normal(30, 10, n_samples).round(1),
        'creatinina': np.random.normal(1.2, 0.3, n_samples).round(2),
        'alt': np.random.normal(40, 15, n_samples).round(1),
        'ast': np.random.normal(35, 12, n_samples).round(1),
        'fosfatase_alcalina': np.random.normal(120, 30, n_samples).round(1),
        'fa': np.random.normal(80, 20, n_samples).round(1),
        'ggt': np.random.normal(25, 10, n_samples).round(1),
        'proteinas_totais': np.random.normal(6.5, 1, n_samples).round(1),
        'albumina': np.random.normal(3.5, 0.5, n_samples).round(1),
        'globulinas': np.random.normal(3.0, 0.8, n_samples).round(1),
        'colesterol': np.random.normal(180, 40, n_samples).round(1),
        'triglicerideos': np.random.normal(100, 30, n_samples).round(1),
        'bilirrubina_total': np.random.normal(0.5, 0.2, n_samples).round(2),
        'calcio': np.random.normal(10, 1, n_samples).round(1),
        'fosforo': np.random.normal(4.5, 1, n_samples).round(1),
        'sodio': np.random.normal(145, 5, n_samples).round(1),
        'potassio': np.random.normal(4.5, 0.5, n_samples).round(1),
        
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
        'feridas_cutaneas': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'poliuria': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'polidipsia': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'dor': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'cirurgia': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
        
        # Diagnóstico mais realista
        'diagnostico': np.random.choice([
            'Normal', 'Infecção Respiratória', 'Doença Renal', 'Diabetes', 
            'Problema Gastrointestinal', 'Dermatite', 'Doença Hepática',
            'Anemia', 'Leucemia', 'Insuficiência Cardíaca', 'Tumor',
            'Parasitose', 'Alergia', 'Fratura', 'Doença Infecciosa'
        ], n_samples, p=[0.3, 0.12, 0.08, 0.08, 0.08, 0.06, 0.05, 0.05, 0.03, 0.03, 0.03, 0.04, 0.03, 0.02, 0.02])
    }
    
    df = pd.DataFrame(data)
    return df

# Carregar dados automaticamente SEMPRE
df = carregar_dataset_completo()

# Verificar se os dados foram carregados
if df is None or len(df) == 0:
    st.error("❌ Erro ao carregar dados. Recarregue a página.")
    st.stop()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/veterinarian.png", width=100)
    st.title("🐾 VetDiagnosisAI")
    st.markdown("---")
    
    st.subheader("📊 Status dos Dados")
    st.success(f"✅ Dataset carregado: {len(df)} registros")
    st.info(f"📅 Colunas: {len(df.columns)}")
    
    # Verificar se as colunas existem antes de acessá-las
    if 'especie' in df.columns:
        st.info(f"🐾 Espécies: {df['especie'].nunique()}")
    if 'diagnostico' in df.columns:
        st.info(f"🏥 Diagnósticos: {df['diagnostico'].nunique()}")
    
    # Mostrar informações sobre o dataset
    if hasattr(df, 'name') or 'veterinary' in str(df.columns):
        st.success("📁 Dataset real carregado")
    else:
        st.info("🔄 Usando dados sintéticos")
    
    # Mostrar primeiras colunas
    st.write("**Colunas principais:**")
    colunas_principais = [col for col in df.columns[:10]]
    for col in colunas_principais:
        st.write(f"• {col}")
    
    if len(df.columns) > 10:
        st.write(f"... e mais {len(df.columns) - 10} colunas")
    
    st.markdown("---")
    
    # Botão para recarregar dados
    if st.button("🔄 Recarregar Dataset"):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    
    st.subheader("📋 Navegação")
    pagina = st.selectbox(
        "Escolha uma página:",
        [
            "🏠 Visão Geral",
            "📊 Análise de Dados", 
            "🤖 Predição de Diagnóstico",
            "📈 Estatísticas",
            "📁 Informações do Dataset"
        ]
    )

# Conteúdo principal baseado na página selecionada
if pagina == "🏠 Visão Geral":
    st.header("🏠 Visão Geral do Sistema")
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total de Registros", len(df))
    
    with col2:
        st.metric("🐾 Espécies Únicas", df['especie'].nunique())
    
    with col3:
        st.metric("🏥 Diagnósticos Únicos", df['diagnostico'].nunique())
    
    with col4:
        st.metric("🔬 Exames Disponíveis", len([col for col in df.columns if col not in ['id', 'especie', 'raca', 'diagnostico']]))
    
    st.markdown("---")
    
    # Distribuições com mais detalhes
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🐾 Distribuição por Espécie")
        especie_counts = df['especie'].value_counts()
        
        # Mostrar contagens
        st.write("**Contagens:**")
        for especie, count in especie_counts.items():
            percentage = (count / len(df)) * 100
            st.write(f"• {especie}: {count} ({percentage:.1f}%)")
        
        # Gráfico
        st.bar_chart(especie_counts)
    
    with col2:
        st.subheader("🏥 Distribuição de Diagnósticos")
        diag_counts = df['diagnostico'].value_counts()
        
        # Mostrar contagens
        st.write("**Top 5 Diagnósticos:**")
        for diag, count in diag_counts.head().items():
            percentage = (count / len(df)) * 100
            st.write(f"• {diag}: {count} ({percentage:.1f}%)")
        
        # Gráfico
        st.bar_chart(diag_counts)
    
    st.markdown("---")
    
    # Estatísticas adicionais
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Estatísticas de Idade")
        idade_stats = df['idade_anos'].describe()
        st.write(f"**Idade Média:** {idade_stats['mean']:.1f} anos")
        st.write(f"**Idade Mínima:** {idade_stats['min']:.1f} anos")
        st.write(f"**Idade Máxima:** {idade_stats['max']:.1f} anos")
        
        # Histograma de idade
        st.bar_chart(df['idade_anos'].value_counts().sort_index())
    
    with col2:
        st.subheader("🌡️ Sinais Vitais Médios")
        temp_media = df['temperatura_retal'].mean()
        pulso_medio = df['pulso'].mean()
        freq_media = df['freq_respiratoria'].mean()
        
        st.write(f"**Temperatura Média:** {temp_media:.1f}°C")
        st.write(f"**Pulso Médio:** {pulso_medio:.0f} bpm")
        st.write(f"**Frequência Respiratória:** {freq_media:.0f} rpm")
        
        # Gráfico de temperatura por espécie
        temp_por_especie = df.groupby('especie')['temperatura_retal'].mean()
        st.bar_chart(temp_por_especie)
    
    st.markdown("---")
    
    # Amostra dos dados com mais informações
    st.subheader("📋 Amostra dos Dados (Primeiros 10 Registros)")
    
    # Selecionar colunas principais para exibir
    colunas_principais = ['id', 'especie', 'raca', 'idade_anos', 'sexo', 'diagnostico', 
                         'temperatura_retal', 'febre', 'vomito', 'diarreia']
    
    if all(col in df.columns for col in colunas_principais):
        st.dataframe(df[colunas_principais].head(10), use_container_width=True)
    else:
        st.dataframe(df.head(10), use_container_width=True)
    
    # Informações sobre o dataset
    st.markdown("---")
    st.subheader("ℹ️ Informações sobre o Dataset")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Colunas Disponíveis:**")
        st.write(f"Total: {len(df.columns)} colunas")
        st.write("• Identificação: id, espécie, raça, idade, sexo")
        st.write("• Exames: hemoglobina, hematócrito, leucócitos, etc.")
        st.write("• Sinais vitais: temperatura, pulso, frequência respiratória")
        st.write("• Sintomas: febre, vômito, diarreia, apatia, etc.")
        st.write("• Diagnóstico: classificação da condição")
    
    with col2:
        st.write("**Qualidade dos Dados:**")
        valores_nulos = df.isnull().sum().sum()
        st.write(f"• Registros sem dados faltantes: {len(df) - valores_nulos}/{len(df)}")
        st.write(f"• Espécies: {', '.join(df['especie'].unique())}")
        st.write(f"• Faixa de idade: {df['idade_anos'].min():.1f} - {df['idade_anos'].max():.1f} anos")
        st.write(f"• Dados sintéticos para demonstração")

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
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("👶 Média de Idade", f"{df['idade_anos'].mean():.1f} anos")
        st.metric("⚖️ Peso Médio", f"{df['peso_kg'].mean():.1f} kg")
        st.metric("🌡️ Temperatura Média", f"{df['temperatura_retal'].mean():.1f}°C")
    
    with col2:
        st.metric("🔥 Taxa de Febre", f"{(df['febre'].sum() / len(df) * 100):.1f}%")
        st.metric("🤮 Taxa de Vômito", f"{(df['vomito'].sum() / len(df) * 100):.1f}%")
        st.metric("💩 Taxa de Diarreia", f"{(df['diarreia'].sum() / len(df) * 100):.1f}%")
    
    with col3:
        st.metric("😴 Taxa de Apatia", f"{(df['apatia'].sum() / len(df) * 100):.1f}%")
        st.metric("📉 Taxa de Perda de Peso", f"{(df['perda_peso'].sum() / len(df) * 100):.1f}%")
        st.metric("🫁 Taxa de Tosse", f"{(df['tosse'].sum() / len(df) * 100):.1f}%")
    
    st.markdown("---")
    
    # Análises por espécie
    st.subheader("🐾 Análises por Espécie")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Estatísticas por Espécie")
        especie_stats = df.groupby('especie').agg({
            'idade_anos': ['mean', 'std'],
            'peso_kg': ['mean', 'std'],
            'temperatura_retal': ['mean', 'std'],
            'febre': 'sum',
            'vomito': 'sum',
            'diarreia': 'sum'
        }).round(2)
        
        # Simplificar nomes das colunas
        especie_stats.columns = ['Idade_Média', 'Idade_Desvio', 'Peso_Médio', 'Peso_Desvio',
                               'Temp_Média', 'Temp_Desvio', 'Casos_Febre', 'Casos_Vômito', 'Casos_Diarreia']
        st.dataframe(especie_stats)
    
    with col2:
        st.subheader("📈 Distribuição de Idades por Espécie")
        # Criar histograma de idades por espécie
        for especie in df['especie'].unique():
            especie_data = df[df['especie'] == especie]['idade_anos']
            st.write(f"**{especie}** - Idades:")
            st.bar_chart(especie_data.value_counts().sort_index())
    
    st.markdown("---")
    
    # Análises por diagnóstico
    st.subheader("🏥 Análises por Diagnóstico")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Estatísticas por Diagnóstico")
        diag_stats = df.groupby('diagnostico').agg({
            'idade_anos': ['mean', 'count'],
            'peso_kg': 'mean',
            'temperatura_retal': 'mean',
            'febre': 'mean',
            'vomito': 'mean'
        }).round(2)
        
        diag_stats.columns = ['Idade_Média', 'N_Casos', 'Peso_Médio', 'Temp_Média', 'Taxa_Febre', 'Taxa_Vômito']
        st.dataframe(diag_stats.sort_values('N_Casos', ascending=False))
    
    with col2:
        st.subheader("📈 Distribuição de Diagnósticos")
        diag_counts = df.groupby('diagnostico').size().sort_values(ascending=False)
        st.bar_chart(diag_counts)
        
        # Mostrar percentuais
        st.write("**Percentuais:**")
        total = len(df)
        for diag, count in diag_counts.items():
            percentage = (count / total) * 100
            st.write(f"• {diag}: {percentage:.1f}%")
    
    st.markdown("---")
    
    # Exames laboratoriais
    st.subheader("🔬 Análise de Exames Laboratoriais")
    
    exames_cols = ['hemoglobina', 'hematocrito', 'leucocitos', 'glicose', 'ureia', 'creatinina']
    exames_disponiveis = [col for col in exames_cols if col in df.columns]
    
    if exames_disponiveis:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Estatísticas dos Exames")
            exames_stats = df[exames_disponiveis].describe().round(2)
            st.dataframe(exames_stats)
        
        with col2:
            st.subheader("📈 Exame Selecionado por Diagnóstico")
            exame_selecionado = st.selectbox("Selecione um exame:", exames_disponiveis)
            
            exame_por_diag = df.groupby('diagnostico')[exame_selecionado].mean().sort_values(ascending=False)
            st.bar_chart(exame_por_diag)
            
            st.write(f"**Média geral de {exame_selecionado}:** {df[exame_selecionado].mean():.2f}")
    
    st.markdown("---")
    
    # Correlações
    st.subheader("🔗 Matriz de Correlações")
    
    colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(colunas_numericas) > 1:
        correlacao = df[colunas_numericas].corr()
        
        # Mostrar apenas correlações relevantes
        st.write("**Correlações mais significativas:**")
        for i in range(len(correlacao.columns)):
            for j in range(i+1, len(correlacao.columns)):
                corr_val = correlacao.iloc[i, j]
                if abs(corr_val) > 0.3:  # Apenas correlações moderadas/fortes
                    st.write(f"• {correlacao.columns[i]} ↔ {correlacao.columns[j]}: {corr_val:.3f}")
    
    st.markdown("---")
    
    # Download dos dados
    st.subheader("📥 Download de Dados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📊 Baixar Dataset Completo (CSV)",
            data=csv,
            file_name='veterinary_data_complete.csv',
            mime='text/csv',
        )
    
    with col2:
        # Estatísticas resumidas
        resumo_stats = df.describe().round(2)
        csv_resumo = resumo_stats.to_csv().encode('utf-8')
        st.download_button(
            label="📈 Baixar Estatísticas Resumidas (CSV)",
            data=csv_resumo,
            file_name='veterinary_statistics.csv',
            mime='text/csv',
        )

elif pagina == "📁 Informações do Dataset":
    st.header("📁 Informações Detalhadas do Dataset")
    
    # Informações gerais
    st.subheader("📊 Resumo Geral")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📄 Total de Registros", len(df))
    
    with col2:
        st.metric("📋 Total de Colunas", len(df.columns))
    
    with col3:
        valores_nulos = df.isnull().sum().sum()
        st.metric("❌ Valores Nulos", valores_nulos)
    
    with col4:
        memoria_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        st.metric("💾 Memória (MB)", f"{memoria_mb:.2f}")
    
    st.markdown("---")
    
    # Informações sobre colunas
    st.subheader("📋 Estrutura das Colunas")
    
    # Análise por tipo de coluna
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔢 Colunas Numéricas")
        colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
        st.write(f"**Total:** {len(colunas_numericas)} colunas")
        for col in colunas_numericas:
            st.write(f"• {col}")
    
    with col2:
        st.subheader("📝 Colunas Categóricas")
        colunas_categoricas = df.select_dtypes(include=['object']).columns.tolist()
        st.write(f"**Total:** {len(colunas_categoricas)} colunas")
        for col in colunas_categoricas:
            st.write(f"• {col}")
    
    st.markdown("---")
    
    # Estatísticas descritivas
    st.subheader("📈 Estatísticas Descritivas")
    
    if len(colunas_numericas) > 0:
        st.dataframe(df[colunas_numericas].describe().round(2), use_container_width=True)
    else:
        st.info("Nenhuma coluna numérica encontrada")
    
    st.markdown("---")
    
    # Valores únicos por coluna
    st.subheader("🔍 Valores Únicos por Coluna")
    
    valores_unicos = []
    for col in df.columns:
        n_unicos = df[col].nunique()
        valores_unicos.append({
            'Coluna': col,
            'Valores Únicos': n_unicos,
            'Tipo': str(df[col].dtype),
            'Valores Nulos': df[col].isnull().sum()
        })
    
    df_unicos = pd.DataFrame(valores_unicos)
    st.dataframe(df_unicos, use_container_width=True)
    
    st.markdown("---")
    
    # Amostra dos dados
    st.subheader("👀 Amostra dos Dados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Primeiros 5 registros:**")
        st.dataframe(df.head(), use_container_width=True)
    
    with col2:
        st.write("**Últimos 5 registros:**")
        st.dataframe(df.tail(), use_container_width=True)
    
    st.markdown("---")
    
    # Informações sobre o arquivo
    st.subheader("📁 Informações do Arquivo")
    
    # Tentar identificar o arquivo carregado
    try:
        data_path = Path("data")
        if data_path.exists():
            csv_files = list(data_path.glob("*.csv"))
            
            st.write("**Arquivos CSV disponíveis na pasta data:**")
            for i, arquivo in enumerate(csv_files, 1):
                tamanho = arquivo.stat().st_size / 1024  # KB
                st.write(f"{i}. {arquivo.name} ({tamanho:.1f} KB)")
            
            # Mostrar qual foi carregado
            st.write(f"\n**Arquivo atual carregado:** {len(df)} registros")
            
    except Exception as e:
        st.info("Informações do arquivo não disponíveis")
    
    # Download
    st.markdown("---")
    st.subheader("📥 Download")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📊 Dataset Completo",
            data=csv,
            file_name='dataset_completo.csv',
            mime='text/csv',
        )
    
    with col2:
        if len(colunas_numericas) > 0:
            csv_numerico = df[colunas_numericas].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="🔢 Apenas Numéricas",
                data=csv_numerico,
                file_name='dados_numericos.csv',
                mime='text/csv',
            )
    
    with col3:
        csv_estatisticas = df.describe().to_csv().encode('utf-8')
        st.download_button(
            label="📈 Estatísticas",
            data=csv_estatisticas,
            file_name='estatisticas.csv',
            mime='text/csv',
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🐾 VetDiagnosisAI - Sistema de Apoio ao Diagnóstico Veterinário</p>
    <p>Desenvolvido para o MBA - Sistema Completo com Datasets Reais</p>
</div>
""", unsafe_allow_html=True)
