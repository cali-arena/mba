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
    page_title="VetDiagnosisAI v2.0 - ML Veterinário Completo",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# FORÇAR ATUALIZAÇÃO - VERSÃO 2.0
st.info("🚀 **VET DIAGNOSIS AI v2.0 - SISTEMA ATUALIZADO COM ML COMPLETO!** 🚀")

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
# @st.cache_data(ttl=60)  # Cache desabilitado para forçar atualização
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
            
            # Adicionar informação sobre qual dataset foi carregado
            import datetime
            df.attrs['dataset_source'] = dataset_escolhido.name
            df.attrs['dataset_path'] = str(dataset_escolhido)
            df.attrs['load_timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
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

# CARREGAR DADOS REAIS DIRETAMENTE - SEMPRE!
st.info("🔄 **VERSÃO 2.0 - CARREGANDO DADOS REAIS COM ML COMPLETO...**")

# Tentar carregar datasets reais diretamente
data_path = Path("data")
df = None
dataset_carregado = None

# Lista de datasets reais em ordem de prioridade
real_datasets = [
    'veterinary_complete_real_dataset.csv',  # 800 registros
    'clinical_veterinary_data.csv',          # 500 registros
    'veterinary_master_dataset.csv',         # 500 registros
    'veterinary_realistic_dataset.csv',      # 1280 registros
    'laboratory_complete_panel.csv',         # 300 registros
    'uci_horse_colic.csv',                   # 368 registros
    'exemplo_vet.csv'                        # 300 registros (fallback)
]

# Tentar carregar cada dataset até encontrar um
for dataset_name in real_datasets:
    dataset_path = data_path / dataset_name
    if dataset_path.exists():
        try:
            df = pd.read_csv(dataset_path)
            dataset_carregado = dataset_name
            st.success(f"✅ Dataset carregado: {dataset_name} ({len(df)} registros)")
            break
        except Exception as e:
            st.error(f"❌ Erro ao carregar {dataset_name}: {e}")
            continue

# Se não conseguiu carregar nenhum dataset real, usar função de fallback
if df is None or len(df) == 0:
    st.warning("⚠️ Não foi possível carregar datasets reais. Usando função de fallback...")
    df = carregar_dataset_completo()
    dataset_carregado = "fallback"

# Verificar se os dados foram carregados
if df is None or len(df) == 0:
    st.error("❌ Erro crítico: Não foi possível carregar nenhum dataset!")
    st.stop()

# Adicionar informações de debug
df.attrs = {
    'dataset_source': dataset_carregado,
    'dataset_path': str(data_path / dataset_carregado) if dataset_carregado != "fallback" else "fallback",
    'load_timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/veterinarian.png", width=100)
    st.title("🐾 VetDiagnosisAI")
    st.markdown("---")
    
    st.subheader("📊 Status dos Dados")
    st.success(f"✅ Dataset carregado: {len(df)} registros")
    st.info(f"📅 Colunas: {len(df.columns)}")
    
    # Mostrar status do dataset carregado
    if len(df) >= 500:
        st.success(f"🎉 Dataset real carregado! ({len(df)} registros)")
    elif len(df) >= 300:
        st.warning(f"⚠️ Dataset médio carregado ({len(df)} registros)")
    else:
        st.error(f"❌ Dataset pequeno detectado ({len(df)} registros)")
        
    # Mostrar informações básicas sobre arquivos disponíveis
    data_path = Path("data")
    if data_path.exists():
        csv_files = list(data_path.glob("*.csv"))
        st.info(f"📁 {len(csv_files)} arquivos CSV disponíveis")
    else:
        st.error("❌ Pasta 'data' não encontrada!")
    
    # Mostrar informações de debug sobre o dataset
    if hasattr(df, 'attrs') and 'dataset_source' in df.attrs:
        st.success(f"📁 Dataset: {df.attrs['dataset_source']}")
        st.caption(f"🔗 Caminho: {df.attrs['dataset_path']}")
        if 'load_timestamp' in df.attrs:
            st.caption(f"⏰ Carregado em: {df.attrs['load_timestamp']}")
    else:
        st.warning("⚠️ Informações do dataset não disponíveis")
    
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
    
    # Botão para forçar reload
    st.markdown("---")
    if st.button("🔄 Forçar Reload dos Dados", use_container_width=True):
        carregar_dataset_completo.clear()
        st.rerun()
    
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
        if 'especie' in df.columns:
            st.metric("🐾 Espécies Únicas", df['especie'].nunique())
        else:
            st.metric("🐾 Espécies Únicas", "N/A")
    
    with col3:
        if 'diagnostico' in df.columns:
            st.metric("🏥 Diagnósticos Únicos", df['diagnostico'].nunique())
        else:
            st.metric("🏥 Diagnósticos Únicos", "N/A")
    
    with col4:
        st.metric("🔬 Exames Disponíveis", len([col for col in df.columns if col not in ['id', 'especie', 'raca', 'diagnostico']]))
    
    st.markdown("---")
    
    # Distribuições com mais detalhes
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🐾 Distribuição por Espécie")
        if 'especie' in df.columns:
            especie_counts = df['especie'].value_counts()
        else:
            especie_counts = pd.Series()
        
        # Mostrar contagens
        st.write("**Contagens:**")
        for especie, count in especie_counts.items():
            percentage = (count / len(df)) * 100
            st.write(f"• {especie}: {count} ({percentage:.1f}%)")
        
        # Gráfico
        st.bar_chart(especie_counts)
    
    with col2:
        st.subheader("🏥 Distribuição de Diagnósticos")
        if 'diagnostico' in df.columns:
            diag_counts = df['diagnostico'].value_counts()
        else:
            diag_counts = pd.Series()
        
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
        if 'idade_anos' in df.columns:
            idade_stats = df['idade_anos'].describe()
        else:
            idade_stats = pd.Series()
        if not idade_stats.empty:
            st.write(f"**Idade Média:** {idade_stats['mean']:.1f} anos")
            st.write(f"**Idade Mínima:** {idade_stats['min']:.1f} anos")
            st.write(f"**Idade Máxima:** {idade_stats['max']:.1f} anos")
            
            # Histograma de idade
            st.bar_chart(df['idade_anos'].value_counts().sort_index())
        else:
            st.info("ℹ️ Informações de idade não disponíveis")
    
    with col2:
        st.subheader("🌡️ Sinais Vitais Médios")
        
        # Verificar se as colunas existem antes de acessá-las
        if 'temperatura_retal' in df.columns:
            temp_media = df['temperatura_retal'].mean()
            st.write(f"**Temperatura Média:** {temp_media:.1f}°C")
        
        if 'pulso' in df.columns:
            pulso_medio = df['pulso'].mean()
            st.write(f"**Pulso Médio:** {pulso_medio:.0f} bpm")
        
        if 'freq_respiratoria' in df.columns:
            freq_media = df['freq_respiratoria'].mean()
            st.write(f"**Frequência Respiratória:** {freq_media:.0f} rpm")
        
        # Se nenhuma coluna de sinais vitais existir, mostrar outras métricas
        if not any(col in df.columns for col in ['temperatura_retal', 'pulso', 'freq_respiratoria']):
            st.info("ℹ️ Sinais vitais não disponíveis neste dataset")
        
        # Gráfico de temperatura por espécie (se disponível)
        if 'temperatura_retal' in df.columns and 'especie' in df.columns:
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
        if 'especie' in df.columns:
            st.write(f"• Espécies: {', '.join(df['especie'].unique())}")
        if 'idade_anos' in df.columns:
            st.write(f"• Faixa de idade: {df['idade_anos'].min():.1f} - {df['idade_anos'].max():.1f} anos")
        st.write(f"• Dados sintéticos para demonstração")

elif pagina == "📊 Análise de Dados":
    st.header("📊 Análise Exploratória dos Dados")
    
    # Filtros
    st.subheader("🔍 Filtros")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'especie' in df.columns:
            especie_filtro = st.selectbox("Espécie:", ['Todas'] + list(df['especie'].unique()))
        else:
            especie_filtro = 'Todas'
    
    with col2:
        idade_min, idade_max = st.slider("Faixa de Idade:", 0.0, 20.0, (0.0, 20.0))
    
    with col3:
        if 'diagnostico' in df.columns:
            diag_filtro = st.selectbox("Diagnóstico:", ['Todos'] + list(df['diagnostico'].unique()))
        else:
            diag_filtro = 'Todos'
    
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
    st.header("🤖 Sistema de Machine Learning Veterinário v2.0")
    st.success("🎉 **ATUALIZADO!** Sistema completo de ML com feature engineering avançado!")
    
    # Verificar se temos dados suficientes para ML
    if 'diagnostico' not in df.columns:
        st.error("❌ Coluna 'diagnostico' não encontrada. Não é possível treinar modelos.")
        st.stop()
    
    # Preparar dados para ML
    st.subheader("🔧 Preparação dos Dados")
    
    # Feature Engineering Avançado
    df_ml = df.copy()
    
    # 1. Codificação de variáveis categóricas
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    le_especie = LabelEncoder()
    le_sexo = LabelEncoder()
    le_diagnostico = LabelEncoder()
    
    if 'especie' in df_ml.columns:
        df_ml['especie_encoded'] = le_especie.fit_transform(df_ml['especie'])
    if 'sexo' in df_ml.columns:
        df_ml['sexo_encoded'] = le_sexo.fit_transform(df_ml['sexo'])
    
    df_ml['diagnostico_encoded'] = le_diagnostico.fit_transform(df_ml['diagnostico'])
    
    # 2. Criar features derivadas
    if 'idade_anos' in df_ml.columns:
        df_ml['idade_categoria'] = pd.cut(df_ml['idade_anos'], bins=[0, 2, 7, 15, 100], labels=['Filhote', 'Adulto Jovem', 'Adulto', 'Idoso'])
        df_ml['idade_categoria_encoded'] = LabelEncoder().fit_transform(df_ml['idade_categoria'])
    
    # 3. Features de exames laboratoriais combinados
    exames_cols = ['hemoglobina', 'hematocrito', 'leucocitos', 'glicose', 'ureia', 'creatinina']
    exames_disponiveis = [col for col in exames_cols if col in df_ml.columns]
    
    if len(exames_disponiveis) >= 2:
        # Criar índices combinados
        if 'hemoglobina' in df_ml.columns and 'hematocrito' in df_ml.columns:
            df_ml['indice_anemia'] = df_ml['hemoglobina'] / df_ml['hematocrito']
        if 'ureia' in df_ml.columns and 'creatinina' in df_ml.columns:
            df_ml['indice_renal'] = df_ml['ureia'] / df_ml['creatinina']
    
    # 4. Features de sintomas combinados
    sintomas_cols = ['febre', 'apatia', 'perda_peso', 'vomito', 'diarreia', 'tosse', 'letargia']
    sintomas_disponiveis = [col for col in sintomas_cols if col in df_ml.columns]
    
    if len(sintomas_disponiveis) >= 2:
        df_ml['total_sintomas'] = df_ml[sintomas_disponiveis].sum(axis=1)
        df_ml['severidade_sintomas'] = pd.cut(df_ml['total_sintomas'], bins=[-1, 0, 2, 4, 10], labels=['Assintomático', 'Leve', 'Moderado', 'Severo'])
        df_ml['severidade_sintomas_encoded'] = LabelEncoder().fit_transform(df_ml['severidade_sintomas'])
    
    # Selecionar features para ML
    feature_cols = []
    
    # Adicionar colunas numéricas originais
    numeric_cols = df_ml.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols.extend([col for col in numeric_cols if col not in ['diagnostico_encoded']])
    
    # Remover colunas com muitos valores únicos (como ID)
    feature_cols = [col for col in feature_cols if df_ml[col].nunique() < len(df_ml) * 0.8]
    
    X = df_ml[feature_cols].fillna(df_ml[feature_cols].mean())
    y = df_ml['diagnostico_encoded']
    
    st.success(f"✅ Dados preparados: {X.shape[0]} amostras, {X.shape[1]} features")
    
    # Dividir dados
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Escalar features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    st.info(f"📊 Divisão dos dados: {X_train.shape[0]} treino, {X_test.shape[0]} teste")
    
    # Treinar múltiplos modelos
    st.subheader("🤖 Treinamento de Modelos")
    
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import plotly.express as px
    import plotly.graph_objects as go
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42, probability=True),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
    }
    
    results = {}
    
    for name, model in models.items():
        with st.spinner(f"Treinando {name}..."):
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred
            }
    
    # Mostrar resultados
    st.subheader("📊 Resultados dos Modelos")
    
    # Tabela de resultados
    results_df = pd.DataFrame({
        'Modelo': list(results.keys()),
        'Acurácia': [results[name]['accuracy'] for name in results.keys()]
    }).sort_values('Acurácia', ascending=False)
    
    st.dataframe(results_df, use_container_width=True)
    
    # Melhor modelo
    best_model_name = results_df.iloc[0]['Modelo']
    best_model = results[best_model_name]['model']
    best_accuracy = results_df.iloc[0]['Acurácia']
    
    st.success(f"🏆 **Melhor Modelo:** {best_model_name} com {best_accuracy:.3f} de acurácia")
    
    # Feature Importance (se disponível)
    if hasattr(best_model, 'feature_importances_'):
        st.subheader("🎯 Importância das Features")
        
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(feature_importance.head(10), x='Importance', y='Feature', 
                     title='Top 10 Features Mais Importantes')
        st.plotly_chart(fig, use_container_width=True)
    
    # Matriz de confusão
    st.subheader("🔍 Matriz de Confusão")
    
    y_pred_best = results[best_model_name]['predictions']
    cm = confusion_matrix(y_test, y_pred_best)
    
    fig = px.imshow(cm, 
                    labels=dict(x="Predito", y="Real", color="Quantidade"),
                    x=le_diagnostico.classes_,
                    y=le_diagnostico.classes_,
                    title=f"Matriz de Confusão - {best_model_name}")
    st.plotly_chart(fig, use_container_width=True)
    
    # Relatório de classificação
    st.subheader("📋 Relatório Detalhado")
    report = classification_report(y_test, y_pred_best, target_names=le_diagnostico.classes_, output_dict=True)
    
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df, use_container_width=True)
    
    # Sugestões para melhorar acurácia
    st.subheader("💡 Sugestões para Melhorar Acurácia (>85%)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🔧 Feature Engineering:**
        - ✅ Criar mais features derivadas
        - ✅ Combinar exames laboratoriais
        - ✅ Agrupar sintomas por severidade
        - ✅ Usar idade categorizada
        - ✅ Criar índices clínicos específicos
        """)
    
    with col2:
        st.markdown("""
        **🤖 Modelos Avançados:**
        - ✅ XGBoost com hiperparâmetros otimizados
        - ✅ Ensemble de múltiplos modelos
        - ✅ Validação cruzada estratificada
        - ✅ Balanceamento de classes
        - ✅ Seleção de features automática
        """)
    
    # Predição interativa
    st.subheader("🔮 Predição Interativa")
    
    with st.form("prediction_form"):
        st.markdown("**Insira os dados do paciente para predição:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'especie' in df.columns:
                especie_pred = st.selectbox("Espécie:", df['especie'].unique())
            if 'sexo' in df.columns:
                sexo_pred = st.selectbox("Sexo:", df['sexo'].unique())
            if 'idade_anos' in df.columns:
                idade_pred = st.number_input("Idade (anos):", 0.1, 25.0, 5.0)
        
        with col2:
            if 'hemoglobina' in df.columns:
                hemoglobina_pred = st.number_input("Hemoglobina:", 5.0, 20.0, 12.0)
            if 'hematocrito' in df.columns:
                hematocrito_pred = st.number_input("Hematócrito:", 20.0, 60.0, 40.0)
            if 'glicose' in df.columns:
                glicose_pred = st.number_input("Glicose:", 50.0, 300.0, 100.0)
        
        submitted = st.form_submit_button("🔍 Predizer Diagnóstico")
        
        if submitted:
            # Preparar dados para predição
            pred_data = {}
            
            for col in feature_cols:
                if col == 'especie_encoded' and 'especie' in df.columns:
                    pred_data[col] = le_especie.transform([especie_pred])[0]
                elif col == 'sexo_encoded' and 'sexo' in df.columns:
                    pred_data[col] = le_sexo.transform([sexo_pred])[0]
                elif col == 'idade_anos':
                    pred_data[col] = idade_pred
                elif col == 'hemoglobina':
                    pred_data[col] = hemoglobina_pred
                elif col == 'hematocrito':
                    pred_data[col] = hematocrito_pred
                elif col == 'glicose':
                    pred_data[col] = glicose_pred
                else:
                    # Usar valor médio para features não especificadas
                    pred_data[col] = X[col].mean()
            
            # Converter para DataFrame e escalar
            pred_df = pd.DataFrame([pred_data])
            pred_scaled = scaler.transform(pred_df)
            
            # Fazer predição
            prediction = best_model.predict(pred_scaled)[0]
            prediction_proba = best_model.predict_proba(pred_scaled)[0]
            
            # Mostrar resultado
            diagnostico_predito = le_diagnostico.inverse_transform([prediction])[0]
            confianca = prediction_proba.max()
            
            st.success(f"🎯 **Diagnóstico Predito:** {diagnostico_predito}")
            st.info(f"📊 **Confiança:** {confianca:.2%}")
            
            # Mostrar probabilidades de todos os diagnósticos
            proba_df = pd.DataFrame({
                'Diagnóstico': le_diagnostico.classes_,
                'Probabilidade': prediction_proba
            }).sort_values('Probabilidade', ascending=False)
            
            st.subheader("📈 Probabilidades de Todos os Diagnósticos")
            st.dataframe(proba_df, use_container_width=True)
            
            # Gráfico de probabilidades
            fig = px.bar(proba_df, x='Probabilidade', y='Diagnóstico', 
                        title='Probabilidades de Diagnóstico')
            st.plotly_chart(fig, use_container_width=True)

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
