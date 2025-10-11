"""
VetDiagnosisAI - Sistema Inteligente de Apoio ao Diagnóstico Veterinário
Aplicação principal Streamlit
"""

import streamlit as st
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

# Inicialização do session_state
if 'df_main' not in st.session_state:
    st.session_state.df_main = None
if 'modelo_treinado' not in st.session_state:
    st.session_state.modelo_treinado = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'target_names' not in st.session_state:
    st.session_state.target_names = None

# Função para carregar dataset automaticamente
@st.cache_data(ttl=3600)  # Cache por 1 hora
def carregar_dataset_fixo():
    """Carrega o dataset de forma fixa e em cache"""
    try:
        import pandas as pd
        import numpy as np
        from pathlib import Path
        
        # Tentar carregar dataset da pasta data - priorizar datasets reais
        data_path = Path("data")
        csv_files = list(data_path.glob("*.csv")) if data_path.exists() else []
        
        if csv_files:
            # Priorizar datasets reais específicos
            datasets_prioritarios = [
                'veterinary_complete_real_dataset.csv',
                'veterinary_master_dataset.csv', 
                'veterinary_realistic_dataset.csv',
                'clinical_veterinary_data.csv',
                'laboratory_complete_panel.csv',
                'uci_horse_colic.csv'
            ]
            
            dataset_escolhido = None
            for dataset in datasets_prioritarios:
                if Path(data_path / dataset).exists():
                    dataset_escolhido = data_path / dataset
                    break
            
            # Se não encontrar um dos prioritários, usar o primeiro disponível
            if not dataset_escolhido:
                dataset_escolhido = csv_files[0]
            
            df = pd.read_csv(dataset_escolhido)
            df = df.dropna(how='all')  # Remove linhas completamente vazias
            
            # Padronizar nomes de colunas se necessário
            if 'especie' in df.columns:
                df['especie'] = df['especie'].str.title()
                df['especie'] = df['especie'].replace({'Canina': 'Cão', 'Felina': 'Gato'})
            
            return df
        
        # Se não encontrar arquivos, criar dados de exemplo
        np.random.seed(42)
        n_samples = 100
        
        # Criar dados sintéticos
        data = {
            'id': range(1, n_samples + 1),
            'especie': np.random.choice(['Cão', 'Gato'], n_samples),
            'raca': np.random.choice(['SRD', 'Pastor', 'Siames', 'Persa'], n_samples),
            'idade_anos': np.random.uniform(1, 15, n_samples).round(1),
            'sexo': np.random.choice(['M', 'F'], n_samples),
            'hemoglobina': np.random.normal(12, 2, n_samples).round(1),
            'hematocrito': np.random.normal(40, 5, n_samples).round(1),
            'leucocitos': np.random.normal(8000, 2000, n_samples).round(0),
            'glicose': np.random.normal(100, 20, n_samples).round(1),
            'ureia': np.random.normal(30, 10, n_samples).round(1),
            'creatinina': np.random.normal(1.2, 0.3, n_samples).round(2),
            'temperatura_retal': np.random.normal(38.5, 0.5, n_samples).round(1),
            'febre': np.random.choice([0, 1], n_samples),
            'apatia': np.random.choice([0, 1], n_samples),
            'perda_peso': np.random.choice([0, 1], n_samples),
            'vomito': np.random.choice([0, 1], n_samples),
            'diarreia': np.random.choice([0, 1], n_samples),
            'diagnostico': np.random.choice(['Normal', 'Infecção', 'Doença Renal', 'Diabetes'], n_samples)
        }
        
        df = pd.DataFrame(data)
        return df
        
    except Exception as e:
        st.error(f"❌ Erro ao carregar dataset: {str(e)}")
        return None

# Carregar dataset automaticamente SEMPRE
if st.session_state.df_main is None:
    # Carregar dados imediatamente sem spinner para melhor UX
    df_auto = carregar_dataset_fixo()
    if df_auto is not None:
        st.session_state.df_main = df_auto
        st.session_state.dataset_carregado_auto = True
        st.session_state.dataset_sempre_carregado = True
        st.session_state.dados_prontos = True
    else:
        st.session_state.dados_prontos = False
        st.error("❌ Não foi possível carregar dataset automaticamente")
else:
    st.session_state.dados_prontos = True

# Garantir que sempre temos dados
if st.session_state.df_main is None:
    df_auto = carregar_dataset_fixo()
    if df_auto is not None:
        st.session_state.df_main = df_auto
        st.session_state.dados_prontos = True

# Sidebar com informações
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/veterinarian.png", width=100)
    st.title("Navegação")
    st.markdown("---")
    
    # Status do dataset (sempre carregado)
    st.subheader("📊 Status dos Dados")
    if st.session_state.df_main is not None:
        st.success(f"✅ Dataset sempre carregado: {len(st.session_state.df_main)} registros")
        
        # Mostrar informações do dataset
        if hasattr(st.session_state.df_main, 'columns'):
            st.caption(f"📋 Colunas: {len(st.session_state.df_main.columns)}")
            if 'diagnostico' in st.session_state.df_main.columns:
                diagnosticos = st.session_state.df_main['diagnostico'].nunique()
                st.caption(f"🏥 Diagnósticos: {diagnosticos}")
            if 'especie' in st.session_state.df_main.columns:
                especies = st.session_state.df_main['especie'].nunique()
                st.caption(f"🐾 Espécies: {especies}")
        
        # Mostrar que está sempre disponível
        st.info("🔄 **Dataset carregado automaticamente** - Sempre disponível!")
        
        # Botão para forçar recarregamento
        if st.button("🔄 Recarregar Dataset", use_container_width=True):
            # Limpar cache
            carregar_dataset_fixo.clear()
            # Recarregar
            df_auto = carregar_dataset_fixo()
            if df_auto is not None:
                st.session_state.df_main = df_auto
                st.success(f"✅ Dataset recarregado: {len(df_auto)} registros")
                st.rerun()
            else:
                st.error("❌ Erro ao recarregar dataset")
    else:
        # Este caso não deveria acontecer mais, mas mantemos como fallback
        st.error("❌ Erro: Dataset não carregado")
        st.markdown("🔄 **Tentando carregar automaticamente...**")
        
        if st.button("📊 Forçar Carregamento", type="primary", use_container_width=True):
            carregar_dataset_fixo.clear()
            df_auto = carregar_dataset_fixo()
            if df_auto is not None:
                st.session_state.df_main = df_auto
                st.session_state.dataset_sempre_carregado = True
                st.success(f"✅ Dataset carregado: {len(df_auto)} registros")
                st.rerun()
            else:
                st.error("❌ Erro ao carregar dataset")
    
    # Status do modelo
    st.subheader("🤖 Status do Modelo")
    if st.session_state.modelo_treinado is not None:
        st.success("✅ Modelo treinado disponível")
    else:
        st.warning("⚠️ Nenhum modelo treinado")
        st.markdown("👉 Vá para **🤖 Treinar Modelo**")
    
    st.markdown("---")
    
    # Datasets sugeridos
    with st.expander("🔗 Datasets Públicos Sugeridos"):
        st.markdown("""
        **1. Kaggle – Veterinary Disease Detection**
        
        Dados de sintomas e diagnósticos veterinários.
        
        [🔗 Acessar](https://www.kaggle.com/datasets/taruntiwarihp/veterinary-disease-detection)
        
        ---
        
        **2. UCI – Horse Colic**
        
        Dados de cólica em cavalos (excelente para ML).
        
        [🔗 Acessar](https://archive.ics.uci.edu/dataset/46/horse+colic)
        
        ---
        
        **3. Kaggle – Animal Blood Samples**
        
        Amostras de sangue de animais para análise.
        
        [🔗 Acessar](https://www.kaggle.com/datasets/andrewmvd/animal-blood-samples)
        
        ---
        
        ⚠️ **Importante:** Verifique as licenças e termos de uso.
        """)
    
    st.markdown("---")
    
    # Avisos legais
    with st.expander("⚠️ Avisos Legais"):
        st.warning("""
        **Esta é uma ferramenta educacional.**
        
        - ❌ NÃO substitui julgamento clínico veterinário
        - ❌ NÃO deve ser usada como única base para decisões
        - ✅ Ideal para ensino e pesquisa
        - ✅ Apoio à decisão para profissionais
        
        **Sempre consulte um médico veterinário licenciado.**
        """)

# Corpo principal - Status do dataset
st.markdown("## 🎯 Bem-vindo ao VetDiagnosisAI")

# Status do dataset sempre carregado
if st.session_state.df_main is not None:
    st.success(f"✅ **Dataset sempre carregado e pronto!** - {len(st.session_state.df_main)} registros disponíveis")
    
    # Mostrar estatísticas rápidas
    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
    
    with col_stats1:
        st.metric("📄 Total de Registros", len(st.session_state.df_main))
    
    with col_stats2:
        if 'diagnostico' in st.session_state.df_main.columns:
            diagnosticos = st.session_state.df_main['diagnostico'].nunique()
            st.metric("🏥 Diagnósticos", diagnosticos)
        else:
            st.metric("🏥 Diagnósticos", "N/A")
    
    with col_stats3:
        if 'especie' in st.session_state.df_main.columns:
            especies = st.session_state.df_main['especie'].nunique()
            st.metric("🐾 Espécies", especies)
        else:
            st.metric("🐾 Espécies", "N/A")
    
    with col_stats4:
        st.metric("📋 Colunas", len(st.session_state.df_main.columns))
    
    st.info("🔄 **O dataset é carregado automaticamente sempre que você acessar a aplicação!**")
else:
    st.error("❌ Erro: Dataset não está carregado. Recarregue a página.")

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 📊 1. Explorar Dados
    
    O dataset já está carregado! Explore os dados:
    
    - **📊 Visão Geral**: Métricas principais
    - **🧪 EDA**: Análise exploratória interativa
    - **📥 Upload**: Adicionar mais dados se necessário
    """)

with col2:
    st.markdown("""
    ### 🔍 2. Explorar & Analisar
    
    Navegue pelas páginas de análise:
    
    - **📊 Visão Geral**: Métricas principais
    - **🧪 EDA**: Análise exploratória interativa
    - **🧠 Insights**: Observações automáticas
    """)

with col3:
    st.markdown("""
    ### 🤖 3. Treinar & Prever
    
    Use Machine Learning:
    
    - **🤖 Treinar Modelo**: Pipeline completo de ML
    - **🔍 Predição**: Diagnósticos com explicabilidade
    """)

st.markdown("---")

# Cards com recursos principais
st.markdown("## 🌟 Principais Recursos")

col1, col2 = st.columns(2)

with col1:
    with st.container():
        st.markdown("""
        ### 📊 Análise de Dados Veterinários
        
        - Suporte a múltiplas espécies (Canina, Felina, Equina)
        - Análise de exames laboratoriais completos
        - Correlação entre sintomas e diagnósticos
        - Detecção automática de valores críticos
        - Comparação com faixas de referência por espécie
        """)

with col2:
    with st.container():
        st.markdown("""
        ### 🤖 Machine Learning Avançado
        
        - Múltiplos algoritmos (LogReg, RF, LightGBM, XGBoost)
        - Balanceamento automático de classes
        - Grid search de hiperparâmetros
        - Explicabilidade com SHAP
        - Validação cruzada estratificada
        """)

st.markdown("---")

# Quick start
st.markdown("## 🚀 Quick Start")

with st.expander("📖 Ver tutorial rápido"):
    st.markdown("""
    ### Passo a Passo
    
    1. **Carregue os dados** (página "📥 Upload de Dados")
       - Use o arquivo `data/exemplo_vet.csv` para começar
       - Ou faça upload do seu próprio dataset
       
    2. **Explore os dados** (página "🧪 Laboratório & Sintomas")
       - Visualize distribuições
       - Identifique correlações
       - Analise por espécie/raça
       
    3. **Treine um modelo** (página "🤖 Treinar Modelo")
       - Selecione algoritmo
       - Configure parâmetros
       - Avalie performance
       
    4. **Faça predições** (página "🔍 Predição")
       - Insira dados manualmente
       - Ou faça upload de arquivo
       - Veja diagnósticos prováveis com explicação
       
    5. **Analise insights** (página "🧠 Insights & Regras")
       - Observações clínicas automáticas
       - Hipóteses baseadas em dados
       - Sugestões de acompanhamento
    """)

st.markdown("---")

# Footer
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🐾 VetDiagnosisAI v1.0 | Desenvolvido para profissionais veterinários e pesquisadores</p>
    <p>⚠️ Ferramenta educacional - Não substitui avaliação clínica profissional</p>
</div>
""", unsafe_allow_html=True)

