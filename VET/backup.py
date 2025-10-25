"""
VetDiagnosisAI - Sistema Inteligente de Apoio ao Diagnóstico Veterinário
Aplicação principal Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.decomposition import PCA
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    # Fallback para matplotlib se plotly não estiver disponível
    import matplotlib.pyplot as plt
    import seaborn as sns

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
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

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

# Função removida - APENAS dados reais serão usados

# Função removida - carregamento direto implementado abaixo

# CARREGAMENTO DE DADOS

# Tentar carregar dados reais primeiro, depois fallback para incorporados
df_real = None
dataset_source = ""

# 1. Tentar carregar dados reais da pasta data
try:
    # Tentar múltiplos caminhos possíveis
    possible_paths = [
        Path("data"),           # Para execução local
        Path("VET/data"),       # Para execução no Streamlit Cloud
        Path(".") / "data",     # Caminho relativo
        Path(".") / "VET" / "data"  # Caminho relativo com VET
    ]
    
    data_path = None
    for path in possible_paths:
        if path.exists():
            csv_files = list(path.glob("*.csv"))
            if csv_files:
                data_path = path
                # Pasta de dados encontrada
                break
    
    if data_path and data_path.exists():
        csv_files = list(data_path.glob("*.csv"))
        if csv_files:
            # Priorizar datasets reais específicos
            datasets_prioritarios = [
                'veterinary_complete_real_dataset.csv',
                'veterinary_master_dataset.csv', 
                'veterinary_realistic_dataset.csv',
                'clinical_veterinary_data.csv',
                'laboratory_complete_panel.csv'
            ]
            
            for dataset_name in datasets_prioritarios:
                dataset_path = data_path / dataset_name
                if dataset_path.exists():
                    df_real = pd.read_csv(dataset_path)
                    if df_real is not None and len(df_real) > 0:
                        dataset_source = f"dados_reais_{dataset_name}"
                        # Dataset carregado
                        break
except Exception as e:
    # Erro ao carregar dados reais

# 2. APENAS dados reais - SEM fallback para sintéticos
if df_real is not None and len(df_real) > 0:
    # SEMPRE definir os dados no session state
    st.session_state.df_main = df_real
    st.session_state.dataset_carregado_auto = True
    st.session_state.dataset_sempre_carregado = True
    st.session_state.dados_prontos = True
    st.session_state.dataset_source = dataset_source
    
    # Adicionar informações de debug
    import datetime
    st.session_state.dataset_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Sistema inicializado
else:
    st.session_state.dados_prontos = False
    st.error("❌ ERRO: Nenhum dataset real encontrado!")
    st.error("📁 Verifique se existem arquivos CSV nas seguintes pastas:")
    
    # Verificar todos os caminhos possíveis
    possible_paths = [
        Path("data"),           
        Path("VET/data"),       
        Path(".") / "data",     
        Path(".") / "VET" / "data"  
    ]
    
    found_files = False
    for data_path in possible_paths:
        if data_path.exists():
            csv_files = list(data_path.glob("*.csv"))
            if csv_files:
                st.info(f"📋 Arquivos encontrados na pasta {data_path}:")
                for file in csv_files:
                    st.write(f"  - {file.name}")
                found_files = True
            else:
                # Pasta sem arquivos CSV
        else:
            # Pasta não encontrada
    
    if not found_files:
        st.info("💡 Para usar o sistema, adicione datasets reais nas seguintes pastas com os seguintes nomes:")
        st.write("📁 Caminhos possíveis:")
        st.write("- data/")
        st.write("- VET/data/")
        st.write("📋 Arquivos necessários:")
        st.write("- veterinary_complete_real_dataset.csv")
        st.write("- veterinary_master_dataset.csv")
        st.write("- veterinary_realistic_dataset.csv")
        st.write("- clinical_veterinary_data.csv")
        st.write("- laboratory_complete_panel.csv")
    
    st.stop()

# Sidebar com informações
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/veterinarian.png", width=100)
    st.title("Navegação")
    
    # Informações do dataset carregado
    if st.session_state.df_main is not None:
        st.success(f"📊 Dataset: {len(st.session_state.df_main)} registros")
    
    # Navegação por páginas
    pagina = st.selectbox(
        "Selecione a página:",
        ["🏠 Visão Geral", "📊 Análise de Dados", "🤖 Treinar Modelo", "🔍 Predição", "📈 Estatísticas", "📁 Informações do Dataset"]
    )

# Título principal
st.markdown('<h1 class="main-header">🐾 VetDiagnosisAI</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Sistema Inteligente de Apoio ao Diagnóstico Veterinário</p>', unsafe_allow_html=True)

# Verificar se os dados estão carregados
if st.session_state.df_main is None:
    st.error("❌ Nenhum dataset carregado. Por favor, verifique os arquivos de dados.")
    st.stop()

df = st.session_state.df_main

# Navegação por páginas
if pagina == "🏠 Visão Geral":
    st.header("📊 Visão Geral do Sistema")
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Registros", len(df))
    
    with col2:
        especies = df['especie'].nunique() if 'especie' in df.columns else 0
        st.metric("Espécies", especies)
    
    with col3:
        diagnosticos = df['diagnostico'].nunique() if 'diagnostico' in df.columns else 0
        st.metric("Diagnósticos", diagnosticos)
    
    with col4:
        colunas = len(df.columns)
        st.metric("Variáveis", colunas)
    
    # Distribuição por espécie
    if 'especie' in df.columns:
        st.subheader("📊 Distribuição por Espécie")
        especie_counts = df['especie'].value_counts()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if PLOTLY_AVAILABLE:
                fig = px.pie(values=especie_counts.values, names=especie_counts.index, 
                            title="Distribuição por Espécie")
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Fallback com matplotlib
                fig, ax = plt.subplots()
                ax.pie(especie_counts.values, labels=especie_counts.index, autopct='%1.1f%%')
                ax.set_title("Distribuição por Espécie")
                st.pyplot(fig)
        
        with col2:
            st.dataframe(especie_counts.reset_index().rename(columns={'index': 'Espécie', 'especie': 'Quantidade'}))
    
    # Distribuição de diagnósticos
    if 'diagnostico' in df.columns:
        st.subheader("🏥 Distribuição de Diagnósticos")
        diag_counts = df['diagnostico'].value_counts().head(10)
        
        if PLOTLY_AVAILABLE:
            fig = px.bar(x=diag_counts.values, y=diag_counts.index, 
                        title="Top 10 Diagnósticos",
                        orientation='h')
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Fallback com matplotlib
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(diag_counts.index, diag_counts.values)
            ax.set_title("Top 10 Diagnósticos")
            ax.set_xlabel("Quantidade")
            st.pyplot(fig)

elif pagina == "📊 Análise de Dados":
    st.header("📊 Análise Detalhada dos Dados")
    
    # Filtros
    st.subheader("🔍 Filtros")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'especie' in df.columns:
            especies_filtro = st.multiselect("Espécie", df['especie'].unique(), default=df['especie'].unique())
        else:
            especies_filtro = []
    
    with col2:
        if 'idade_anos' in df.columns:
            idade_range = st.slider("Idade (anos)", 
                                  float(df['idade_anos'].min()), 
                                  float(df['idade_anos'].max()), 
                                  (float(df['idade_anos'].min()), float(df['idade_anos'].max())))
        else:
            idade_range = (0, 20)
    
    with col3:
        if 'diagnostico' in df.columns:
            diag_filtro = st.multiselect("Diagnóstico", df['diagnostico'].unique(), default=df['diagnostico'].unique())
        else:
            diag_filtro = []
    
    # Aplicar filtros
    df_filtrado = df.copy()
    
    if especies_filtro and 'especie' in df.columns:
        df_filtrado = df_filtrado[df_filtrado['especie'].isin(especies_filtro)]
    
    if 'idade_anos' in df.columns:
        df_filtrado = df_filtrado[
            (df_filtrado['idade_anos'] >= idade_range[0]) & 
            (df_filtrado['idade_anos'] <= idade_range[1])
        ]
    
    if diag_filtro and 'diagnostico' in df.columns:
        df_filtrado = df_filtrado[df_filtrado['diagnostico'].isin(diag_filtro)]
    
    st.info(f"📊 Mostrando {len(df_filtrado)} registros de {len(df)} totais")
    
    # Visualizações
    if len(df_filtrado) > 0:
        # Distribuição de idade
        if 'idade_anos' in df_filtrado.columns:
            st.subheader("📈 Distribuição de Idade")
            if PLOTLY_AVAILABLE:
                fig = px.histogram(df_filtrado, x='idade_anos', nbins=20, title="Distribuição de Idade")
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Fallback com matplotlib
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(df_filtrado['idade_anos'], bins=20, alpha=0.7)
                ax.set_title("Distribuição de Idade")
                ax.set_xlabel("Idade (anos)")
                ax.set_ylabel("Frequência")
                st.pyplot(fig)
        
        # Correlações entre variáveis numéricas
        numeric_cols = df_filtrado.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            st.subheader("🔗 Matriz de Correlação")
            corr_matrix = df_filtrado[numeric_cols].corr()
            
            if PLOTLY_AVAILABLE:
                fig = px.imshow(corr_matrix, 
                               text_auto=True, 
                               aspect="auto",
                               title="Matriz de Correlação")
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Fallback com matplotlib
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
                ax.set_title("Matriz de Correlação")
                st.pyplot(fig)
        
        # Tabela de dados
        st.subheader("📋 Dados Filtrados")
        st.dataframe(df_filtrado.head(100), use_container_width=True)

elif pagina == "🤖 Treinar Modelo":
    st.header("🚀 Gradient Boosting Otimizado - Sistema de Aprendizado Contínuo")
    
    if st.session_state.df_main is not None:
        df = st.session_state.df_main
        
        # Verificar se temos dados suficientes para ML
        if 'diagnostico' not in df.columns:
            st.error("❌ Coluna 'diagnostico' não encontrada. Não é possível treinar modelos.")
        else:
            # Dados disponíveis
            
            # Mostrar informações dos dados
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total de Registros", len(df))
            with col2:
                st.metric("Diagnósticos Únicos", df['diagnostico'].nunique())
            with col3:
                st.metric("Features Disponíveis", len(df.columns))
            
            # Preparar dados para ML
            st.subheader("🔧 Preparação dos Dados")
            
            # Feature Engineering Avançado
            df_ml = df.copy()
            
            # 1. Codificação de variáveis categóricas
            le_especie = LabelEncoder()
            le_sexo = LabelEncoder()
            le_diagnostico = LabelEncoder()
            
            if 'especie' in df_ml.columns:
                df_ml['especie_encoded'] = le_especie.fit_transform(df_ml['especie'])
            if 'sexo' in df_ml.columns:
                df_ml['sexo_encoded'] = le_sexo.fit_transform(df_ml['sexo'])
            
            df_ml['diagnostico_encoded'] = le_diagnostico.fit_transform(df_ml['diagnostico'])
            
            # 2. Criar features derivadas avançadas
            if 'idade_anos' in df_ml.columns:
                try:
                    df_ml['idade_categoria'] = pd.cut(df_ml['idade_anos'], bins=[0, 1, 3, 7, 12, 100], labels=['Filhote', 'Jovem', 'Adulto', 'Maduro', 'Idoso'])
                    df_ml['idade_categoria_encoded'] = LabelEncoder().fit_transform(df_ml['idade_categoria'])
                except Exception as e:
                    # Erro ao criar categoria de idade
                    # Usar categorização simples como fallback
                    df_ml['idade_categoria_encoded'] = (df_ml['idade_anos'] // 5).astype(int)
                
                # Features de idade
                try:
                    df_ml['idade_quadrado'] = df_ml['idade_anos'] ** 2
                    df_ml['idade_log'] = np.log1p(df_ml['idade_anos'])
                    df_ml['idade_senior'] = (df_ml['idade_anos'] > 7).astype(int)
                    df_ml['idade_filhote'] = (df_ml['idade_anos'] < 1).astype(int)
                except Exception as e:
                    # Erro ao criar features de idade
            
            # 3. Features de exames laboratoriais combinados avançados
            exames_cols = ['hemoglobina', 'hematocrito', 'leucocitos', 'glicose', 'ureia', 'creatinina', 'alt', 'ast', 'fosfatase_alcalina', 'proteinas_totais', 'albumina']
            exames_disponiveis = [col for col in exames_cols if col in df_ml.columns]
            
            if len(exames_disponiveis) >= 3:
                # Criar índices clínicos específicos
                if 'hemoglobina' in df_ml.columns and 'hematocrito' in df_ml.columns:
                    df_ml['indice_anemia'] = df_ml['hemoglobina'] / (df_ml['hematocrito'] / 3)
                    df_ml['anemia_grave'] = (df_ml['hemoglobina'] < 8).astype(int)
                
                if 'ureia' in df_ml.columns and 'creatinina' in df_ml.columns:
                    df_ml['indice_renal'] = df_ml['ureia'] / df_ml['creatinina']
                    df_ml['insuficiencia_renal'] = ((df_ml['ureia'] > 60) | (df_ml['creatinina'] > 2)).astype(int)
                
                if 'glicose' in df_ml.columns:
                    df_ml['diabetes'] = (df_ml['glicose'] > 150).astype(int)
                    df_ml['hipoglicemia'] = (df_ml['glicose'] < 60).astype(int)
                
                if 'leucocitos' in df_ml.columns:
                    df_ml['leucocitose'] = (df_ml['leucocitos'] > 12000).astype(int)
                    df_ml['leucopenia'] = (df_ml['leucocitos'] < 4000).astype(int)
            
            # 4. Features de sintomas combinados avançados
            sintomas_cols = ['febre', 'apatia', 'perda_peso', 'vomito', 'diarreia', 'tosse', 'letargia', 'feridas_cutaneas', 'poliuria', 'polidipsia']
            sintomas_disponiveis = [col for col in sintomas_cols if col in df_ml.columns]
            
            if len(sintomas_disponiveis) >= 2:
                try:
                    df_ml['total_sintomas'] = df_ml[sintomas_disponiveis].sum(axis=1)
                    df_ml['severidade_sintomas'] = pd.cut(df_ml['total_sintomas'], bins=[-1, 0, 1, 3, 5, 10], labels=['Assintomático', 'Leve', 'Moderado', 'Severo', 'Crítico'])
                    df_ml['severidade_sintomas_encoded'] = LabelEncoder().fit_transform(df_ml['severidade_sintomas'])
                except Exception as e:
                    # Erro ao criar features de sintomas
                    # Fallback simples
                    df_ml['total_sintomas'] = df_ml[sintomas_disponiveis].sum(axis=1)
                    df_ml['severidade_sintomas_encoded'] = (df_ml['total_sintomas'] > 2).astype(int)
                
                # Síndromes específicas
                if all(col in df_ml.columns for col in ['febre', 'tosse']):
                    df_ml['sindrome_respiratoria'] = (df_ml['febre'] & df_ml['tosse']).astype(int)
                
                if all(col in df_ml.columns for col in ['vomito', 'diarreia']):
                    df_ml['sindrome_gastrointestinal'] = (df_ml['vomito'] | df_ml['diarreia']).astype(int)
                
                if all(col in df_ml.columns for col in ['poliuria', 'polidipsia']):
                    df_ml['sindrome_polidipsica'] = (df_ml['poliuria'] & df_ml['polidipsia']).astype(int)
                
                if all(col in df_ml.columns for col in ['apatia', 'perda_peso']):
                    df_ml['sindrome_sistemica'] = (df_ml['apatia'] & df_ml['perda_peso']).astype(int)
            
            # Selecionar features para ML
            feature_cols = []
            
            try:
                # Adicionar colunas numéricas originais
                numeric_cols = df_ml.select_dtypes(include=[np.number]).columns.tolist()
                feature_cols.extend([col for col in numeric_cols if col not in ['diagnostico_encoded']])
                
                # Remover colunas com muitos valores únicos (como ID)
                feature_cols = [col for col in feature_cols if df_ml[col].nunique() < len(df_ml) * 0.8]
                
                # Verificar se temos features suficientes
                if len(feature_cols) < 3:
                    # Poucas features disponíveis
                    feature_cols = [col for col in numeric_cols if col not in ['diagnostico_encoded']]
                
                X = df_ml[feature_cols].fillna(df_ml[feature_cols].mean())
                y = df_ml['diagnostico_encoded']
                
            except Exception as e:
                st.error(f"❌ Erro na preparação dos dados: {e}")
                st.stop()
            
            # Dados preparados
            
            # Dividir dados
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            # Escalar features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Divisão dos dados
            
            # Sistema de Gradient Boosting Otimizado
            st.subheader("🚀 Gradient Boosting Ultra-Otimizado")
            # Treinamento do modelo
            
            # Configurações avançadas
            col1, col2 = st.columns(2)
            with col1:
                use_advanced_features = st.checkbox("🔧 Feature Engineering Avançado", value=True)
                use_feature_selection = st.checkbox("🎯 Seleção de Features", value=True)
            with col2:
                use_hyperparameter_tuning = st.checkbox("⚙️ Otimização de Hiperparâmetros", value=True)
                save_model = st.checkbox("💾 Salvar Modelo Treinado", value=True)
            
            # Feature Engineering Avançado
            if use_advanced_features:
                st.subheader("🔧 Feature Engineering Avançado")
                
                # Criar features polinomiais
                try:
                    from sklearn.preprocessing import PolynomialFeatures
                    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
                    X_poly = poly.fit_transform(X)
                    # Features polinomiais criadas
                    X = X_poly
                except Exception as e:
                    # Erro ao criar features polinomiais
            
            # Seleção de Features
            if use_feature_selection:
                st.subheader("🎯 Seleção de Features")
                
                # Usar SelectKBest para selecionar as melhores features
                try:
                    k_best = min(50, X.shape[1])  # Máximo 50 features ou todas se menos
                    selector = SelectKBest(score_func=f_classif, k=k_best)
                    X_selected = selector.fit_transform(X, y)
                    selected_features = selector.get_support(indices=True)
                    # Features selecionadas
                    X = X_selected
                except Exception as e:
                    # Erro na seleção de features
            
            # Gradient Boosting Ultra-Otimizado
            st.subheader("🎯 Gradient Boosting Ultra-Otimizado")
            
            # Hiperparâmetros otimizados para alta performance
            gb_params = {
                'n_estimators': 1000,
                'learning_rate': 0.01,
                'max_depth': 12,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'subsample': 0.8,
                'max_features': 'sqrt',
                'random_state': 42,
                'validation_fraction': 0.1,
                'n_iter_no_change': 50,
                'tol': 1e-4
            }
            
            # Otimização adicional de hiperparâmetros
            if use_hyperparameter_tuning:
                # Otimizando hiperparâmetros
                
                # Grid de hiperparâmetros para otimização
                param_grid = {
                    'n_estimators': [800, 1000, 1200],
                    'learning_rate': [0.005, 0.01, 0.02],
                    'max_depth': [10, 12, 15],
                    'subsample': [0.7, 0.8, 0.9],
                    'min_samples_split': [2, 3, 5]
                }
                
                # RandomizedSearchCV para otimização
                gb_base = GradientBoostingClassifier(random_state=42)
                random_search = RandomizedSearchCV(
                    gb_base, param_grid, n_iter=20, cv=5, 
                    scoring='accuracy', random_state=42, n_jobs=-1
                )
                
                with st.spinner("🔄 Otimizando hiperparâmetros..."):
                    random_search.fit(X_train_scaled, y_train)
                
                # Usar os melhores parâmetros encontrados
                gb_params.update(random_search.best_params_)
                # Melhores parâmetros encontrados
            
            # Criar modelo final otimizado
            gb_model = GradientBoostingClassifier(**gb_params)
            
            # Treinar modelo Gradient Boosting otimizado
            # Treinando modelo
            
            with st.spinner("🚀 Treinando modelo com 1000 estimadores..."):
                # Treinar modelo
                gb_model.fit(X_train_scaled, y_train)
                
                # Fazer predições
                y_pred = gb_model.predict(X_test_scaled)
                y_pred_proba = gb_model.predict_proba(X_test_scaled)
            
            # Calcular métricas detalhadas
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            precision = precision_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred, average='macro')
            
            # Validação cruzada estratificada
            cv_scores = cross_val_score(gb_model, X_train_scaled, y_train, cv=10, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Modelo treinado
            
            # Salvar modelo se solicitado
            if save_model:
                try:
                    # Salvar modelo e scaler
                    model_data = {
                        'model': gb_model,
                        'scaler': scaler,
                        'feature_names': list(df_ml.columns),
                        'target_names': le_diagnostico.classes_,
                        'accuracy': accuracy,
                        'cv_mean': cv_mean,
                        'timestamp': datetime.now().isoformat(),
                        'training_samples': len(X_train),
                        'test_samples': len(X_test)
                    }
                    
                    # Salvar usando joblib
                    model_path = Path("models")
                    model_path.mkdir(exist_ok=True)
                    
                    joblib.dump(model_data, model_path / "gb_optimized_model.pkl")
                    # Modelo salvo
                    
                    # Salvar também no session state para uso imediato
                    st.session_state.gb_model = gb_model
                    st.session_state.scaler = scaler
                    st.session_state.le_diagnostico = le_diagnostico
                    st.session_state.model_trained = True
                    
                except Exception as e:
                    # Erro ao salvar modelo
                    # Salvar pelo menos no session state
                    st.session_state.gb_model = gb_model
                    st.session_state.scaler = scaler
                    st.session_state.le_diagnostico = le_diagnostico
                    st.session_state.model_trained = True
            
            # Mostrar resultados do Gradient Boosting
            st.subheader("🎯 Resultados do Gradient Boosting Ultra-Otimizado")
            
            # Métricas principais
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🎯 Acurácia", f"{accuracy:.1%}", delta=f"{accuracy-0.85:.1%}" if accuracy > 0.85 else None)
            with col2:
                st.metric("📊 F1-Score", f"{f1:.3f}")
            with col3:
                st.metric("🎪 Precision", f"{precision:.3f}")
            with col4:
                st.metric("🎭 Recall", f"{recall:.3f}")
            
            # Validação cruzada
            col1, col2 = st.columns(2)
            with col1:
                st.metric("✅ CV Mean (10-fold)", f"{cv_mean:.3f}")
            with col2:
                st.metric("📈 CV Std", f"{cv_std:.3f}")
            
            # Status da meta de 85%
            if accuracy >= 0.85:
                st.success(f"🎉 META ALCANÇADA! Acurácia de {accuracy:.1%} >= 85%!")
            else:
                st.warning(f"🎯 Meta: 85% | Atual: {accuracy:.1%} | Faltam: {(0.85-accuracy)*100:.1f}%")
            
            # Feature Importance
            st.subheader("🎯 Importância das Features")
            
            if hasattr(gb_model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'Feature': [f'Feature_{i}' for i in range(len(gb_model.feature_importances_))],
                    'Importance': gb_model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                # Top 15 features mais importantes
                top_features = feature_importance.head(15)
                
                if PLOTLY_AVAILABLE:
                    fig = px.bar(
                        top_features, 
                        x='Importance', 
                        y='Feature',
                        orientation='h',
                        title='Top 15 Features Mais Importantes',
                        color='Importance',
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Fallback com matplotlib
                    fig, ax = plt.subplots(figsize=(10, 8))
                    ax.barh(top_features['Feature'], top_features['Importance'])
                    ax.set_title('Top 15 Features Mais Importantes')
                    ax.set_xlabel('Importância')
                    st.pyplot(fig)
                
                # Tabela de importância
                st.dataframe(top_features, use_container_width=True)
            
            # Confusion Matrix
            st.subheader("🎯 Matriz de Confusão")
            
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            
            if PLOTLY_AVAILABLE:
                fig = px.imshow(
                    cm, 
                    text_auto=True, 
                    aspect="auto",
                    title="Matriz de Confusão - Gradient Boosting",
                    labels=dict(x="Predito", y="Real", color="Quantidade")
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Fallback com matplotlib
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title("Matriz de Confusão - Gradient Boosting")
                ax.set_xlabel("Predito")
                ax.set_ylabel("Real")
                st.pyplot(fig)
            
            # Classification Report
            st.subheader("📊 Relatório de Classificação")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)
            
            # Sistema de Aprendizado Contínuo
            st.subheader("🧠 Sistema de Aprendizado Contínuo")
            
            col1, col2 = st.columns(2)
            with col1:
                st.info("📈 **Funcionalidades Implementadas:**")
                st.write("✅ Modelo salvo automaticamente")
                st.write("✅ Hiperparâmetros otimizados")
                st.write("✅ Feature engineering avançado")
                st.write("✅ Validação cruzada 10-fold")
                st.write("✅ Persistência no session state")
            
            with col2:
                st.info("🚀 **Próximos Passos:**")
                st.write("🔄 Retreinamento incremental")
                st.write("📊 Monitoramento de performance")
                st.write("🎯 Ajuste automático de parâmetros")
                st.write("📈 Análise de drift de dados")
                st.write("🔧 Auto-tuning contínuo")
            
            # Sugestões para melhorar ainda mais
            st.subheader("💡 Sugestões para Atingir 85%+ de Acurácia")
            
            if accuracy < 0.85:
                st.warning("🎯 **Para alcançar 85%+ de acurácia:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("🔧 **Feature Engineering:**")
                    st.write("• Criar mais features derivadas")
                    st.write("• Combinar exames laboratoriais")
                    st.write("• Agrupar sintomas por severidade")
                    st.write("• Usar idade categorizada")
                    st.write("• Criar índices clínicos específicos")
                
                with col2:
                    st.write("🚀 **Modelos Avançados:**")
                    st.write("• XGBoost com hiperparâmetros otimizados")
                    st.write("• Ensemble de múltiplos modelos")
                    st.write("• Validação cruzada estratificada")
                    st.write("• Balanceamento de classes")
                    st.write("• Seleção de features automática")
            else:
                st.success("🎉 **Meta alcançada!** Continue adicionando dados para melhorar ainda mais!")
    
    else:
        st.error("❌ Nenhum dataset carregado. Por favor, carregue um dataset primeiro.")

elif pagina == "🔍 Predição":
    st.header("🔍 Predição com Gradient Boosting Otimizado")
    
    if hasattr(st.session_state, 'gb_model') and st.session_state.gb_model is not None:
        # Modelo carregado
        
        # Formulário de entrada
        st.subheader("📝 Dados do Paciente")
        
        col1, col2 = st.columns(2)
        
        with col1:
            especie = st.selectbox("Espécie", ["Cão", "Gato"])
            idade = st.number_input("Idade (anos)", min_value=0.1, max_value=30.0, value=5.0)
            sexo = st.selectbox("Sexo", ["M", "F"])
            
            # Exames laboratoriais
            st.subheader("🧪 Exames Laboratoriais")
            hemoglobina = st.number_input("Hemoglobina (g/dL)", min_value=5.0, max_value=20.0, value=12.0)
            hematocrito = st.number_input("Hematócrito (%)", min_value=15.0, max_value=60.0, value=40.0)
            leucocitos = st.number_input("Leucócitos (/μL)", min_value=1000, max_value=30000, value=8000)
            glicose = st.number_input("Glicose (mg/dL)", min_value=50.0, max_value=400.0, value=100.0)
        
        with col2:
            ureia = st.number_input("Ureia (mg/dL)", min_value=10.0, max_value=200.0, value=30.0)
            creatinina = st.number_input("Creatinina (mg/dL)", min_value=0.5, max_value=10.0, value=1.2)
            alt = st.number_input("ALT (U/L)", min_value=10.0, max_value=500.0, value=40.0)
            ast = st.number_input("AST (U/L)", min_value=10.0, max_value=400.0, value=30.0)
            
            # Sintomas
            st.subheader("🩺 Sintomas")
            febre = st.checkbox("Febre")
            apatia = st.checkbox("Apatia")
            perda_peso = st.checkbox("Perda de Peso")
            vomito = st.checkbox("Vômito")
            diarreia = st.checkbox("Diarreia")
            tosse = st.checkbox("Tosse")
            letargia = st.checkbox("Letargia")
            feridas_cutaneas = st.checkbox("Feridas Cutâneas")
            poliuria = st.checkbox("Poliúria")
            polidipsia = st.checkbox("Polidipsia")
        
        # Botão de predição
        if st.button("🔮 Realizar Predição", type="primary"):
            # Preparar dados para predição
            dados_paciente = {
                'especie': especie,
                'idade_anos': idade,
                'sexo': sexo,
                'hemoglobina': hemoglobina,
                'hematocrito': hematocrito,
                'leucocitos': leucocitos,
                'glicose': glicose,
                'ureia': ureia,
                'creatinina': creatinina,
                'alt': alt,
                'ast': ast,
                'febre': 1 if febre else 0,
                'apatia': 1 if apatia else 0,
                'perda_peso': 1 if perda_peso else 0,
                'vomito': 1 if vomito else 0,
                'diarreia': 1 if diarreia else 0,
                'tosse': 1 if tosse else 0,
                'letargia': 1 if letargia else 0,
                'feridas_cutaneas': 1 if feridas_cutaneas else 0,
                'poliuria': 1 if poliuria else 0,
                'polidipsia': 1 if polidipsia else 0
            }
            
            # Fazer predição
            try:
                # Aqui você implementaria a lógica de predição
                # Predição realizada
                
            except Exception as e:
                st.error(f"❌ Erro na predição: {str(e)}")
    
    else:
        # Nenhum modelo treinado

elif pagina == "📈 Estatísticas":
    st.header("📈 Estatísticas Detalhadas")
    
    # Estatísticas descritivas
    st.subheader("📊 Estatísticas Descritivas")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        st.dataframe(df[numeric_cols].describe(), use_container_width=True)
    
    # Distribuições
    st.subheader("📈 Distribuições")
    
    # Selecionar variável para análise
    if len(numeric_cols) > 0:
        var_analise = st.selectbox("Selecione uma variável para análise", numeric_cols)
        
        col1, col2 = st.columns(2)

        with col1:
            # Histograma
            if PLOTLY_AVAILABLE:
                fig = px.histogram(df, x=var_analise, nbins=30, title=f"Distribuição de {var_analise}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Fallback com matplotlib
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.hist(df[var_analise], bins=30, alpha=0.7)
                ax.set_title(f"Distribuição de {var_analise}")
                ax.set_xlabel(var_analise)
                ax.set_ylabel("Frequência")
                st.pyplot(fig)
        
        with col2:
            # Box plot
            if PLOTLY_AVAILABLE:
                fig = px.box(df, y=var_analise, title=f"Box Plot de {var_analise}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Fallback com matplotlib
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.boxplot(df[var_analise])
                ax.set_title(f"Box Plot de {var_analise}")
                ax.set_ylabel(var_analise)
                st.pyplot(fig)
    
    # Análise por diagnóstico
    if 'diagnostico' in df.columns and len(numeric_cols) > 0:
        st.subheader("🏥 Análise por Diagnóstico")
        
        diag_selecionado = st.selectbox("Selecione um diagnóstico", df['diagnostico'].unique())
        df_diag = df[df['diagnostico'] == diag_selecionado]
        
        # Casos do diagnóstico
        
        if len(df_diag) > 0:
            # Estatísticas do diagnóstico selecionado
            st.dataframe(df_diag[numeric_cols].describe(), use_container_width=True)

elif pagina == "📁 Informações do Dataset":
    st.header("📁 Informações do Dataset")
    
    # Informações básicas
    st.subheader("📊 Informações Básicas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total de Registros", len(df))
        st.metric("Total de Colunas", len(df.columns))
        st.metric("Memória Usada", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    with col2:
        st.metric("Registros com Valores Nulos", df.isnull().sum().sum())
        st.metric("Tipos de Dados Únicos", df.dtypes.nunique())
        st.metric("Colunas Numéricas", len(df.select_dtypes(include=[np.number]).columns))
    
    # Estrutura do dataset
    st.subheader("🏗️ Estrutura do Dataset")
    
    # Tipos de dados
    st.write("**Tipos de Dados:**")
    tipos_dados = df.dtypes.value_counts()
    st.dataframe(tipos_dados.reset_index().rename(columns={'index': 'Tipo', 0: 'Quantidade'}), use_container_width=True)
    
    # Colunas e tipos
    st.write("**Colunas e Tipos:**")
    colunas_info = pd.DataFrame({
        'Coluna': df.columns,
        'Tipo': df.dtypes,
        'Valores Únicos': df.nunique(),
        'Valores Nulos': df.isnull().sum()
    })
    st.dataframe(colunas_info, use_container_width=True)
    
    # Amostra dos dados
    st.subheader("👀 Amostra dos Dados")
    st.write("**Primeiras 10 linhas:**")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Valores únicos por coluna categórica
    st.subheader("📋 Valores Únicos")
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            if df[col].nunique() <= 20:  # Só mostrar se não tiver muitos valores únicos
                st.write(f"**{col}:** {list(df[col].unique())}")
            else:
                st.write(f"**{col}:** {df[col].nunique()} valores únicos")

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #666;">🐾 VetDiagnosisAI - Sistema Inteligente de Apoio ao Diagnóstico Veterinário</p>',
    unsafe_allow_html=True
)
