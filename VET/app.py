"""
VetDiagnosisAI - App Simples para Veterinários
Interface focada apenas em predições rápidas
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from pathlib import Path
from datetime import datetime
import sys
import traceback

# Configuração da página
st.set_page_config(
    page_title="VetDiagnosisAI - Predição Rápida",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personalizado para interface limpa e moderna
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
        font-size: 2.5rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 1px solid #ffb347;
    }
    .form-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #28a745;
    }
    .symptom-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 10px;
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    .success-message {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
    }
    /* Esconder sidebar completamente */
    section[data-testid="stSidebar"] {display: none !important;}
    .stApp > div:first-child {padding-left: 1rem !important;}
    div[data-testid="stSidebar"] {display: none !important;}
    .css-1d391kg {display: none !important;}
    .css-1v0mbdj {display: none !important;}
    .css-1cypcdb {display: none !important;}
    .css-1v3fvcr {display: none !important;}
    .stApp > div:first-child > div:first-child {display: none !important;}
</style>
""", unsafe_allow_html=True)

# Header principal
st.markdown('<h1 class="main-header">🐾 VetDiagnosisAI - Predição Rápida</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666;">Sistema Inteligente de Apoio ao Diagnóstico Veterinário</p>', unsafe_allow_html=True)

# Função para carregar modelo
@st.cache_data
def carregar_modelo():
    """Carrega o modelo treinado"""
    try:
        # Lista de caminhos possíveis para o modelo (Streamlit Cloud compatível)
        possible_paths = [
            "models/model_minimal.pkl",
            "models/gb_model_optimized.pkl",
            "models/gb_optimized_model.pkl",
            "./models/model_minimal.pkl",
            "./models/gb_model_optimized.pkl",
            "./models/gb_optimized_model.pkl"
        ]
        
        model_data = None
        found_path = None
        
        for model_path in possible_paths:
            if Path(model_path).exists():
                found_path = model_path
                model_data = joblib.load(model_path)
                break
        
        if model_data is not None:
            st.success(f"✅ Modelo encontrado em: {found_path}")
            return model_data
        else:
            st.error("❌ Modelo não encontrado em nenhum dos caminhos:")
            for path in possible_paths:
                exists = "✅" if Path(path).exists() else "❌"
                st.write(f"  {exists} {path}")
            
            st.info(f"📁 Diretório atual: {Path.cwd()}")
            st.info(f"📂 Conteúdo do diretório: {list(Path('.').iterdir())}")
            
            # Verificar se existe pasta models
            if Path("models").exists():
                st.info(f"📂 Conteúdo da pasta models: {list(Path('models').iterdir())}")
            
            return None
        
    except Exception as e:
        st.error(f"❌ Erro ao carregar modelo: {e}")
        st.code(traceback.format_exc())
        return None

# Carregar modelo
with st.spinner("🔄 Carregando modelo..."):
    model_data = carregar_modelo()

if model_data is None:
    st.error("❌ Não foi possível carregar o modelo!")
    st.info("📧 Verifique se o arquivo do modelo existe e tente novamente.")
    
    # Mostrar informações de debug
    with st.expander("🔍 Informações de Debug", expanded=True):
        st.write("**Diretório atual:**", Path.cwd())
        st.write("**Arquivos no diretório:**", list(Path('.').iterdir()))
        if Path("models").exists():
            st.write("**Arquivos em models/:**", list(Path("models").iterdir()))
        else:
            st.write("❌ Pasta 'models' não encontrada")
    
    st.stop()

# Extrair componentes do modelo
modelo = model_data['model']
scaler = model_data['scaler']
le_diagnostico = model_data['le_diagnostico']
accuracy = model_data.get('accuracy', 0)
training_date = model_data.get('timestamp', 'N/A')

# Mostrar informações do modelo
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("🎯 Acurácia do Modelo", f"{accuracy:.1%}")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("📊 Última Atualização", training_date.split('T')[0] if 'T' in training_date else training_date)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("🧠 Tipo de Modelo", "Gradient Boosting")
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# Formulário de predição
st.subheader("🔍 Predição de Diagnóstico")

# Dividir em colunas para melhor organização
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="form-section">', unsafe_allow_html=True)
    st.markdown("**📋 Dados Básicos do Animal**")
    
    especie = st.selectbox(
        "Espécie",
        options=["Canina", "Felina", "Equina", "Bovino", "Suíno", "Ave", "Outro"],
        help="Selecione a espécie do animal"
    )
    
    raca = st.text_input(
        "Raça",
        placeholder="Ex: Labrador, Persa, SRD...",
        help="Digite a raça do animal"
    )
    
    idade_anos = st.number_input(
        "Idade (anos)",
        min_value=0.0,
        max_value=30.0,
        value=1.0,
        step=0.1,
        help="Idade do animal em anos"
    )
    
    sexo = st.selectbox(
        "Sexo",
        options=["M", "F"],
        help="Sexo do animal"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="form-section">', unsafe_allow_html=True)
    st.markdown("**🧪 Exames Laboratoriais**")
    
    hemoglobina = st.number_input(
        "Hemoglobina (g/dL)",
        min_value=0.0,
        max_value=30.0,
        value=12.0,
        step=0.1,
        help="Valor de hemoglobina"
    )
    
    hematocrito = st.number_input(
        "Hematócrito (%)",
        min_value=0.0,
        max_value=100.0,
        value=40.0,
        step=0.1,
        help="Valor de hematócrito"
    )
    
    leucocitos = st.number_input(
        "Leucócitos (x10³/μL)",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=0.1,
        help="Contagem de leucócitos"
    )
    
    glicose = st.number_input(
        "Glicose (mg/dL)",
        min_value=0.0,
        max_value=500.0,
        value=100.0,
        step=1.0,
        help="Nível de glicose"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Segunda linha de exames
col3, col4 = st.columns(2)

with col3:
    st.markdown('<div class="form-section">', unsafe_allow_html=True)
    st.markdown("**🔬 Mais Exames**")
    
    ureia = st.number_input(
        "Ureia (mg/dL)",
        min_value=0.0,
        max_value=200.0,
        value=30.0,
        step=1.0,
        help="Nível de ureia"
    )
    
    creatinina = st.number_input(
        "Creatinina (mg/dL)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1,
        help="Nível de creatinina"
    )
    
    alt = st.number_input(
        "ALT (U/L)",
        min_value=0.0,
        max_value=1000.0,
        value=50.0,
        step=1.0,
        help="Alanina aminotransferase"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="form-section">', unsafe_allow_html=True)
    st.markdown("**🏥 Sintomas Clínicos**")
    
    # Sintomas como checkboxes
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
    st.markdown('</div>', unsafe_allow_html=True)

# Botão de predição
if st.button("🔍 Realizar Predição", type="primary", use_container_width=True):
    
    # Preparar dados para predição
    try:
        # Converter sintomas para valores binários
        sintomas = [febre, apatia, perda_peso, vomito, diarreia, tosse, letargia, feridas_cutaneas, poliuria, polidipsia]
        sintomas_values = [1 if s else 0 for s in sintomas]
        
        # Criar array com todos os dados
        dados_predicao = np.array([
            especie == "Canina", especie == "Felina",  # One-hot encoding para espécie
            idade_anos,
            sexo == "M",  # 1 para macho, 0 para fêmea
            hemoglobina, hematocrito, leucocitos, 10.0,  # Plaquetas padrão
            glicose, ureia, creatinina, alt, 50.0,  # AST padrão
            100.0, 7.0, 3.5, 200.0, 100.0, 2.0  # Valores padrão para outros exames
        ] + sintomas_values).reshape(1, -1)
        
        # Fazer predição
        predicao = modelo.predict(dados_predicao)
        probabilidades = modelo.predict_proba(dados_predicao)
        
        # Obter diagnóstico
        diagnostico_predito = le_diagnostico.inverse_transform(predicao)[0]
        confianca = max(probabilidades[0]) * 100
        
        # Mostrar resultado
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        st.markdown(f"### 🎯 **Diagnóstico Predito: {diagnostico_predito}**")
        st.markdown(f"### 📊 **Confiança: {confianca:.1f}%**")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Mostrar probabilidades de todos os diagnósticos
        st.subheader("📈 Probabilidades por Diagnóstico")
        
        probabilidades_df = pd.DataFrame({
            'Diagnóstico': le_diagnostico.classes_,
            'Probabilidade (%)': probabilidades[0] * 100
        }).sort_values('Probabilidade (%)', ascending=False)
        
        # Gráfico de barras das probabilidades
        import plotly.express as px
        fig = px.bar(
            probabilidades_df.head(5),
            x='Probabilidade (%)',
            y='Diagnóstico',
            orientation='h',
            title='Top 5 Diagnósticos Mais Prováveis',
            color='Probabilidade (%)',
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabela com todas as probabilidades
        st.dataframe(probabilidades_df, use_container_width=True)
        
        # Log da predição para análise posterior
        log_predicao = {
            'timestamp': datetime.now().isoformat(),
            'especie': especie,
            'idade': idade_anos,
            'sexo': sexo,
            'sintomas': sintomas,
            'diagnostico_predito': diagnostico_predito,
            'confianca': confianca,
            'probabilidades': probabilidades[0].tolist()
        }
        
        # Salvar log (implementar sistema de logging posteriormente)
        st.markdown('<div class="success-message">✅ Predição realizada com sucesso!</div>', unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ Erro na predição: {e}")
        st.info("Por favor, verifique os dados inseridos e tente novamente.")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>🐾 VetDiagnosisAI - Sistema Inteligente de Diagnóstico Veterinário</p>
    <p><small>Para dúvidas ou suporte, entre em contato com o administrador do sistema.</small></p>
</div>
""", unsafe_allow_html=True)
