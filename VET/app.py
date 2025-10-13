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
import requests
import json
import os

# Configuração da página
st.set_page_config(
    page_title="VetDiagnosisAI - Sistema Inteligente",
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
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        max-width: 80%;
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: auto;
        text-align: right;
    }
    .assistant-message {
        background: linear-gradient(135deg, #f0f2f6 0%, #e8f4f8 100%);
        color: #333;
        margin-right: auto;
    }
    .chat-container {
        height: 400px;
        overflow-y: auto;
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        background: #fafafa;
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
st.markdown('<h1 class="main-header">🐾 VetDiagnosisAI - Sistema Inteligente</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666;">Sistema Inteligente de Apoio ao Diagnóstico Veterinário com IA Conversacional</p>', unsafe_allow_html=True)

# Inicializar session state para chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chat_tabs" not in st.session_state:
    st.session_state.chat_tabs = ["Chat Principal"]

# Função para chamar DeepSeek API
def call_deepseek_api(message, context=""):
    """Chama a API do DeepSeek para obter resposta inteligente"""
    try:
        # Configuração da API (você pode adicionar sua chave API aqui)
        api_key = os.getenv("DEEPSEEK_API_KEY", "sk-your-api-key-here")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Contexto veterinário para melhorar as respostas
        system_prompt = f"""Você é um assistente veterinário inteligente especializado em diagnóstico e tratamento animal. 
        
        Contexto atual: {context}
        
        Responda de forma técnica mas acessível, sempre considerando:
        - Sintomas apresentados
        - Possíveis diagnósticos diferenciais
        - Recomendações de exames complementares
        - Tratamentos sugeridos
        - Quando encaminhar para especialista
        
        Seja preciso, empático e educativo."""
        
        data = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        # Se não tiver API key, usar resposta simulada
        if api_key == "sk-your-api-key-here":
            return f"🤖 **Assistente Veterinário IA**\n\nBaseado na sua pergunta sobre '{message}', aqui estão algumas considerações importantes:\n\n• **Sintomas observados:** Analise detalhadamente todos os sintomas apresentados\n• **Exames complementares:** Considere hemograma, bioquímica e exames específicos\n• **Diagnóstico diferencial:** Liste as principais hipóteses diagnósticas\n• **Tratamento:** Inicie tratamento sintomático enquanto aguarda confirmação\n\n*Para respostas mais precisas, configure sua chave API do DeepSeek nas configurações.*"
        
        response = requests.post("https://api.deepseek.com/v1/chat/completions", 
                               headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            return f"❌ Erro na API: {response.status_code}"
            
    except Exception as e:
        return f"❌ Erro ao conectar com IA: {str(e)}"

def obter_recomendacoes_personalizadas(diagnostico, sintomas, dados_laboratoriais):
    """Obtém recomendações personalizadas do DeepSeek baseadas no caso específico"""
    try:
        # Montar contexto do caso
        contexto = f"""
        CASO VETERINÁRIO:
        - Diagnóstico: {diagnostico}
        - Sintomas: {', '.join(sintomas)}
        - Dados laboratoriais: {dados_laboratoriais}
        
        Baseado neste caso específico, forneça recomendações detalhadas para:
        1. Exames complementares prioritários
        2. Protocolo de medicação com doses específicas
        3. Protocolo cirúrgico se necessário
        4. Protocolo de anestesia com medicações e doses
        5. Monitoramento pós-operatório
        
        Seja específico com doses, frequências e durações baseado na literatura veterinária.
        """
        
        resposta = call_deepseek_api(contexto)
        return resposta
        
    except Exception as e:
        return f"❌ Erro ao obter recomendações personalizadas: {str(e)}"

# Funções para sugestões baseadas no diagnóstico
def sugerir_doencas(diagnostico):
    """Sugere doenças relacionadas baseadas no diagnóstico"""
    doencas_sugeridas = {
        "Infecção": ["Sepse", "Pneumonia", "Cistite", "Dermatite", "Otite"],
        "Intoxicação": ["Envenenamento", "Insuficiência hepática", "Nefrotoxicidade", "Gastroenterite tóxica"],
        "Trauma": ["Fraturas", "Hemorragia interna", "Concussão", "Lacerações", "Hematomas"],
        "Tumor": ["Carcinoma", "Sarcoma", "Linfoma", "Adenoma", "Melanoma"],
        "Doença renal": ["Insuficiência renal", "Nefrite", "Cálculos renais", "Glomerulonefrite"],
        "Doença cardíaca": ["Cardiomiopatia", "Arritmia", "Insuficiência cardíaca", "Endocardite"],
        "Diabetes": ["Cetoacidose", "Hipoglicemia", "Retinopatia", "Nefropatia diabética"]
    }
    return doencas_sugeridas.get(diagnostico, ["Diagnóstico a confirmar com exames complementares"])

def sugerir_medicamentos(diagnostico):
    """Sugere medicamentos baseados no diagnóstico"""
    medicamentos = {
        "Infecção": [
            {"nome": "Amoxicilina", "dose": "10-20 mg/kg", "frequencia": "2x/dia", "duracao": "7-10 dias"},
            {"nome": "Ceftriaxona", "dose": "25 mg/kg", "frequencia": "1x/dia", "duracao": "5-7 dias"},
            {"nome": "Metronidazol", "dose": "10-15 mg/kg", "frequencia": "2x/dia", "duracao": "5-7 dias"}
        ],
        "Intoxicação": [
            {"nome": "Carvão ativado", "dose": "1-3 g/kg", "frequencia": "Imediato", "duracao": "1 dose"},
            {"nome": "Fluidos IV", "dose": "10-20 ml/kg/h", "frequencia": "Contínuo", "duracao": "24-48h"},
            {"nome": "Protetor hepático", "dose": "20-50 mg/kg", "frequencia": "2x/dia", "duracao": "7-14 dias"}
        ],
        "Trauma": [
            {"nome": "Morfina", "dose": "0.1-0.3 mg/kg", "frequencia": "4-6x/dia", "duracao": "3-5 dias"},
            {"nome": "Anti-inflamatório", "dose": "0.2 mg/kg", "frequencia": "1x/dia", "duracao": "3-5 dias"},
            {"nome": "Antibiótico profilático", "dose": "10 mg/kg", "frequencia": "2x/dia", "duracao": "5-7 dias"}
        ],
        "Tumor": [
            {"nome": "Quimioterapia", "dose": "Conforme protocolo", "frequencia": "Semanal", "duracao": "4-6 ciclos"},
            {"nome": "Corticosteroides", "dose": "0.5-1 mg/kg", "frequencia": "2x/dia", "duracao": "Conforme resposta"},
            {"nome": "Analgésicos", "dose": "0.1-0.3 mg/kg", "frequencia": "2-3x/dia", "duracao": "Conforme necessário"}
        ]
    }
    return medicamentos.get(diagnostico, [
        {"nome": "Tratamento sintomático", "dose": "Conforme sintomas", "frequencia": "Conforme necessário", "duracao": "Até melhora"}
    ])

def sugerir_exames(diagnostico):
    """Sugere exames complementares baseados no diagnóstico"""
    exames = {
        "Doença renal": [
            {"exame": "Urina completa", "prioridade": "Alta", "justificativa": "Avaliar função renal"},
            {"exame": "Proteinúria", "prioridade": "Alta", "justificativa": "Detectar perda de proteínas"},
            {"exame": "Ultrassom abdominal", "prioridade": "Alta", "justificativa": "Avaliar morfologia renal"},
            {"exame": "Radiografia abdominal", "prioridade": "Média", "justificativa": "Detectar cálculos ou obstruções"},
            {"exame": "Pressão arterial", "prioridade": "Alta", "justificativa": "Hipertensão renal comum"}
        ],
        "Infecção": [
            {"exame": "Hemograma completo", "prioridade": "Alta", "justificativa": "Avaliar resposta inflamatória"},
            {"exame": "Cultura e antibiograma", "prioridade": "Alta", "justificativa": "Identificar patógeno e sensibilidade"},
            {"exame": "PCR", "prioridade": "Média", "justificativa": "Detectar infecção sistêmica"},
            {"exame": "Raio-X torácico", "prioridade": "Média", "justificativa": "Pneumonia ou outras infecções"}
        ],
        "Tumor": [
            {"exame": "Biópsia", "prioridade": "Alta", "justificativa": "Confirmação histológica"},
            {"exame": "Radiografia torácica", "prioridade": "Alta", "justificativa": "Estadiamento - metástases"},
            {"exame": "Ultrassom abdominal", "prioridade": "Alta", "justificativa": "Estadiamento - metástases"},
            {"exame": "CT/MRI", "prioridade": "Média", "justificativa": "Avaliação detalhada do tumor"}
        ],
        "Trauma": [
            {"exame": "Radiografia", "prioridade": "Alta", "justificativa": "Detectar fraturas ou lesões"},
            {"exame": "Ultrassom FAST", "prioridade": "Alta", "justificativa": "Detectar hemorragia interna"},
            {"exame": "Hemograma", "prioridade": "Alta", "justificativa": "Avaliar perda sanguínea"}
        ]
    }
    return exames.get(diagnostico, [
        {"exame": "Avaliação clínica completa", "prioridade": "Alta", "justificativa": "Diagnóstico diferencial necessário"}
    ])

def sugerir_cirurgias(diagnostico):
    """Sugere cirurgias e protocolos baseados no diagnóstico"""
    cirurgias = {
        "Tumor": {
            "procedimentos": ["Tumor excision", "Mastectomia", "Amputação", "Biópsia cirúrgica"],
            "protocolo": "Protocolo oncológico: Pré-op: hemograma, função renal/hepática. Pós-op: quimioterapia adjuvante se indicado",
            "anestesia": {
                "premedicacao": "Midazolam 0.1-0.2 mg/kg + Morfina 0.1-0.3 mg/kg",
                "inducao": "Propofol 4-6 mg/kg IV",
                "manutencao": "Isoflurano 1-2% + Oxigênio",
                "analgesia_pos": "Morfina 0.1-0.3 mg/kg 4-6x/dia por 3-5 dias"
            }
        },
        "Trauma": {
            "procedimentos": ["Reparação de fraturas", "Laparotomia exploratória", "Toracotomia", "Sutura de lacerações"],
            "protocolo": "Protocolo de trauma: Estabilização → Cirurgia → Monitoramento intensivo",
            "anestesia": {
                "premedicacao": "Midazolam 0.05-0.1 mg/kg + Fentanil 2-5 mcg/kg",
                "inducao": "Etomidato 0.5-1 mg/kg IV (hemodinamicamente estável)",
                "manutencao": "Isoflurano 1-1.5% + Oxigênio",
                "analgesia_pos": "Fentanil 2-5 mcg/kg 2-3x/dia + Anti-inflamatório"
            }
        },
        "Obstrução": {
            "procedimentos": ["Enterotomia", "Gastrotomia", "Uretrostomia", "Cistotomia"],
            "protocolo": "Protocolo de obstrução: Descompressão → Cirurgia → Fluidoterapia agressiva",
            "anestesia": {
                "premedicacao": "Midazolam 0.1 mg/kg + Buprenorfina 0.01-0.02 mg/kg",
                "inducao": "Propofol 3-5 mg/kg IV",
                "manutencao": "Isoflurano 1-2% + Oxigênio",
                "analgesia_pos": "Buprenorfina 0.01-0.02 mg/kg 3x/dia por 5-7 dias"
            }
        },
        "Doença renal": {
            "procedimentos": ["Nefrectomia", "Ureterostomia", "Cistotomia", "Biópsia renal"],
            "protocolo": "Protocolo renal: Fluidoterapia cuidadosa → Monitoramento eletrólitos → Cirurgia se necessário",
            "anestesia": {
                "premedicacao": "Midazolam 0.05 mg/kg + Morfina 0.05-0.1 mg/kg",
                "inducao": "Propofol 2-4 mg/kg IV",
                "manutencao": "Isoflurano 0.8-1.5% + Oxigênio",
                "analgesia_pos": "Morfina 0.05-0.1 mg/kg 3-4x/dia + Monitoramento renal"
            }
        }
    }
    return cirurgias.get(diagnostico, {
        "procedimentos": ["Avaliação cirúrgica necessária"],
        "protocolo": "Protocolo padrão: Avaliação pré-anestésica → Cirurgia → Monitoramento pós-operatório",
        "anestesia": {
            "premedicacao": "Midazolam 0.1 mg/kg + Analgésico",
            "inducao": "Propofol 4-6 mg/kg IV",
            "manutencao": "Isoflurano 1-2% + Oxigênio",
            "analgesia_pos": "Analgesia multimodal conforme necessário"
        }
    })

# Sistema de abas
tab_names = ["🔍 Predição", "💬 Chat IA"]
tabs = st.tabs(tab_names)

# ABA 1: PREDIÇÃO
with tabs[0]:
    st.subheader("🔍 Predição de Diagnóstico")

# Função para carregar modelo
@st.cache_data
def carregar_modelo():
    """Carrega o modelo treinado"""
    try:
        # Lista de caminhos possíveis para o modelo (Streamlit Cloud compatível)
        possible_paths = [
            "VET/models/model_minimal.pkl",
            "VET/models/gb_model_optimized.pkl", 
            "VET/models/gb_optimized_model.pkl",
            "./VET/models/model_minimal.pkl",
            "./VET/models/gb_model_optimized.pkl",
            "./VET/models/gb_optimized_model.pkl",
            "models/model_minimal.pkl",
            "models/gb_model_optimized.pkl",
            "models/gb_optimized_model.pkl"
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
        
        # Criar array com todos os dados (39 features)
        dados_predicao = np.array([
            # Espécie (2 features)
            especie == "Canina", especie == "Felina",
            # Idade e sexo (2 features)
            idade_anos, sexo == "M",
            # Exames laboratoriais básicos (8 features)
            hemoglobina, hematocrito, leucocitos, 10.0,  # Plaquetas padrão
            glicose, ureia, creatinina, alt,
            # Mais exames laboratoriais (10 features)
            50.0,  # AST padrão
            100.0, 7.0, 3.5, 200.0, 100.0, 2.0,  # Outros exames padrão
            1.0, 1.5, 2.0,  # Mais 3 exames padrão
            # Sintomas clínicos (17 features)
        ] + sintomas_values + [0, 0, 0, 0, 0, 0, 0]).reshape(1, -1)
        
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
        
        # Sugestões baseadas no diagnóstico
        st.subheader("📋 Recomendações Baseadas no Diagnóstico")
        
        # Primeira linha: Exames e Doenças
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔬 Exames Complementares")
            exames = sugerir_exames(diagnostico_predito)
            for exame in exames:
                prioridade_color = "🔴" if exame['prioridade'] == "Alta" else "🟡"
                st.markdown(f"{prioridade_color} **{exame['exame']}** ({exame['prioridade']})")
                st.markdown(f"   *{exame['justificativa']}*")
                st.markdown("")
        
        with col2:
            st.subheader("🏥 Doenças Relacionadas")
            doencas = sugerir_doencas(diagnostico_predito)
            for doenca in doencas:
                st.markdown(f"• {doenca}")
        
        # Segunda linha: Medicamentos
        st.subheader("💊 Protocolo de Medicação")
        medicamentos = sugerir_medicamentos(diagnostico_predito)
        for med in medicamentos:
            with st.expander(f"💊 {med['nome']}"):
                st.markdown(f"**Dose:** {med['dose']}")
                st.markdown(f"**Frequência:** {med['frequencia']}")
                st.markdown(f"**Duração:** {med['duracao']}")
        
        # Terceira linha: Cirurgia e Anestesia
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("🔪 Protocolo Cirúrgico")
            cirurgias = sugerir_cirurgias(diagnostico_predito)
            st.markdown(f"**Procedimentos:**")
            for procedimento in cirurgias['procedimentos']:
                st.markdown(f"• {procedimento}")
            
            st.markdown(f"**Protocolo:** {cirurgias['protocolo']}")
        
        with col4:
            st.subheader("💉 Protocolo de Anestesia")
            anestesia = cirurgias['anestesia']
            st.markdown(f"**Pré-medicação:** {anestesia['premedicacao']}")
            st.markdown(f"**Indução:** {anestesia['inducao']}")
            st.markdown(f"**Manutenção:** {anestesia['manutencao']}")
            st.markdown(f"**Analgesia pós-op:** {anestesia['analgesia_pos']}")
        
        # Recomendações personalizadas do DeepSeek
        st.subheader("🤖 Recomendações Personalizadas (IA)")
        if st.button("🔍 Obter Recomendações Personalizadas"):
            with st.spinner("Consultando IA para recomendações personalizadas..."):
                sintomas_lista = [nome for nome, valor in [
                    ("Febre", febre), ("Apatia", apatia), ("Perda de peso", perda_peso),
                    ("Vômito", vomito), ("Diarreia", diarreia), ("Tosse", tosse),
                    ("Letargia", letargia), ("Feridas cutâneas", feridas_cutaneas),
                    ("Poliúria", poliuria), ("Polidipsia", polidipsia)
                ] if valor]
                
                dados_lab = f"Hb: {hemoglobina}, Ht: {hematocrito}, Leucócitos: {leucocitos}, Glicose: {glicose}, Ureia: {ureia}, Creatinina: {creatinina}, ALT: {alt}"
                
                recomendacoes = obter_recomendacoes_personalizadas(diagnostico_predito, sintomas_lista, dados_lab)
                st.markdown(recomendacoes)
        
        st.divider()
        
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

# ABA 2: CHAT IA
with tabs[1]:
    st.subheader("💬 Chat com IA Veterinária")
    st.info("🤖 Converse com nossa IA especializada em medicina veterinária. Faça perguntas sobre diagnósticos, tratamentos e casos clínicos.")
    
    # Interface do chat
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Container do chat
        chat_container = st.container()
        
        with chat_container:
            # Mostrar histórico do chat
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f'<div class="chat-message user-message"><strong>Você:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message assistant-message"><strong>IA Veterinária:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("**💡 Dicas:**")
        st.markdown("• Pergunte sobre sintomas")
        st.markdown("• Consulte diagnósticos")
        st.markdown("• Solicite tratamentos")
        st.markdown("• Peça exames")
        
        if st.button("🗑️ Limpar Chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Input do usuário
    user_input = st.text_area("Digite sua pergunta:", height=100, placeholder="Ex: Cão com vômito e diarreia há 2 dias, o que pode ser?")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("📤 Enviar", type="primary"):
            if user_input.strip():
                # Adicionar mensagem do usuário
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_input,
                    "timestamp": datetime.now()
                })
                
                # Gerar resposta da IA
                with st.spinner("🤖 IA pensando..."):
                    context = f"Histórico: {len(st.session_state.chat_history)} mensagens"
                    ai_response = call_deepseek_api(user_input, context)
                
                # Adicionar resposta da IA
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": ai_response,
                    "timestamp": datetime.now()
                })
                
                st.rerun()
    
    with col2:
        if st.button("🔄 Nova Aba"):
            new_tab_name = f"Chat {len(st.session_state.chat_tabs) + 1}"
            st.session_state.chat_tabs.append(new_tab_name)
            st.rerun()


# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>🐾 VetDiagnosisAI - Sistema Inteligente de Diagnóstico Veterinário</p>
    <p><small>Para dúvidas ou suporte, entre em contato com o administrador do sistema.</small></p>
</div>
""", unsafe_allow_html=True)
