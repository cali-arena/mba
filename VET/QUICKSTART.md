# 🚀 Quick Start Guide - VetDiagnosisAI

## 📦 Instalação Rápida

```bash
# 1. Navegue até a pasta do projeto
cd VET

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Execute o aplicativo
streamlit run app.py
```

O aplicativo abrirá automaticamente no navegador em `http://localhost:8501`

---

## 🎯 Primeiros Passos

### 1️⃣ Carregar Dados

**Opção A: Dataset de Exemplo (Recomendado)**
1. Vá para **📥 Upload de Dados**
2. Clique em **📂 Dataset de Exemplo**
3. Clique no botão **🔄 Carregar Dataset de Exemplo**

**Opção B: Seus Próprios Dados**
1. Vá para **📥 Upload de Dados**
2. Faça upload de arquivo CSV ou XLSX
3. Mapeie as colunas (automático ou manual)
4. Salve e carregue

### 2️⃣ Explorar os Dados

**📊 Visão Geral**
- Métricas principais
- Distribuições de espécies e diagnósticos
- Alertas de valores críticos

**🧪 Laboratório & Sintomas (EDA)**
- Análise de exames laboratoriais
- Distribuição de sintomas
- Correlações
- Detecção de outliers

### 3️⃣ Treinar um Modelo

1. Vá para **🤖 Treinar Modelo**
2. Selecione o algoritmo (Random Forest recomendado para início)
3. Configure os parâmetros (padrões são bons para começar)
4. Clique em **🎯 Treinar Modelo**
5. Aguarde o treinamento e analise os resultados
6. **Salve o modelo** para usar em predições

### 4️⃣ Fazer Predições

**Entrada Manual:**
1. Vá para **🔍 Predição**
2. Selecione **📝 Entrada Manual**
3. Preencha os dados do animal
4. Insira os exames laboratoriais
5. Marque os sintomas presentes
6. Clique em **🔍 Fazer Predição**

**Upload em Lote:**
1. Vá para **🔍 Predição**
2. Selecione **📤 Upload de Arquivo**
3. Faça upload de CSV/XLSX com múltiplos casos
4. Clique em **🔍 Fazer Predições em Lote**
5. Baixe os resultados

### 5️⃣ Analisar Insights

1. Vá para **🧠 Insights & Regras**
2. Visualize insights gerais do dataset
3. Analise diagnósticos específicos
4. Use o gerador de hipóteses para casos individuais

---

## 📝 Exemplo de Workflow Completo

```
1. Carregar Dataset de Exemplo
   ↓
2. Explorar em "Visão Geral" (entender os dados)
   ↓
3. Analisar em "Laboratório & Sintomas" (EDA detalhada)
   ↓
4. Treinar Modelo em "Treinar Modelo"
   - Algoritmo: Random Forest
   - Test size: 20%
   - Salvar modelo após treino
   ↓
5. Fazer Predições em "Predição"
   - Testar com entrada manual
   - Analisar explicabilidade
   ↓
6. Gerar Insights em "Insights & Regras"
   - Ver hipóteses diagnósticas
   - Consultar regras clínicas
```

---

## 🔧 Solução de Problemas

### Erro ao carregar SHAP
```bash
pip install shap==0.43.0
```

### Erro ao carregar LightGBM/XGBoost
```bash
pip install lightgbm xgboost
```

### Erro com OpenPyXL (arquivos Excel)
```bash
pip install openpyxl
```

### Port 8501 já em uso
```bash
streamlit run app.py --server.port 8502
```

---

## 📊 Estrutura de Dados Recomendada

### CSV Exemplo:
```csv
id,especie,raca,sexo,idade_anos,hemoglobina,creatinina,ureia,glicose,alt,febre,apatia,vomito,diagnostico
VET001,Canina,Labrador,M,5.5,14.2,1.1,35,95,45,0,0,0,Saudável
VET002,Felina,Persa,F,8.0,11.5,2.8,75,180,120,0,1,1,Doença Renal Crônica
```

### Colunas Mínimas Necessárias:
- `especie`: Canina, Felina, Equina
- Ao menos alguns exames: hemoglobina, creatinina, ureia, glicose, alt, ast, etc.
- `diagnostico`: Para treinar modelos

---

## 💡 Dicas e Boas Práticas

### Para Análise Exploratória:
- Use os filtros na barra lateral para análises específicas
- Compare valores com as faixas de referência por espécie
- Identifique outliers antes de treinar modelos

### Para Treinamento de Modelos:
- Comece com Random Forest (bom equilíbrio)
- Use Grid Search se tiver tempo (melhora performance)
- Sempre valide com dados de teste independentes
- Salve o modelo após treinar

### Para Predições:
- Verifique os alertas de valores críticos
- Analise a explicabilidade (features importantes)
- Considere Top 3 diagnósticos, não apenas o primeiro
- Leia as recomendações clínicas geradas

### Para Produção/Uso Clínico:
- ⚠️ **NUNCA** use como única fonte de diagnóstico
- Sempre consulte um médico veterinário
- Use como ferramenta de **apoio** à decisão
- Valide com dados locais antes de confiar

---

## 🔗 Links Úteis

- **Datasets Públicos:**
  - [Kaggle - Veterinary Disease Detection](https://www.kaggle.com/datasets/taruntiwarihp/veterinary-disease-detection)
  - [UCI - Horse Colic](https://archive.ics.uci.edu/dataset/46/horse+colic)
  - [Kaggle - Animal Blood Samples](https://www.kaggle.com/datasets/andrewmvd/animal-blood-samples)

- **Documentação Streamlit:** https://docs.streamlit.io
- **Scikit-learn:** https://scikit-learn.org
- **SHAP:** https://shap.readthedocs.io

---

## 📞 Suporte

Para dúvidas ou problemas:
1. Verifique o README.md principal
2. Consulte a documentação inline nas páginas
3. Revise os exemplos no código
4. Abra uma issue no repositório

---

## ⚠️ Avisos Legais

**IMPORTANTE:**
- Este é um sistema educacional e de pesquisa
- NÃO substitui avaliação clínica profissional
- Sempre consulte um médico veterinário licenciado
- Valide todos os resultados antes de uso clínico

---

**Desenvolvido para profissionais veterinários e pesquisadores** 🐾

