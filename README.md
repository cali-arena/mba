# 📊 MBA - Projetos de Análise de Dados

Este repositório contém projetos de análise de dados e machine learning desenvolvidos durante o MBA.

## 📁 Projetos

### 1. 📈 Dashboard de Análise de Vendas (`testes/`)

Dashboard interativo completo para análise de vendas com:
- **KPIs principais** (Vendas totais, quantidade, ticket médio)
- **Análise por categoria** com gráficos interativos
- **Análise temporal** (evolução de vendas)
- **Matriz de correlações**
- **Top produtos**
- **Machine Learning**: 11 modelos com recomendação automática do melhor modelo

#### 🚀 Como Executar

```bash
cd testes
streamlit run mba1.py
```

**Funcionalidades de ML:**
- Linear Regression
- Ridge, Lasso, ElasticNet
- Decision Tree
- Random Forest
- Gradient Boosting
- AdaBoost
- K-Nearest Neighbors
- Support Vector Machine
- Neural Network

### 2. 🏥 Sistema Veterinário - VET (`VET/`)

Sistema completo de diagnóstico veterinário com IA e análise de dados clínicos.

#### 🚀 Como Executar

**Versão Simplificada (Recomendada para Deploy):**
```bash
cd VET
streamlit run app_simple.py
```

**Versão Completa:**
```bash
cd VET
streamlit run app.py
```

**Funcionalidades:**
- Análise de dados clínicos
- Predição de diagnósticos
- Insights e regras clínicas
- Upload de dados
- Treinamento de modelos
- Visualizações interativas
- **Dados sintéticos incluídos** (não precisa de arquivos externos)

## 📦 Instalação

### Pré-requisitos
- Python 3.8+
- pip

### Instalar Dependências

```bash
pip install -r requirements.txt
```

## 🌐 Deploy no Streamlit Cloud

1. Faça fork/clone deste repositório
2. Acesse [Streamlit Cloud](https://streamlit.io/cloud)
3. Conecte sua conta GitHub
4. Selecione este repositório
5. Escolha o arquivo principal:
   - Para Dashboard de Vendas: `testes/mba1.py`
   - Para Sistema VET: `VET/app.py`
6. Deploy!

## 📊 Dados

### Dashboard de Vendas
- Arquivo: `testes/sales_data.csv`
- 100+ produtos
- Múltiplas categorias (Electronics, Clothing, Grocery, Toys)
- Período: Janeiro a Abril 2024

### Sistema VET
- Múltiplos datasets na pasta `VET/data/`
- Dados clínicos veterinários reais
- Painéis laboratoriais completos

## 🛠️ Tecnologias Utilizadas

- **Streamlit**: Framework web para aplicações de dados
- **Pandas**: Manipulação de dados
- **Plotly**: Visualizações interativas
- **Scikit-learn**: Machine Learning
- **NumPy**: Computação numérica

## 📝 Licença

Este projeto é para fins educacionais - MBA.

## 👨‍💻 Autor

Lucas - MBA Student

---

⭐ Se este projeto foi útil, dê uma estrela!

