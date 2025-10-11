import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Dashboard de Vendas",
    page_icon="📊",
    layout="wide"
)

# Título principal
st.title("📊 Dashboard de Análise de Vendas")
st.markdown("---")

# Carregar dados
@st.cache_data
def load_data():
    df = pd.read_csv('sales_data.csv')
    df['Date_Sold'] = pd.to_datetime(df['Date_Sold'])
    df['Month'] = df['Date_Sold'].dt.month
    df['Month_Name'] = df['Date_Sold'].dt.strftime('%B')
    df['Day_of_Week'] = df['Date_Sold'].dt.day_name()
    return df

# Carregar os dados
try:
    df = load_data()
    
    # Criar abas
    tab1, tab2 = st.tabs(["📊 Dashboard de Vendas", "🤖 Machine Learning"])
    
    # ==================== ABA 1: DASHBOARD ====================
    with tab1:
        # Sidebar com filtros
        st.sidebar.header("🔍 Filtros")
        
        # Filtro de categoria
        categorias = ['Todas'] + sorted(df['Category'].unique().tolist())
        categoria_selecionada = st.sidebar.selectbox("Selecione a Categoria:", categorias)
        
        # Filtro de data
        data_inicial = st.sidebar.date_input("Data Inicial:", df['Date_Sold'].min())
        data_final = st.sidebar.date_input("Data Final:", df['Date_Sold'].max())
        
        # Aplicar filtros
        df_filtrado = df.copy()
        if categoria_selecionada != 'Todas':
            df_filtrado = df_filtrado[df_filtrado['Category'] == categoria_selecionada]
        
        df_filtrado = df_filtrado[
            (df_filtrado['Date_Sold'] >= pd.to_datetime(data_inicial)) &
            (df_filtrado['Date_Sold'] <= pd.to_datetime(data_final))
        ]
        
        # KPIs principais
        st.header("📈 Métricas Principais")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_vendas = df_filtrado['Total_Sales'].sum()
            st.metric("💰 Vendas Totais", f"${total_vendas:,.2f}")
        
        with col2:
            quantidade_total = df_filtrado['Quantity_Sold'].sum()
            st.metric("📦 Quantidade Vendida", f"{quantidade_total:,}")
        
        with col3:
            ticket_medio = df_filtrado['Total_Sales'].sum() / len(df_filtrado) if len(df_filtrado) > 0 else 0
            st.metric("🎫 Ticket Médio", f"${ticket_medio:,.2f}")
        
        with col4:
            produtos_unicos = df_filtrado['Product_ID'].nunique()
            st.metric("🏷️ Produtos Únicos", f"{produtos_unicos}")
        
        st.markdown("---")
        
        # Seção de análise por categoria
        st.header("📊 Análise por Categoria")
        col1, col2 = st.columns(2)
        
        with col1:
            # Vendas por categoria
            vendas_categoria = df_filtrado.groupby('Category')['Total_Sales'].sum().reset_index()
            vendas_categoria = vendas_categoria.sort_values('Total_Sales', ascending=False)
            
            fig_categoria = px.bar(
                vendas_categoria,
                x='Category',
                y='Total_Sales',
                title='Vendas Totais por Categoria',
                labels={'Total_Sales': 'Vendas ($)', 'Category': 'Categoria'},
                color='Total_Sales',
                color_continuous_scale='Blues'
            )
            fig_categoria.update_layout(showlegend=False)
            st.plotly_chart(fig_categoria, use_container_width=True)
        
        with col2:
            # Gráfico de pizza - distribuição de vendas
            fig_pie = px.pie(
                vendas_categoria,
                values='Total_Sales',
                names='Category',
                title='Distribuição de Vendas por Categoria',
                hole=0.4
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Quantidade vendida por categoria
        col1, col2 = st.columns(2)
        
        with col1:
            quantidade_categoria = df_filtrado.groupby('Category')['Quantity_Sold'].sum().reset_index()
            fig_quant = px.bar(
                quantidade_categoria,
                x='Category',
                y='Quantity_Sold',
                title='Quantidade Vendida por Categoria',
                labels={'Quantity_Sold': 'Quantidade', 'Category': 'Categoria'},
                color='Quantity_Sold',
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig_quant, use_container_width=True)
        
        with col2:
            # Preço médio por categoria
            preco_medio_cat = df_filtrado.groupby('Category')['Price'].mean().reset_index()
            fig_preco = px.bar(
                preco_medio_cat,
                x='Category',
                y='Price',
                title='Preço Médio por Categoria',
                labels={'Price': 'Preço Médio ($)', 'Category': 'Categoria'},
                color='Price',
                color_continuous_scale='Oranges'
            )
            st.plotly_chart(fig_preco, use_container_width=True)
        
        st.markdown("---")
        
        # Análise temporal
        st.header("📅 Análise Temporal")
        col1, col2 = st.columns(2)
        
        with col1:
            # Vendas ao longo do tempo
            vendas_tempo = df_filtrado.groupby('Date_Sold')['Total_Sales'].sum().reset_index()
            fig_tempo = px.line(
                vendas_tempo,
                x='Date_Sold',
                y='Total_Sales',
                title='Evolução das Vendas ao Longo do Tempo',
                labels={'Date_Sold': 'Data', 'Total_Sales': 'Vendas ($)'},
                markers=True
            )
            fig_tempo.update_traces(line_color='#1f77b4', line_width=2)
            st.plotly_chart(fig_tempo, use_container_width=True)
        
        with col2:
            # Vendas por mês
            vendas_mes = df_filtrado.groupby('Month_Name')['Total_Sales'].sum().reset_index()
            # Ordenar por mês
            meses_ordem = ['January', 'February', 'March', 'April', 'May', 'June', 
                           'July', 'August', 'September', 'October', 'November', 'December']
            vendas_mes['Month_Name'] = pd.Categorical(vendas_mes['Month_Name'], 
                                                       categories=meses_ordem, 
                                                       ordered=True)
            vendas_mes = vendas_mes.sort_values('Month_Name')
            
            fig_mes = px.bar(
                vendas_mes,
                x='Month_Name',
                y='Total_Sales',
                title='Vendas por Mês',
                labels={'Month_Name': 'Mês', 'Total_Sales': 'Vendas ($)'},
                color='Total_Sales',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_mes, use_container_width=True)
        
        st.markdown("---")
        
        # Matriz de correlação
        st.header("🔗 Análise de Correlações")
        
        # Calcular correlações
        colunas_numericas = ['Price', 'Quantity_Sold', 'Total_Sales']
        correlacao = df_filtrado[colunas_numericas].corr()
        
        # Heatmap de correlação
        fig_corr = go.Figure(data=go.Heatmap(
            z=correlacao.values,
            x=correlacao.columns,
            y=correlacao.columns,
            colorscale='RdBu',
            zmid=0,
            text=correlacao.values.round(3),
            texttemplate='%{text}',
            textfont={"size": 14},
            colorbar=dict(title="Correlação")
        ))
        
        fig_corr.update_layout(
            title='Matriz de Correlação entre Variáveis',
            xaxis_title='Variáveis',
            yaxis_title='Variáveis',
            height=500
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Análise de correlação
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Interpretação das Correlações:")
            st.write(f"""
            - **Preço vs Vendas Totais**: {correlacao.loc['Price', 'Total_Sales']:.3f}
            - **Quantidade vs Vendas Totais**: {correlacao.loc['Quantity_Sold', 'Total_Sales']:.3f}
            - **Preço vs Quantidade**: {correlacao.loc['Price', 'Quantity_Sold']:.3f}
            """)
            
            # Interpretação
            if correlacao.loc['Price', 'Total_Sales'] > 0.7:
                st.success("✅ Forte correlação positiva entre Preço e Vendas Totais!")
            elif correlacao.loc['Price', 'Total_Sales'] > 0.3:
                st.info("ℹ️ Correlação moderada entre Preço e Vendas Totais")
            else:
                st.warning("⚠️ Correlação fraca entre Preço e Vendas Totais")
        
        with col2:
            # Scatter plot - Preço vs Total de Vendas
            fig_scatter = px.scatter(
                df_filtrado,
                x='Price',
                y='Total_Sales',
                color='Category',
                size='Quantity_Sold',
                title='Relação entre Preço e Vendas Totais',
                labels={'Price': 'Preço ($)', 'Total_Sales': 'Vendas Totais ($)'},
                hover_data=['Product_Name']
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.markdown("---")
        
        # Top produtos
        st.header("🏆 Top Produtos")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top 10 Produtos por Vendas")
            top_produtos = df_filtrado.groupby('Product_Name').agg({
                'Total_Sales': 'sum',
                'Quantity_Sold': 'sum'
            }).sort_values('Total_Sales', ascending=False).head(10).reset_index()
            
            fig_top = px.bar(
                top_produtos,
                x='Total_Sales',
                y='Product_Name',
                orientation='h',
                title='Top 10 Produtos mais Vendidos',
                labels={'Total_Sales': 'Vendas ($)', 'Product_Name': 'Produto'},
                color='Total_Sales',
                color_continuous_scale='Sunset'
            )
            st.plotly_chart(fig_top, use_container_width=True)
        
        with col2:
            st.subheader("Top 10 Produtos por Quantidade")
            top_quantidade = df_filtrado.groupby('Product_Name')['Quantity_Sold'].sum().sort_values(ascending=False).head(10).reset_index()
            
            fig_top_quant = px.bar(
                top_quantidade,
                x='Quantity_Sold',
                y='Product_Name',
                orientation='h',
                title='Top 10 Produtos por Quantidade',
                labels={'Quantity_Sold': 'Quantidade', 'Product_Name': 'Produto'},
                color='Quantity_Sold',
                color_continuous_scale='Teal'
            )
            st.plotly_chart(fig_top_quant, use_container_width=True)
        
        st.markdown("---")
        
        # Tabela de dados
        st.header("📋 Visualização dos Dados")
        
        # Estatísticas descritivas
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Estatísticas Descritivas")
            st.dataframe(df_filtrado[colunas_numericas].describe().round(2), use_container_width=True)
        
        with col2:
            st.subheader("📈 Resumo por Categoria")
            resumo_categoria = df_filtrado.groupby('Category').agg({
                'Total_Sales': ['sum', 'mean'],
                'Quantity_Sold': ['sum', 'mean'],
                'Price': 'mean'
            }).round(2)
            resumo_categoria.columns = ['Total Vendas', 'Média Vendas', 'Total Qtd', 'Média Qtd', 'Preço Médio']
            st.dataframe(resumo_categoria, use_container_width=True)
        
        # Mostrar dados brutos
        if st.checkbox("🔍 Mostrar dados brutos"):
            st.subheader("Dados Completos")
            st.dataframe(df_filtrado, use_container_width=True)
            
            # Botão de download
            csv = df_filtrado.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Baixar dados filtrados como CSV",
                data=csv,
                file_name='sales_data_filtrado.csv',
                mime='text/csv',
            )
        
        # Insights automáticos
        st.markdown("---")
        st.header("💡 Insights Automáticos")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            categoria_mais_vendida = vendas_categoria.iloc[0]
            st.info(f"""
            **🏆 Categoria Campeã:**  
            {categoria_mais_vendida['Category']}  
            Vendas: ${categoria_mais_vendida['Total_Sales']:,.2f}
            """)
        
        with col2:
            produto_mais_vendido = df_filtrado.groupby('Product_Name')['Total_Sales'].sum().idxmax()
            valor_mais_vendido = df_filtrado.groupby('Product_Name')['Total_Sales'].sum().max()
            st.success(f"""
            **⭐ Produto Destaque:**  
            {produto_mais_vendido}  
            Vendas: ${valor_mais_vendido:,.2f}
            """)
        
        with col3:
            dia_melhor = df_filtrado.groupby('Date_Sold')['Total_Sales'].sum().idxmax()
            vendas_melhor_dia = df_filtrado.groupby('Date_Sold')['Total_Sales'].sum().max()
            st.warning(f"""
            **📅 Melhor Dia de Vendas:**  
            {dia_melhor.strftime('%d/%m/%Y')}  
            Vendas: ${vendas_melhor_dia:,.2f}
            """)
    
    # ==================== ABA 2: MACHINE LEARNING ====================
    with tab2:
        st.header("🤖 Modelos de Machine Learning")
        st.write("Escolha modelos para prever **Vendas Totais** baseado em Preço e Quantidade")
        
        # Preparar dados para ML
        df_ml = df.copy()
        
        # Encode da categoria
        le = LabelEncoder()
        df_ml['Category_Encoded'] = le.fit_transform(df_ml['Category'])
        
        # Features e target
        features = ['Price', 'Quantity_Sold', 'Category_Encoded']
        target = 'Total_Sales'
        
        X = df_ml[features]
        y = df_ml[target]
        
        # Dividir dados
        test_size = st.sidebar.slider("Tamanho do conjunto de teste (%)", 10, 40, 20) / 100
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Escalar dados
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Dicionário de modelos
        modelos = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(),
            'Lasso Regression': Lasso(),
            'ElasticNet': ElasticNet(),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'AdaBoost': AdaBoostRegressor(random_state=42),
            'K-Nearest Neighbors': KNeighborsRegressor(),
            'Support Vector Machine': SVR(),
            'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        }
        
        st.markdown("---")
        
        # Botão para treinar todos os modelos
        if st.button("🚀 Analisar Todos os Modelos (Recomendação Automática)", type="primary"):
            with st.spinner("🔄 Treinando e avaliando todos os modelos..."):
                resultados = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, (nome, modelo) in enumerate(modelos.items()):
                    status_text.text(f"Treinando: {nome}...")
                    
                    try:
                        # Decidir se usa dados escalados ou não
                        if nome in ['Support Vector Machine', 'Neural Network', 'K-Nearest Neighbors']:
                            X_train_use = X_train_scaled
                            X_test_use = X_test_scaled
                        else:
                            X_train_use = X_train
                            X_test_use = X_test
                        
                        # Treinar
                        modelo.fit(X_train_use, y_train)
                        
                        # Predições
                        y_pred_train = modelo.predict(X_train_use)
                        y_pred_test = modelo.predict(X_test_use)
                        
                        # Métricas
                        r2_train = r2_score(y_train, y_pred_train)
                        r2_test = r2_score(y_test, y_pred_test)
                        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
                        mae_test = mean_absolute_error(y_test, y_pred_test)
                        
                        # Cross-validation
                        cv_scores = cross_val_score(modelo, X_train_use, y_train, cv=5, scoring='r2')
                        cv_mean = cv_scores.mean()
                        
                        resultados.append({
                            'Modelo': nome,
                            'R² Treino': r2_train,
                            'R² Teste': r2_test,
                            'RMSE': rmse_test,
                            'MAE': mae_test,
                            'CV Score': cv_mean,
                            'Overfitting': abs(r2_train - r2_test)
                        })
                    except Exception as e:
                        st.warning(f"⚠️ Erro ao treinar {nome}: {str(e)}")
                    
                    progress_bar.progress((idx + 1) / len(modelos))
                
                status_text.text("✅ Análise completa!")
                progress_bar.empty()
                
                # Criar DataFrame com resultados
                df_resultados = pd.DataFrame(resultados)
                df_resultados = df_resultados.sort_values('R² Teste', ascending=False)
                
                # Identificar melhor modelo
                melhor_modelo_nome = df_resultados.iloc[0]['Modelo']
                melhor_r2 = df_resultados.iloc[0]['R² Teste']
                
                st.markdown("---")
                
                # Exibir recomendação
                st.success(f"""
                ### 🏆 RECOMENDAÇÃO AUTOMÁTICA
                
                **Melhor Modelo Detectado:** `{melhor_modelo_nome}`  
                **R² Score:** {melhor_r2:.4f}  
                **Acurácia:** {melhor_r2 * 100:.2f}%
                
                Este modelo apresentou a melhor performance para prever as vendas!
                """)
                
                st.markdown("---")
                
                # Tabela de comparação
                st.subheader("📊 Comparação de Todos os Modelos")
                
                # Formatar DataFrame
                df_display = df_resultados.copy()
                df_display['R² Treino'] = df_display['R² Treino'].apply(lambda x: f"{x:.4f}")
                df_display['R² Teste'] = df_display['R² Teste'].apply(lambda x: f"{x:.4f}")
                df_display['RMSE'] = df_display['RMSE'].apply(lambda x: f"{x:.2f}")
                df_display['MAE'] = df_display['MAE'].apply(lambda x: f"{x:.2f}")
                df_display['CV Score'] = df_display['CV Score'].apply(lambda x: f"{x:.4f}")
                df_display['Overfitting'] = df_display['Overfitting'].apply(lambda x: f"{x:.4f}")
                
                st.dataframe(df_display, use_container_width=True)
                
                # Visualizações
                col1, col2 = st.columns(2)
                
                with col1:
                    # Gráfico de R² Score
                    fig_r2 = px.bar(
                        df_resultados,
                        x='R² Teste',
                        y='Modelo',
                        orientation='h',
                        title='Comparação de R² Score (Teste)',
                        labels={'R² Teste': 'R² Score', 'Modelo': 'Modelo'},
                        color='R² Teste',
                        color_continuous_scale='Viridis',
                        text='R² Teste'
                    )
                    fig_r2.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                    fig_r2.update_layout(height=600)
                    st.plotly_chart(fig_r2, use_container_width=True)
                
                with col2:
                    # Gráfico de RMSE
                    fig_rmse = px.bar(
                        df_resultados,
                        x='RMSE',
                        y='Modelo',
                        orientation='h',
                        title='Comparação de RMSE (menor é melhor)',
                        labels={'RMSE': 'RMSE', 'Modelo': 'Modelo'},
                        color='RMSE',
                        color_continuous_scale='Reds_r',
                        text='RMSE'
                    )
                    fig_rmse.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                    fig_rmse.update_layout(height=600)
                    st.plotly_chart(fig_rmse, use_container_width=True)
                
                # Análise de Overfitting
                st.markdown("---")
                st.subheader("⚠️ Análise de Overfitting")
                
                fig_over = px.bar(
                    df_resultados,
                    x='Overfitting',
                    y='Modelo',
                    orientation='h',
                    title='Diferença entre R² Treino e Teste (menor é melhor)',
                    labels={'Overfitting': 'Diferença', 'Modelo': 'Modelo'},
                    color='Overfitting',
                    color_continuous_scale='Oranges',
                    text='Overfitting'
                )
                fig_over.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                st.plotly_chart(fig_over, use_container_width=True)
                
                # Treinar melhor modelo para predições
                st.markdown("---")
                st.subheader(f"🎯 Predições com o Melhor Modelo: {melhor_modelo_nome}")
                
                melhor_modelo = modelos[melhor_modelo_nome]
                
                if melhor_modelo_nome in ['Support Vector Machine', 'Neural Network', 'K-Nearest Neighbors']:
                    melhor_modelo.fit(X_train_scaled, y_train)
                    y_pred = melhor_modelo.predict(X_test_scaled)
                else:
                    melhor_modelo.fit(X_train, y_train)
                    y_pred = melhor_modelo.predict(X_test)
                
                # Gráfico de predições vs real
                df_pred = pd.DataFrame({
                    'Real': y_test.values,
                    'Predito': y_pred
                })
                
                fig_pred = px.scatter(
                    df_pred,
                    x='Real',
                    y='Predito',
                    title=f'Valores Reais vs Preditos - {melhor_modelo_nome}',
                    labels={'Real': 'Vendas Reais ($)', 'Predito': 'Vendas Preditas ($)'},
                    trendline='ols'
                )
                
                # Linha de referência perfeita
                max_val = max(df_pred['Real'].max(), df_pred['Predito'].max())
                fig_pred.add_trace(
                    go.Scatter(
                        x=[0, max_val],
                        y=[0, max_val],
                        mode='lines',
                        name='Predição Perfeita',
                        line=dict(color='red', dash='dash')
                    )
                )
                
                st.plotly_chart(fig_pred, use_container_width=True)
                
                # Importância das features (se disponível)
                if hasattr(melhor_modelo, 'feature_importances_'):
                    st.markdown("---")
                    st.subheader("📊 Importância das Variáveis")
                    
                    importancias = pd.DataFrame({
                        'Feature': features,
                        'Importância': melhor_modelo.feature_importances_
                    }).sort_values('Importância', ascending=False)
                    
                    # Traduzir nomes
                    importancias['Feature'] = importancias['Feature'].replace({
                        'Price': 'Preço',
                        'Quantity_Sold': 'Quantidade',
                        'Category_Encoded': 'Categoria'
                    })
                    
                    fig_imp = px.bar(
                        importancias,
                        x='Importância',
                        y='Feature',
                        orientation='h',
                        title='Importância das Variáveis na Predição',
                        color='Importância',
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)
        
        st.markdown("---")
        
        # Seção de teste individual
        st.subheader("🧪 Testar Modelo Individual")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            modelo_selecionado = st.selectbox(
                "Escolha um modelo:",
                list(modelos.keys())
            )
            
            if st.button("Treinar Modelo Selecionado"):
                with st.spinner(f"Treinando {modelo_selecionado}..."):
                    modelo = modelos[modelo_selecionado]
                    
                    if modelo_selecionado in ['Support Vector Machine', 'Neural Network', 'K-Nearest Neighbors']:
                        X_train_use = X_train_scaled
                        X_test_use = X_test_scaled
                    else:
                        X_train_use = X_train
                        X_test_use = X_test
                    
                    modelo.fit(X_train_use, y_train)
                    y_pred = modelo.predict(X_test_use)
                    
                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = mean_absolute_error(y_test, y_pred)
                    
                    st.success("✅ Modelo treinado com sucesso!")
                    
                    st.metric("R² Score", f"{r2:.4f}")
                    st.metric("RMSE", f"${rmse:.2f}")
                    st.metric("MAE", f"${mae:.2f}")
        
        with col2:
            st.info("""
            **Dicas sobre os modelos:**
            
            - **Random Forest**: Excelente para dados não-lineares, robusto
            - **Gradient Boosting**: Alta precisão, pode demorar mais
            - **Linear Regression**: Simples e rápido, bom para relações lineares
            - **Neural Network**: Poderoso para padrões complexos
            - **Decision Tree**: Fácil interpretação, pode ter overfitting
            
            **Métricas:**
            - **R²**: Quanto da variação é explicada (0-1, maior melhor)
            - **RMSE**: Erro médio em unidades originais (menor melhor)
            - **MAE**: Erro absoluto médio (menor melhor)
            """)

except FileNotFoundError:
    st.error("❌ Arquivo 'sales_data.csv' não encontrado! Certifique-se de que o arquivo está no mesmo diretório.")
except Exception as e:
    st.error(f"❌ Erro ao carregar os dados: {str(e)}")
