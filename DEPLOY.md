# 🚀 Guia de Deploy no Streamlit Cloud

## 📋 Pré-requisitos

✅ Repositório no GitHub: https://github.com/cali-arena/mba  
✅ Conta no [Streamlit Cloud](https://streamlit.io/cloud) (gratuita)

---

## 🌐 Deploy - Dashboard de Vendas

### Passo 1: Acesse Streamlit Cloud
1. Vá para https://share.streamlit.io/
2. Faça login com sua conta GitHub

### Passo 2: Novo App
1. Clique em "**New app**"
2. Selecione:
   - **Repository**: `cali-arena/mba`
   - **Branch**: `main`
   - **Main file path**: `testes/mba1.py`
3. Clique em "**Deploy!**"

### Passo 3: Aguarde
- O deploy leva cerca de 2-3 minutos
- O Streamlit instalará automaticamente as dependências do `requirements.txt`

### 🎯 URL Final
Seu app estará disponível em:
```
https://[your-app-name].streamlit.app
```

---

## 🏥 Deploy - Sistema Veterinário (VET)

### Configure um segundo app seguindo os mesmos passos:
1. **New app** → **Deploy another app**
2. Selecione:
   - **Repository**: `cali-arena/mba`
   - **Branch**: `main`
   - **Main file path**: `VET/app.py`
3. Clique em "**Deploy!**"

**Nota**: Como o VET tem seu próprio `requirements.txt` em `VET/requirements.txt`, você pode precisar:
- Copiar as dependências extras para o `requirements.txt` raiz, OU
- Configurar o Python dependencies file path para `VET/requirements.txt`

---

## 🔧 Solução de Problemas

### Erro: "ModuleNotFoundError"
**Solução**: Adicione a biblioteca faltante no `requirements.txt`

### Erro: "FileNotFoundError" para CSV
**Solução**: Certifique-se de que o caminho está correto:
- Dashboard: `sales_data.csv` (está na pasta `testes/`)
- VET: Os dados estão em `data/`

### App reiniciando/lento
**Solução**: 
- Use `@st.cache_data` para cache (já implementado)
- O plano gratuito tem recursos limitados

### Erro de memória
**Solução**: Remova modelos pesados ou otimize o código

---

## 📊 Apps Disponíveis

### 1. 📈 Dashboard de Vendas
- **Path**: `testes/mba1.py`
- **Funcionalidades**:
  - Análise de vendas completa
  - 11 modelos de Machine Learning
  - Recomendação automática do melhor modelo
  - Gráficos interativos
  - Correlações e insights

### 2. 🏥 Sistema Veterinário
- **Path**: `VET/app.py`
- **Funcionalidades**:
  - Diagnóstico veterinário com IA
  - Análise de dados clínicos
  - Upload de dados
  - Treinamento de modelos
  - Insights médicos

---

## 🔄 Atualizar o App

Após fazer mudanças no código:

```bash
git add .
git commit -m "Descrição das mudanças"
git push origin main
```

O Streamlit Cloud detectará automaticamente e fará redeploy!

---

## 🌟 Configurações Adicionais (Opcional)

### Secrets (Variáveis de Ambiente)
Se precisar de API keys ou senhas:
1. No dashboard do Streamlit Cloud
2. App settings → Secrets
3. Adicione no formato TOML:
```toml
[passwords]
admin = "sua-senha"

[api_keys]
openai = "sua-api-key"
```

### Domínio Customizado
- Disponível em planos pagos
- Configure em App settings → General

---

## 📱 Compartilhar

Após o deploy, compartilhe a URL:
- `https://seu-app.streamlit.app`
- O app é público por padrão
- Para apps privados, use plano pago

---

## 💡 Dicas

✅ Use `@st.cache_data` para otimizar performance  
✅ Mantenha o `requirements.txt` atualizado  
✅ Teste localmente antes de fazer push  
✅ Use branches para testar features  
✅ Monitore logs no Streamlit Cloud para debug  

---

## 🆘 Suporte

- **Documentação Streamlit Cloud**: https://docs.streamlit.io/streamlit-community-cloud
- **Fórum da Comunidade**: https://discuss.streamlit.io/
- **GitHub Issues**: Problemas específicos do projeto

---

**Pronto para deploy! 🚀**

Link do repositório: https://github.com/cali-arena/mba

