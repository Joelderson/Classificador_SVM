# Sistema de Classificação SVM - GVA/NAAT

## Descrição

Este é um sistema completo de classificação multiclasse usando Support Vector Machine (SVM) desenvolvido para análise de dados de condição de máquinas. O sistema inclui uma interface gráfica intuitiva e funcionalidades avançadas de machine learning.

## Características Principais

- **Interface Gráfica Intuitiva**: Interface Tkinter profissional com logos GVA/NAAT
- **Classificação Multiclasse**: Suporte para 2-5 classes de condições
- **Otimização Automática**: GridSearchCV para encontrar os melhores hiperparâmetros
- **Análise de Features**: Permutation Importance para identificar features mais relevantes
- **Visualizações**: Matriz de confusão e gráficos de importância das features
- **Processamento Robusto**: Suporte para diferentes formatos de arquivos CSV

## Estrutura do Projeto

```
teste_svm/
├── svm_classifier_app.py      # Aplicação principal
├── requirements.txt           # Dependências Python
├── gva.jpg                   # Logo GVA
├── naat.jpg                  # Logo NAAT
├── dados_testes_svm/         # Dados de teste organizados por condição
│   ├── Vetores_97_normal_0_condicao_sem_falha/
│   ├── Vetores_falha_B_014_286_0/
│   ├── Vetores_falha_IR_014_274_0/
│   └── Vetores_falha_OR_014_309_1/
└── resultados_svm/           # Resultados gerados pela aplicação
```

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/Joelderson/Teste_SVM.git
cd Teste_SVM
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Como Usar

1. **Execute a aplicação**:
```bash
python svm_classifier_app.py
```

2. **Configure os dados**:
   - Selecione o número de classes (2-5)
   - Escolha as pastas contendo os dados CSV para cada condição
   - Defina nomes descritivos para cada condição

3. **Processe e treine**:
   - Clique em "Carregar e Processar Dados"
   - Clique em "Treinar Modelo" (pode levar alguns minutos)
   - Clique em "Avaliar Modelo" para ver os resultados

4. **Visualize os resultados**:
   - Os resultados são salvos automaticamente na pasta `resultados_svm/`
   - Use "Abrir Pasta de Resultados" para acessar os arquivos

## Funcionalidades Técnicas

### Backend (SVMSystem)
- **Carregamento de Dados**: Processamento robusto de arquivos CSV
- **Pré-processamento**: Normalização automática com StandardScaler
- **Treinamento**: SVM com kernel RBF e otimização via GridSearchCV
- **Avaliação**: Métricas completas (acurácia, relatório de classificação, matriz de confusão)
- **Análise de Features**: Permutation Importance para identificar features mais importantes

### Frontend (SVM_GUI)
- **Interface Profissional**: Design moderno com logos institucionais
- **Flexibilidade**: Suporte para 2-5 classes de condições
- **Feedback em Tempo Real**: Log detalhado de todas as operações
- **Gestão de Resultados**: Organização automática dos arquivos de saída

## Dependências

- Python 3.7+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- tkinter (incluído no Python)
- PIL (Pillow)

## Estrutura dos Dados

Os dados devem estar organizados em pastas separadas por condição, contendo arquivos CSV com:
- Features numéricas (com ou sem cabeçalho)
- Um arquivo por segmento de dados
- Número consistente de features em todos os arquivos

## Resultados

O sistema gera automaticamente:
- **Relatório de Classificação**: Acurácia e métricas detalhadas
- **Matriz de Confusão**: Visualização da performance do modelo
- **Importância das Features**: Gráfico das features mais relevantes
- **Logs Detalhados**: Histórico completo do processamento

## Contribuição

Para contribuir com o projeto:
1. Faça um fork do repositório
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Abra um Pull Request

## Licença

Este projeto está sob a licença MIT.

## Contato

Desenvolvido por Joelderson para GVA/NAAT. 