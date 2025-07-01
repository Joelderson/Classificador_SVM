# Sistema de Classificação SVM - GVA/NAAT

## Descrição

Este é um sistema completo de classificação multiclasse usando Support Vector Machine (SVM) desenvolvido para análise de dados de condição de máquinas. O sistema inclui uma interface gráfica intuitiva e funcionalidades avançadas de machine learning.

## Características Principais

- **Interface Gráfica Intuitiva**: Interface Tkinter profissional com logos GVA/NAAT
- **Classificação Multiclasse**: Suporte para 2-7 classes de condições
- **Otimização Automática**: GridSearchCV para encontrar os melhores hiperparâmetros
- **Análise de Features**: Permutation Importance para identificar features mais relevantes
- **Visualizações**: Matriz de confusão, curva de aprendizado e gráficos de importância das features
- **Processamento Robusto**: Suporte para diferentes formatos de arquivos CSV
- **Seleção de Features**: Interface para selecionar features específicas para treinamento

## Estrutura do Projeto

```
teste_svm/
├── classificador_svm_multiclasse.py  # Aplicação principal atualizada
├── svm_classifier_app.py             # Versão anterior da aplicação
├── requirements.txt                  # Dependências Python
├── gva.jpg                          # Logo GVA
├── naat.jpg                         # Logo NAAT
├── features_separadas_svm/          # Dados de exemplo organizados por condição
│   ├── Features_separadas_vetores_menores/
│   │   ├── 97_normal_0_vetores_features/
│   │   ├── B_007_118_0_vetores_features/
│   │   ├── B_021_222_0_vetores_features/
│   │   ├── IR_007_105_0_vetores_features/
│   │   ├── IR_021_209_0_vetores_features/
│   │   ├── OR_007_@3_144_0_vetores_features/
│   │   └── OR_021_@3_246_0_vetores_features/
│   └── Features_vetores_maiores/
│       ├── 97_0_normal/
│       ├── B_007_118_0_Vetores_de_Features_maior/
│       ├── B_021_222_0_Vetores_de_Features_maior/
│       ├── IR_007_105_0_Vetores_de_Features_maior/
│       ├── IR_021_209_0_Vetores_de_Features_maior/
│       ├── OR_007_@3_144_0_Vetores_de_Features_maior/
│       └── OR_021_@3_246_0_Vetores_de_Features_maior/
└── resultados_svm/                  # Resultados gerados pela aplicação
```

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/Joelderson/Classificador_SVM.git
cd Classificador_SVM
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Como Usar

1. **Execute a aplicação**:
```bash
python classificador_svm_multiclasse.py
```

2. **Configure os dados**:
   - Selecione o número de classes (2-7)
   - Escolha as pastas contendo os dados CSV para cada condição
   - Defina nomes descritivos para cada condição

3. **Selecione as features**:
   - Após carregar os dados, selecione as features desejadas na lista
   - Você pode selecionar múltiplas features para o treinamento

4. **Processe e treine**:
   - Clique em "Carregar e Processar Dados"
   - Clique em "Treinar Modelo" (pode levar alguns minutos)
   - Clique em "Avaliar Modelo" para ver os resultados

5. **Visualize os resultados**:
   - Os resultados são salvos automaticamente na pasta `resultados_svm/`
   - Use "Abrir Pasta de Resultados" para acessar os arquivos

## Funcionalidades Técnicas

### Backend (SVMSystem)
- **Carregamento de Dados**: Processamento robusto de arquivos CSV
- **Pré-processamento**: Normalização automática com seleção inteligente do melhor scaler
- **Treinamento**: SVM com kernel RBF e otimização via GridSearchCV
- **Avaliação**: Métricas completas (acurácia, relatório de classificação, matriz de confusão)
- **Análise de Features**: Permutation Importance para identificar features mais importantes
- **Curva de Aprendizado**: Análise do comportamento do modelo durante o treinamento
- **Distribuição de Features**: Análise da feature mais difícil de classificar

### Frontend (SVM_GUI)
- **Interface Profissional**: Design moderno com logos institucionais
- **Flexibilidade**: Suporte para 2-7 classes de condições
- **Seleção de Features**: Interface para escolher features específicas
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
- **Ranking das Features**: Lista ordenada das features por importância
- **Curva de Aprendizado**: Análise do comportamento do modelo
- **Distribuição da Feature Mais Difícil**: Análise da feature com maior sobreposição entre classes
- **Logs Detalhados**: Histórico completo do processamento

## Dados de Exemplo

O repositório inclui dados de exemplo organizados em 7 condições diferentes:
- **Normal**: Condição sem falha (97_normal_0)
- **Falha Bearing**: Diferentes tipos de falha no rolamento (B_007_118_0, B_021_222_0)
- **Falha Inner Race**: Falhas na pista interna (IR_007_105_0, IR_021_209_0)
- **Falha Outer Race**: Falhas na pista externa (OR_007_@3_144_0, OR_021_@3_246_0)

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