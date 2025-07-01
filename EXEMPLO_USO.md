# Guia de Exemplo de Uso - Sistema SVM GVA/NAAT

## 🚀 Início Rápido

### 1. Instalação Automática
```bash
# Clone o repositório
git clone https://github.com/Joelderson/Classificador_SVM.git
cd Classificador_SVM

# Execute o script de setup
python setup.py
```

### 2. Execução Manual
```bash
# Instale as dependências
pip install -r requirements.txt

# Execute a aplicação
python classificador_svm_multiclasse.py
```

## 📋 Exemplo Prático de Uso

### Passo 1: Configuração Inicial
1. **Abra a aplicação** - A interface gráfica será exibida com os logos GVA/NAAT
2. **Configure o número de classes** - Selecione 7 classes (para usar todos os dados de exemplo)
3. **Selecione as pastas de dados**:
   - Clique em "Selecionar..." para cada condição
   - Navegue até `features_separadas_svm/Features_separadas_vetores_menores/`
   - Selecione as seguintes pastas:
     - `97_normal_0_vetores_features` → Nome: "Normal"
     - `B_007_118_0_vetores_features` → Nome: "Falha Bearing 007"
     - `B_021_222_0_vetores_features` → Nome: "Falha Bearing 021"
     - `IR_007_105_0_vetores_features` → Nome: "Falha Inner Race 007"
     - `IR_021_209_0_vetores_features` → Nome: "Falha Inner Race 021"
     - `OR_007_@3_144_0_vetores_features` → Nome: "Falha Outer Race 007"
     - `OR_021_@3_246_0_vetores_features` → Nome: "Falha Outer Race 021"

### Passo 2: Carregamento e Processamento
1. **Clique em "Carregar e Processar Dados"**
   - O sistema detectará automaticamente o separador dos arquivos CSV
   - Processará todos os arquivos e extrairá as features numéricas
   - Normalizará os dados usando o melhor método disponível
   - Dividirá os dados em conjuntos de treino e teste

### Passo 3: Seleção de Features
1. **Após o carregamento**, a lista de features será preenchida
2. **Selecione as features desejadas**:
   - Clique nas features que deseja usar para treinamento
   - Você pode selecionar múltiplas features (Ctrl+clique)
   - Recomendação: Selecione as primeiras 10-20 features para começar

### Passo 4: Treinamento do Modelo
1. **Clique em "Treinar Modelo"**
   - O sistema iniciará a busca em grade (GridSearchCV)
   - Testará diferentes valores de C e gamma
   - Pode levar alguns minutos dependendo do hardware
   - Os melhores parâmetros serão selecionados automaticamente

### Passo 5: Avaliação e Resultados
1. **Clique em "Avaliar Modelo"**
   - O modelo será testado no conjunto de teste
   - Serão gerados relatórios e gráficos automaticamente
   - Os resultados serão salvos na pasta `resultados_svm/`

## 📊 Resultados Gerados

Após a avaliação, você encontrará na pasta `resultados_svm/`:

### Arquivos de Texto
- **`relatorio_classificacao.txt`** - Acurácia e métricas detalhadas
- **`ranking_features_utilizadas.txt`** - Lista das features por importância

### Gráficos
- **`matriz_confusao.png`** - Performance do modelo por classe
- **`importancia_features.png`** - Gráfico de importância das features
- **`ranking_features_utilizadas.png`** - Ranking visual das features
- **`curva_aprendizado.png`** - Comportamento do modelo durante treinamento
- **`distribuicao_feature_mais_dificil.png`** - Análise da feature mais problemática

## 🔧 Configurações Avançadas

### Usando Dados Próprios
1. **Organize seus dados** em pastas separadas por condição
2. **Formato dos arquivos**: CSV com features numéricas
3. **Estrutura recomendada**:
   ```
   meus_dados/
   ├── condicao_1/
   │   ├── arquivo1.csv
   │   ├── arquivo2.csv
   │   └── ...
   ├── condicao_2/
   │   ├── arquivo1.csv
   │   └── ...
   └── ...
   ```

### Otimização de Performance
- **Reduza o número de features** se o treinamento for lento
- **Use menos classes** para testes iniciais
- **Ajuste o número de folds** no GridSearchCV (código fonte)

## 🐛 Solução de Problemas

### Erro: "Nenhum arquivo .csv encontrado"
- Verifique se as pastas contêm arquivos CSV
- Confirme se os arquivos têm extensão `.csv`

### Erro: "Número de features diferente do esperado"
- Todos os arquivos devem ter o mesmo número de colunas numéricas
- Verifique se há arquivos com formatos diferentes

### Treinamento muito lento
- Reduza o número de features selecionadas
- Use menos classes para testes
- Considere usar dados de exemplo menores

### Erro de memória
- Reduza o número de arquivos por condição
- Selecione menos features
- Feche outras aplicações

## 📈 Interpretação dos Resultados

### Acurácia
- **> 90%**: Excelente performance
- **80-90%**: Boa performance
- **70-80%**: Performance aceitável
- **< 70%**: Pode precisar de ajustes

### Matriz de Confusão
- **Diagonal principal**: Classificações corretas
- **Outras células**: Erros de classificação
- **Análise**: Identifique quais classes são confundidas

### Importância das Features
- **Features no topo**: Mais importantes para classificação
- **Features na base**: Menos importantes
- **Uso**: Foque nas features mais importantes para otimização

## 🎯 Dicas de Uso

1. **Comece com dados de exemplo** para familiarizar-se com a interface
2. **Teste com poucas classes** antes de usar todas as 7
3. **Experimente diferentes seleções de features**
4. **Use o botão "Reiniciar"** para limpar resultados anteriores
5. **Verifique os logs** para entender o processamento
6. **Salve os resultados** importantes antes de reiniciar

## 📞 Suporte

Para dúvidas ou problemas:
- Verifique o README.md principal
- Consulte os logs da aplicação
- Verifique se todas as dependências estão instaladas
- Teste com os dados de exemplo fornecidos 