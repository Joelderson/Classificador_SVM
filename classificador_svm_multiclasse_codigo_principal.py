import os
import warnings
import shutil
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.inspection import permutation_importance
import csv

warnings.filterwarnings('ignore')

# --- Backend: Lógica do Sistema SVM ---
class SVMSystem:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.X_train, self.X_test = None, None
        self.y_train, self.y_test = None, None
        self.feature_names = None
        self.class_names = []
        self.results_path = "resultados_svm"
        os.makedirs(self.results_path, exist_ok=True)

    def clear_results_folder(self):
        """Apaga todos os arquivos e pastas dentro da pasta de resultados."""
        for filename in os.listdir(self.results_path):
            file_path = os.path.join(self.results_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Falha ao deletar {file_path}. Razão: {e}")

    def load_data_from_folders(self, folder_paths, class_names):
        """Carrega dados considerando cada arquivo CSV como um segmento único (linha de features), com detecção automática de separador e robustez a cabeçalhos/linhas vazias."""
        all_features = []
        all_labels = []
        arquivos_invalidos = []
        arquivos_validos_por_pasta = []
        feature_names_from_header = None
        expected_num_features = None

        if not folder_paths or len(folder_paths) != len(class_names):
            raise ValueError("O número de pastas e nomes de classes não corresponde ou está vazio.")
        self.class_names = class_names

        for i, folder_path in enumerate(folder_paths):
            if not folder_path or not os.path.isdir(folder_path):
                raise FileNotFoundError(f"Pasta para a Condição {i+1} não encontrada ou inválida: {folder_path}")

            files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
            if not files:
                raise ValueError(f"Nenhum arquivo .csv encontrado na pasta para a Condição {i+1}: {folder_path}")

            validos_pasta = 0
            for file in files:
                file_path = os.path.join(folder_path, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        sample = f.read(2048)
                        sniffer = csv.Sniffer()
                        try:
                            dialect = sniffer.sniff(sample)
                            sep = dialect.delimiter
                        except Exception:
                            sep = ','
                    df = pd.read_csv(file_path, sep=sep, header=0, skip_blank_lines=True)
                    if df.shape[0] == 0:
                        df = pd.read_csv(file_path, sep=sep, header=None, skip_blank_lines=True)
                    linha_valida = None
                    for idx, row in df.iterrows():
                        if not pd.isnull(row).all():
                            linha_valida = row
                            break
                    if linha_valida is None:
                        arquivos_invalidos.append((file, 'Arquivo vazio ou sem linhas de dados.'))
                        continue
                    # Seleciona apenas as colunas numéricas
                    valores_numericos = []
                    nomes_numericos = []
                    for col, val in zip(linha_valida.index, linha_valida.values):
                        try:
                            v = float(val)
                            if not pd.isnull(v):
                                valores_numericos.append(v)
                                nomes_numericos.append(str(col))
                        except Exception:
                            continue
                    if not valores_numericos:
                        arquivos_invalidos.append((file, f'Nenhuma coluna numérica detectada na linha: {linha_valida.values}'))
                        continue
                    if expected_num_features is None:
                        expected_num_features = len(valores_numericos)
                        if feature_names_from_header is None:
                            feature_names_from_header = nomes_numericos
                    if len(valores_numericos) != expected_num_features:
                        arquivos_invalidos.append((file, f'Número de features numéricas ({len(valores_numericos)}) diferente do esperado ({expected_num_features}).'))
                        continue
                    all_features.append(valores_numericos)
                    all_labels.append(i)
                    validos_pasta += 1
                except Exception as e:
                    arquivos_invalidos.append((file, f'Erro crítico: {e}'))
            arquivos_validos_por_pasta.append((class_names[i], validos_pasta, len(files)))

        if not all_features:
            msg = 'Nenhum dado válido foi carregado.\nResumo dos problemas encontrados:\n'
            for pasta, validos, total in arquivos_validos_por_pasta:
                msg += f'- {pasta}: {validos} de {total} arquivos válidos\n'
            if arquivos_invalidos:
                msg += '\nArquivos rejeitados e motivo:\n'
                for arq, motivo in arquivos_invalidos:
                    msg += f'  {arq}: {motivo}\n'
            raise ValueError(msg)
        X = np.array(all_features)
        y = np.array(all_labels)
        if feature_names_from_header and len(feature_names_from_header) == X.shape[1]:
            self.feature_names = feature_names_from_header
        else:
            self.feature_names = [f"Feature_{j+1}" for j in range(X.shape[1])]
        return X, y

    def prepare_data(self, X, y, test_size=0.2, random_state=42):
        """Divide os dados em treino/teste e normaliza, evitando data leakage. Escolhe automaticamente a melhor normalização."""
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
        
        # Primeiro divide os dados em treino/teste
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Testa diferentes normalizações usando apenas os dados de TREINO
        scalers = {
            'StandardScaler': StandardScaler(),
            'MinMaxScaler': MinMaxScaler(),
            'RobustScaler': RobustScaler()
        }
        
        melhor_scaler = None
        melhor_nome = None
        melhor_score = float('inf')
        
        # Avalia cada scaler usando apenas dados de TREINO (evita data leakage)
        for nome, scaler in scalers.items():
            # Cria uma cópia do scaler para não afetar o original
            scaler_copy = type(scaler)()
            
            # Aplica fit_transform apenas nos dados de TREINO
            X_train_scaled = scaler_copy.fit_transform(self.X_train)
            
            # Critérios de avaliação baseados nas propriedades esperadas de cada normalização
            if nome == 'StandardScaler':
                # Ideal: média próxima de 0, desvio padrão próximo de 1
                score = abs(X_train_scaled.mean()) + abs(X_train_scaled.std() - 1)
            elif nome == 'MinMaxScaler':
                # Ideal: mínimo próximo de 0, máximo próximo de 1
                score = abs(X_train_scaled.min()) + abs(X_train_scaled.max() - 1)
            elif nome == 'RobustScaler':
                # Ideal: mediana próxima de 0, IQR próximo de 1
                score = abs(np.median(X_train_scaled)) + abs(np.percentile(X_train_scaled, 75) - np.percentile(X_train_scaled, 25) - 1)
            
            if score < melhor_score:
                melhor_score = score
                melhor_scaler = type(scaler)()  # Cria uma nova instância
                melhor_nome = nome
        
        # Aplica o melhor scaler: FIT apenas nos dados de TREINO, TRANSFORM nos dados de TESTE
        # ⚠️ IMPORTANTE: Isso evita data leakage - os parâmetros de normalização são aprendidos
        # apenas dos dados de treino e aplicados aos dados de teste
        self.scaler = melhor_scaler
        self.X_train = self.scaler.fit_transform(self.X_train)  # FIT + TRANSFORM (treino)
        self.X_test = self.scaler.transform(self.X_test)        # Apenas TRANSFORM (teste)
        self.normalization_used = melhor_nome

    def train_model(self):
        """Treina o modelo SVM com Kernel RBF usando GridSearchCV para encontrar os melhores hiperparâmetros."""
        # Grade de busca expandida para uma otimização mais fina
        param_grid = {
            'C': [0.1, 1, 10, 100, 1000],
            'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 'scale']
        }

        # Usando n_jobs=-1 para usar todos os processadores disponíveis
        grid_search = GridSearchCV(
            SVC(kernel='rbf', random_state=42, probability=True),
            param_grid,
            cv=5, # 5-fold cross-validation
            refit=True, # Refit o melhor modelo nos dados de treino completos
            verbose=2,
            n_jobs=-1 
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        self.model = grid_search.best_estimator_
        
        train_accuracy = self.model.score(self.X_train, self.y_train)
        
        result = permutation_importance(
            self.model, self.X_test, self.y_test, n_repeats=10, random_state=42, n_jobs=-1
        )
        
        sorted_idx = result.importances_mean.argsort()

        plt.figure(figsize=(12, 8))
        plt.barh(range(self.X_test.shape[1]), result.importances_mean[sorted_idx])
        plt.yticks(range(self.X_test.shape[1]), np.array(self.feature_names)[sorted_idx])
        plt.xlabel("Redução na Acurácia (Importância por Permutação)")
        plt.title("Importância das Features para o Modelo")
        plt.tight_layout()
        
        plot_path = os.path.join(self.results_path, "importancia_features.png")
        plt.savefig(plot_path)
        plt.close()
        
        return train_accuracy, grid_search.best_params_

    def generate_feature_ranking(self):
        """Gera ranking das features baseado na importância por permutação do modelo treinado."""
        if self.model is None or self.X_test is None:
            raise ValueError("O modelo precisa ser treinado primeiro.")
            
        print(">>> Gerando ranking das features utilizadas no modelo...")
        
        # Calcula a importância por permutação no conjunto de teste
        result = permutation_importance(
            self.model, self.X_test, self.y_test, n_repeats=10, random_state=42, n_jobs=-1
        )
        
        # Cria um DataFrame com as importâncias
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': result.importances_mean
        })
        
        # Ordena por importância (decrescente)
        feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
        
        print("=== RANKING DE IMPORTÂNCIA DAS FEATURES UTILIZADAS ===")
        for i, (_, row) in enumerate(feature_importance_df.iterrows(), 1):
            print(f"{i}. {row['feature']}: {row['importance']:.4f}")
        
        # Salva o ranking completo
        ranking_path = os.path.join(self.results_path, "ranking_features_utilizadas.txt")
        with open(ranking_path, "w", encoding="utf-8") as f:
            f.write("RANKING DE IMPORTÂNCIA DAS FEATURES UTILIZADAS NO MODELO\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total de features utilizadas: {len(self.feature_names)}\n")
            f.write(f"Features: {', '.join(self.feature_names)}\n\n")
            f.write("Ranking por Importância:\n")
            f.write("-" * 40 + "\n")
            for i, (_, row) in enumerate(feature_importance_df.iterrows(), 1):
                f.write(f"{i}. {row['feature']}: {row['importance']:.4f}\n")
        
        # Cria gráfico do ranking
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(feature_importance_df)), feature_importance_df['importance'])
        plt.yticks(range(len(feature_importance_df)), feature_importance_df['feature'])
        plt.xlabel("Importância da Feature (Permutation Importance)")
        plt.title(f"Ranking de Importância das {len(self.feature_names)} Features Utilizadas")
        plt.gca().invert_yaxis()  # Inverte para mostrar a mais importante no topo
        plt.tight_layout()
        
        ranking_plot_path = os.path.join(self.results_path, "ranking_features_utilizadas.png")
        plt.savefig(ranking_plot_path)
        plt.close()
        
        print(f"Ranking salvo em: {ranking_path}")
        print(f"Gráfico do ranking salvo em: {ranking_plot_path}")
        
        return feature_importance_df

    def evaluate_model(self):
        """Avalia o modelo no conjunto de teste."""
        if self.model is None:
            raise ValueError("Modelo não foi treinado ainda.")
            
        y_pred = self.model.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred, target_names=self.class_names)
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        
        # Salvar resultados
        self.save_results(test_accuracy, report, conf_matrix)
        
        # Gerar ranking das features utilizadas
        feature_ranking = self.generate_feature_ranking()
        
        return test_accuracy, report, conf_matrix, feature_ranking

    def save_results(self, accuracy, report, conf_matrix):
        """Salva o relatório e a matriz de confusão em arquivos."""
        # Salvar relatório de texto com codificação UTF-8 para evitar problemas com caracteres
        report_path = os.path.join(self.results_path, "relatorio_classificacao.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"Acurácia no Teste: {accuracy:.4f}\n\n")
            f.write("="*30 + "\n")
            f.write("RELATÓRIO DE CLASSIFICAÇÃO\n")
            f.write("="*30 + "\n")
            f.write(report)

        # Salvar Matriz de Confusão
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Matriz de Confusão')
        plt.ylabel('Valor Real')
        plt.xlabel('Valor Previsto')
        plt.tight_layout()
        plot_path = os.path.join(self.results_path, "matriz_confusao.png")
        plt.savefig(plot_path)
        plt.close() # Fecha a figura para não exibir na GUI diretamente

    def plot_feature_importance(self):
        """Calcula e plota a importância das features usando Permutation Importance."""
        if self.model is None or self.X_test is None:
            raise ValueError("O modelo precisa ser treinado e avaliado primeiro.")

        # Calcula a importância por permutação no conjunto de teste
        result = permutation_importance(
            self.model, self.X_test, self.y_test, n_repeats=10, random_state=42, n_jobs=-1
        )
        
        sorted_idx = result.importances_mean.argsort()

        plt.figure(figsize=(12, 8))
        plt.barh(range(self.X_test.shape[1]), result.importances_mean[sorted_idx])
        plt.yticks(range(self.X_test.shape[1]), np.array(self.feature_names)[sorted_idx])
        plt.xlabel("Redução na Acurácia (Importância por Permutação)")
        plt.title("Importância das Features para o Modelo")
        plt.tight_layout()
        
        plot_path = os.path.join(self.results_path, "importancia_features.png")
        plt.savefig(plot_path)
        plt.close()
        return plot_path

    def plot_learning_curve(self):
        """Gera e salva a curva de aprendizado do SVM com os dados carregados."""
        if self.model is None or self.X_train is None or self.y_train is None:
            raise ValueError("O modelo precisa ser treinado e os dados carregados.")

        print(">>> Gerando curva de aprendizado do SVM...")
        train_sizes, train_scores, test_scores = learning_curve(
            self.model,
            self.X_train,
            self.y_train,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            shuffle=True,
            random_state=42
        )

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.figure(figsize=(10, 7))
        plt.title("Curva de Aprendizado - SVM")
        plt.xlabel("Quantidade de Dados de Treinamento")
        plt.ylabel("Acurácia")
        plt.grid(True)

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.2, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.2, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Acurácia Treinamento")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Acurácia Validação")

        plt.legend(loc="best")
        plt.tight_layout()
        plot_path = os.path.join(self.results_path, "curva_aprendizado.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Curva de aprendizado salva em: {plot_path}")
        return plot_path

    def plot_hardest_feature_distribution(self, X, y):
        import seaborn as sns
        import pandas as pd
        import matplotlib.pyplot as plt
        ranking_path = os.path.join(self.results_path, 'ranking_features_utilizadas.txt')
        feature_name = None
        # Lê o arquivo de ranking linha a linha e pega a última feature do ranking
        with open(ranking_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip() and line[0].isdigit()]
            if lines:
                last_line = lines[-1]
                # Exemplo de linha: '7. feature_name: 0.0123'
                parts = last_line.split('. ', 1)
                if len(parts) == 2:
                    feature_name = parts[1].split(':')[0].strip()
        if not feature_name or feature_name not in self.feature_names:
            print('Feature menos importante não encontrada nos dados.')
            return
        # Monta DataFrame para plot
        df = pd.DataFrame(X, columns=self.feature_names)
        df['classe'] = y
        plt.figure(figsize=(10,6))
        sns.violinplot(x='classe', y=feature_name, data=df, inner='quartile', palette='Set2')
        plt.title(f'Distribuição da feature mais difícil: {feature_name}\n(Sobreposição entre as classes)')
        plt.xlabel('Classe')
        plt.ylabel(feature_name)
        plt.tight_layout()
        plot_path = os.path.join(self.results_path, 'distribuicao_feature_mais_dificil.png')
        plt.savefig(plot_path)
        plt.close()
        print(f'Gráfico de distribuição salvo em: {plot_path}')



# --- Frontend: Interface Gráfica ---
class SVM_GUI:
    def __init__(self, root):
        self.root = root
        self.svm_system = SVMSystem()
        self.folder_paths = [tk.StringVar() for _ in range(7)]
        self.class_name_vars = [tk.StringVar() for _ in range(7)]
        self.class_input_frames = [] # Container para os widgets de input
        self.setup_gui()

    def setup_gui(self):
        self.root.title("GVA/NAAT - Sistema de Classificação SVM")
        self.root.geometry("1050x820")
        self.root.configure(bg='#f5f5f5')

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TLabel", font=('Arial', 10), background='#f5f5f5')
        style.configure("TButton", font=('Arial', 10, 'bold'), padding=5)
        style.configure("TEntry", font=('Arial', 10))
        style.configure("TFrame", background='#f5f5f5')
        style.configure("TLabelframe", padding=10, relief="solid", borderwidth=1, background='#f5f5f5')
        style.configure("TLabelframe.Label", font=('Arial', 12, 'bold'), background='#f5f5f5')

        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        try:
            gva_img = Image.open("gva.jpg").resize((150, 75), Image.LANCZOS)
            self.gva_photo = ImageTk.PhotoImage(gva_img)
            gva_label = ttk.Label(header_frame, image=self.gva_photo)
            gva_label.pack(side=tk.LEFT, padx=10, pady=5)
        except FileNotFoundError:
            gva_label = ttk.Label(header_frame, text="GVA Logo\n(gva.jpg não encontrado)")
            gva_label.pack(side=tk.LEFT, padx=10, pady=5)
        title_label = ttk.Label(header_frame, text="Classificador SVM Multiclasse", font=('Arial', 22, 'bold'), background='#f5f5f5', foreground='#1976d2')
        title_label.pack(side=tk.LEFT, expand=True)
        try:
            naat_img = Image.open("naat.jpg").resize((100, 75), Image.LANCZOS)
            self.naat_photo = ImageTk.PhotoImage(naat_img)
            naat_label = ttk.Label(header_frame, image=self.naat_photo)
            naat_label.pack(side=tk.RIGHT, padx=10, pady=5)
        except FileNotFoundError:
            naat_label = ttk.Label(header_frame, text="NAAT Logo\n(naat.jpg não encontrado)")
            naat_label.pack(side=tk.RIGHT, padx=10, pady=5)

        # --- Painel de configuração ---
        config_frame = ttk.Frame(main_frame)
        config_frame.pack(fill=tk.X, pady=10, anchor=tk.N)

        # Esquerda: Carregamento de Dados
        data_frame = ttk.LabelFrame(config_frame, text="1. Carregamento de Dados")
        data_frame.pack(side=tk.LEFT, fill=tk.X, padx=(0, 10), expand=True)
        num_classes_frame = ttk.Frame(data_frame)
        num_classes_frame.pack(fill=tk.X, padx=5, pady=(0, 10))
        ttk.Label(num_classes_frame, text="Número de Classes:").pack(side=tk.LEFT)
        self.num_classes_var = tk.IntVar(value=4)
        num_classes_combo = ttk.Combobox(
            num_classes_frame,
            textvariable=self.num_classes_var,
            values=[2, 3, 4, 5, 6, 7],
            state="readonly",
            width=5
        )
        num_classes_combo.pack(side=tk.LEFT, padx=5)
        num_classes_combo.bind("<<ComboboxSelected>>", self.update_class_inputs)
        inputs_container = ttk.Frame(data_frame)
        inputs_container.pack(fill=tk.X)
        for i in range(7):
            row_frame = ttk.Frame(inputs_container)
            self.class_name_vars[i].set(f"Condição {i+1}")
            ttk.Label(row_frame, text=f"Pasta da Condição {i+1}:", width=18).pack(side=tk.LEFT)
            entry = ttk.Entry(row_frame, textvariable=self.folder_paths[i])
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            btn = ttk.Button(row_frame, text="Selecionar...", command=lambda i=i: self.select_folder(i))
            btn.pack(side=tk.LEFT, padx=5)
            ttk.Label(row_frame, text="Nome:").pack(side=tk.LEFT, padx=(10, 5))
            name_entry = ttk.Entry(row_frame, textvariable=self.class_name_vars[i], width=20)
            name_entry.pack(side=tk.LEFT, padx=(0, 5))
            self.class_input_frames.append(row_frame)

        # Direita: Seleção de Features
        self.features_frame = ttk.LabelFrame(config_frame, text="2. Seleção de Features para Treinamento")
        self.features_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(10, 0), ipadx=10, ipady=10)
        ttk.Label(self.features_frame, text="Selecione as features desejadas para o treinamento do modelo:", font=('Arial', 10, 'italic')).pack(anchor='w', padx=10, pady=(5, 0))
        self.listbox_features = tk.Listbox(self.features_frame, selectmode='multiple', width=35, height=10, exportselection=False, font=('Arial', 10, 'bold'), bg='#e3f2fd', fg='#1976d2', relief='groove', bd=2)
        self.listbox_features.pack(side='left', padx=(10,0), pady=10, fill='y')
        self.scrollbar_features = ttk.Scrollbar(self.features_frame, orient='vertical', command=self.listbox_features.yview)
        self.scrollbar_features.pack(side='right', fill='y', padx=(0,10), pady=10)
        self.listbox_features.config(yscrollcommand=self.scrollbar_features.set)
        self.features_disponiveis = []

        # --- Frame de Controle e Resultados ---
        results_main_frame = ttk.Frame(main_frame)
        results_main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Seção de Controle ---
        control_frame = ttk.LabelFrame(results_main_frame, text="3. Ações")
        control_frame.pack(fill=tk.X, pady=(10, 10))

        # Frame para botões de ação principal (agrupados à esquerda)
        action_buttons_frame = ttk.Frame(control_frame)
        action_buttons_frame.pack(side=tk.LEFT, padx=5, pady=5)

        ttk.Button(action_buttons_frame, text="Carregar e Processar Dados", command=self.load_and_process).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_buttons_frame, text="Treinar Modelo", command=self.train_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_buttons_frame, text="Avaliar Modelo", command=self.evaluate_model).pack(side=tk.LEFT, padx=5)

        # Frame para botões de utilidade (agrupados à direita)
        utility_frame = ttk.Frame(control_frame)
        utility_frame.pack(side=tk.RIGHT, padx=5, pady=5)

        ttk.Button(utility_frame, text="Reiniciar", command=self.restart_training).pack(side=tk.LEFT, padx=5)
        ttk.Button(utility_frame, text="Abrir Pasta de Resultados", command=self.open_results_folder).pack(side=tk.LEFT, padx=5)

        # --- Seção de Resultados ---
        results_frame = ttk.LabelFrame(results_main_frame, text="Log e Resultados")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.results_text = scrolledtext.ScrolledText(results_frame, height=15, width=100, font=('Courier New', 10), relief=tk.FLAT)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # --- Status Bar ---
        self.status_var = tk.StringVar(value="Pronto.")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=5, font=('Arial', 9))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def update_class_inputs(self, event=None):
        """Mostra ou esconde os campos de input de acordo com o número de classes selecionado."""
        num_classes = self.num_classes_var.get()
        for i, frame in enumerate(self.class_input_frames):
            if i < num_classes:
                frame.pack(fill=tk.X, padx=5, pady=8)
            else:
                frame.pack_forget()

    # --- Funções de Callback da GUI ---
    def select_folder(self, index):
        folder_path = filedialog.askdirectory(title=f"Selecione a pasta para a Condição {index+1}")
        if folder_path:
            self.folder_paths[index].set(folder_path)

    def log(self, message):
        self.results_text.insert(tk.END, message + "\n")
        self.results_text.see(tk.END)
        self.root.update_idletasks()

    def load_and_process(self):
        self.log(">>> Iniciando carregamento e processamento dos dados...")
        self.status_var.set("Carregando dados...")
        try:
            num_classes = self.num_classes_var.get()
            paths = [var.get() for var in self.folder_paths[:num_classes]]
            names = [var.get() for var in self.class_name_vars[:num_classes]]
            
            if any(not p for p in paths) or any(not n for n in names):
                raise ValueError(f"Por favor, selecione todas as {num_classes} pastas e preencha todos os nomes das condições.")

            X, y = self.svm_system.load_data_from_folders(paths, names)
            self.log(f"Total de {X.shape[0]} amostras carregadas de {len(names)} classes, com {X.shape[1]} features.")
            
            # Preencher lista de features para seleção
            self.features_disponiveis = self.svm_system.feature_names
            self.listbox_features.delete(0, tk.END)
            for feat in self.features_disponiveis:
                self.listbox_features.insert(tk.END, feat)
            self.features_frame.pack(fill=tk.X, padx=10, pady=(0, 10))  # Agora garante que aparece
            self.log("Selecione as features desejadas na lista acima antes de treinar o modelo.")
            
            self.svm_system.prepare_data(X, y)
            self.log("Dados divididos em conjuntos de treino e teste e normalizados.")
            self.log(f"- Treino: {self.svm_system.X_train.shape[0]} amostras.")
            self.log(f"- Teste: {self.svm_system.X_test.shape[0]} amostras.")
            # Feedback da normalização
            self.log(f"Normalização utilizada: {self.svm_system.normalization_used}")
            self.status_var.set(f"Dados carregados e normalizados com {self.svm_system.normalization_used}. Pronto para treinar.")

        except Exception as e:
            messagebox.showerror("Erro no Carregamento", str(e))
            self.log(f"ERRO: {e}\n")
            self.status_var.set("Erro ao carregar dados.")

    def train_model(self):
        # Antes de treinar, filtrar X_train e X_test pelas features selecionadas
        indices = self.listbox_features.curselection()
        if not indices:
            messagebox.showerror('Erro', 'Selecione ao menos uma feature na lista para treinar.')
            return
        features_escolhidas = [self.features_disponiveis[i] for i in indices]
        # Filtra os dados do SVMSystem
        idxs = [self.svm_system.feature_names.index(f) for f in features_escolhidas]
        self.svm_system.X_train = self.svm_system.X_train[:, idxs]
        self.svm_system.X_test = self.svm_system.X_test[:, idxs]
        self.svm_system.feature_names = features_escolhidas
        # Continua fluxo normal
        self.log(f"Features selecionadas para treinamento: {', '.join(features_escolhidas)}")
        if self.svm_system.X_train is None:
            messagebox.showerror("Erro", "Carregue e processe os dados primeiro.")
            return

        self.log(">>> Iniciando treinamento do modelo SVM...")
        self.status_var.set("Treinando modelo com GridSearchCV...")
        try:
            self.log("Iniciando busca em grade (GridSearchCV) para C e Gamma.")
            self.log("Isso pode levar alguns minutos...")
            
            train_accuracy, best_params = self.svm_system.train_model()
            
            self.log(f"Busca em grade concluída!")
            self.log(f"Melhores parâmetros encontrados: {best_params}")
            self.log(f"Modelo treinado com sucesso com os melhores parâmetros!")
            self.log(f"Acurácia no conjunto de treino: {train_accuracy:.4f}\n")
            self.status_var.set(f"Modelo treinado. Acurácia de treino: {train_accuracy:.4f}")

            # Gerar curva de aprendizado
            self.log(">>> Gerando curva de aprendizado...")
            curva_path = self.svm_system.plot_learning_curve()
            self.log(f"Curva de aprendizado salva em: {curva_path}\n")

        except Exception as e:
            messagebox.showerror("Erro no Treinamento", str(e))
            self.log(f"ERRO: {e}\n")
            self.status_var.set("Erro ao treinar o modelo.")
            
    def evaluate_model(self):
        if self.svm_system.model is None:
            messagebox.showerror("Erro", "Treine o modelo primeiro.")
            return

        self.log(">>> Iniciando avaliação do modelo no conjunto de teste...")
        self.status_var.set("Avaliando modelo...")
        try:
            accuracy, report, conf_matrix, feature_ranking = self.svm_system.evaluate_model()
            
            self.log("=== AVALIAÇÃO DO MODELO (TESTE) ===")
            self.log(f"Acurácia no Teste: {accuracy:.4f}")
            self.log("\n--- Relatório de Classificação ---")
            self.log(report)
            
            # Gera e salva o gráfico de importância das features automaticamente
            self.log("\n>>> Gerando análise de importância das features...")
            plot_path = self.svm_system.plot_feature_importance()
            self.log(f"Gráfico de importância das features salvo em: {plot_path}\n")
            
            self.status_var.set(f"Avaliação concluída. Acurácia de teste: {accuracy:.4f}")
            messagebox.showinfo("Avaliação Concluída", f"Resultados e gráfico de features salvos em '{self.svm_system.results_path}'.\nAcurácia no teste: {accuracy:.4f}")

            # Gerar gráfico de distribuição da feature menos importante
            self.svm_system.plot_hardest_feature_distribution(self.svm_system.X_test, self.svm_system.y_test)

        except Exception as e:
            messagebox.showerror("Erro na Avaliação", str(e))
            self.log(f"ERRO: {e}\n")
            self.status_var.set("Erro ao avaliar o modelo.")

    def restart_training(self):
        """Reinicia a aplicação para um novo treinamento, limpando os resultados anteriores."""
        if not messagebox.askyesno("Confirmar Reinício", "Isso irá apagar todos os arquivos na pasta 'resultados_svm' e reiniciar a interface. Deseja continuar?"):
            return

        self.log(">>> Reiniciando o sistema de treinamento...")
        
        try:
            # 1. Limpa a pasta de resultados
            self.svm_system.clear_results_folder()
            self.log("Pasta de resultados foi limpa.")

            # 2. Reinicia o backend
            self.svm_system = SVMSystem()

            # 3. Reseta a GUI
            for i in range(7):
                self.folder_paths[i].set("")
                self.class_name_vars[i].set(f"Condição {i+1}")
            
            self.num_classes_var.set(4) # Volta para o padrão de 4 classes
            self.update_class_inputs()
            
            # Limpa a caixa de log
            self.results_text.delete('1.0', tk.END)
            
            self.status_var.set("Sistema reiniciado. Pronto para carregar novos dados.")
            
            self.log("Sistema reiniciado com sucesso. Por favor, carregue novos dados.")

        except Exception as e:
            messagebox.showerror("Erro ao Reiniciar", f"Ocorreu um erro: {str(e)}")
            self.log(f"ERRO: Falha ao reiniciar o sistema: {e}")

    def open_results_folder(self):
        path = os.path.realpath(self.svm_system.results_path)
        try:
            os.startfile(path)
        except AttributeError:
            # Para Linux/Mac
            import subprocess
            subprocess.Popen(['xdg-open', path])




if __name__ == "__main__":
    root = tk.Tk()
    app = SVM_GUI(root)
    root.mainloop() 