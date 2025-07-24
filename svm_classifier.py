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
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

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
        """Carrega dados de um número variável de pastas, tentando extrair nomes das features do cabeçalho."""
        all_features = []
        all_labels = []
        
        if not folder_paths or len(folder_paths) != len(class_names):
            raise ValueError("O número de pastas e nomes de classes não corresponde ou está vazio.")
            
        self.class_names = class_names
        
        expected_num_features = None
        feature_names_from_header = None # Guardará os nomes das features do cabeçalho

        for i, folder_path in enumerate(folder_paths):
            if not folder_path or not os.path.isdir(folder_path):
                raise FileNotFoundError(f"Pasta para a Condição {i+1} não encontrada ou inválida: {folder_path}")

            files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
            if not files:
                raise ValueError(f"Nenhum arquivo .csv encontrado na pasta para a Condição {i+1}: {folder_path}")

            for file in files:
                file_path = os.path.join(folder_path, file)
                try:
                    # TENTATIVA 1: Assumir que não há cabeçalho e o arquivo é puramente numérico.
                    df_attempt1 = pd.read_csv(file_path, header=None)
                    numeric_df_attempt1 = df_attempt1.apply(pd.to_numeric, errors='coerce')

                    if not numeric_df_attempt1.isnull().values.any():
                        # SUCESSO na Tentativa 1: O arquivo é puramente numérico.
                        segment_data = numeric_df_attempt1.values.flatten()
                    else:
                        # TENTATIVA 2: Falha na primeira tentativa, assumir que há um cabeçalho.
                        print(f"Aviso: Texto detectado em '{file}'. Tentando processar com cabeçalho...")
                        df_attempt2 = pd.read_csv(file_path, header=0) # Trata a primeira linha como cabeçalho
                        
                        # Se ainda não capturamos os nomes das features, fazemos agora.
                        if feature_names_from_header is None:
                            feature_names_from_header = df_attempt2.columns.tolist()

                        numeric_df_attempt2 = df_attempt2.apply(pd.to_numeric, errors='coerce')

                        if not numeric_df_attempt2.isnull().values.any():
                            # SUCESSO na Tentativa 2: Cabeçalho removido com sucesso.
                            print(f"  -> Sucesso: Cabeçalho de '{file}' processado e arquivo carregado.")
                            segment_data = numeric_df_attempt2.values.flatten()
                        else:
                            # FALHA FINAL: O arquivo contém dados não numéricos mesmo após remover o cabeçalho.
                            print(f"  -> Falha: O arquivo '{file}' contém dados inválidos e será ignorado.")
                            continue # Pula este arquivo problemático

                    # Se o número de features ainda não foi definido, usa o deste arquivo
                    if expected_num_features is None:
                        if segment_data.shape[0] > 0:
                            expected_num_features = segment_data.shape[0]
                            print(f"Sistema detectou {expected_num_features} features por segmento.")
                        else:
                            print(f"Aviso: Arquivo {file} está vazio ou inválido. Ignorando.")
                            continue # Pula para o próximo arquivo

                    # Verifica se o segmento atual tem o número de features esperado
                    if segment_data.shape[0] != expected_num_features:
                        print(f"Aviso: O arquivo {file} tem {segment_data.shape[0]} features, mas o esperado é {expected_num_features}. Arquivo ignorado.")
                        continue
                        
                    all_features.append(segment_data)
                    all_labels.append(i) # Rótulo da classe (0 a 3)
                except Exception as e:
                    print(f"Erro crítico ao processar o arquivo '{file}': {e}. Ignorando.")

        if not all_features:
            raise ValueError("Nenhum dado válido foi carregado. Verifique os arquivos CSV e sua consistência.")
            
        X = np.array(all_features)
        y = np.array(all_labels)

        # Define os nomes das features: usa os do cabeçalho se forem válidos, senão, usa genéricos.
        if feature_names_from_header and len(feature_names_from_header) == X.shape[1]:
            print("Usando nomes das features extraídos do cabeçalho do arquivo CSV.")
            self.feature_names = feature_names_from_header
        else:
            if feature_names_from_header:
                print("Aviso: Nomes das features do cabeçalho não correspondem à quantidade de dados. Usando nomes genéricos.")
            self.feature_names = [f"Feature_{j+1}" for j in range(X.shape[1])]
        
        return X, y

    def prepare_data(self, X, y, test_size=0.3, random_state=42):
        """Divide os dados em treino/teste e normaliza."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

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

# --- Frontend: Interface Gráfica ---
class SVM_GUI:
    def __init__(self, root):
        self.root = root
        self.svm_system = SVMSystem()
        self.folder_paths = [tk.StringVar() for _ in range(5)]
        self.class_name_vars = [tk.StringVar() for _ in range(5)]
        self.class_input_frames = [] # Container para os widgets de input
        self.setup_gui()

    def setup_gui(self):
        """Configura a interface gráfica com layout profissional e número de classes flexível."""
        self.root.title("GVA/NAAT - Sistema de Classificação SVM")
        self.root.geometry("950x800")

        # --- Estilo ---
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TLabel", font=('Arial', 10))
        style.configure("TButton", font=('Arial', 10, 'bold'), padding=5)
        style.configure("TEntry", font=('Arial', 10))
        style.configure("TFrame")
        style.configure("TLabelframe", padding=10, relief="solid", borderwidth=1)
        style.configure("TLabelframe.Label", font=('Arial', 12, 'bold'))

        # --- Layout Principal ---
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- Header com Logos ---
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

        title_label = ttk.Label(header_frame, text="Classificador SVM Multiclasse", font=('Arial', 20, 'bold'))
        title_label.pack(side=tk.LEFT, expand=True)

        try:
            naat_img = Image.open("naat.jpg").resize((100, 75), Image.LANCZOS)
            self.naat_photo = ImageTk.PhotoImage(naat_img)
            naat_label = ttk.Label(header_frame, image=self.naat_photo)
            naat_label.pack(side=tk.RIGHT, padx=10, pady=5)
        except FileNotFoundError:
            naat_label = ttk.Label(header_frame, text="NAAT Logo\n(naat.jpg não encontrado)")
            naat_label.pack(side=tk.RIGHT, padx=10, pady=5)
        
        # --- Frame de Configuração (Dados e Parâmetros) ---
        config_frame = ttk.Frame(main_frame)
        config_frame.pack(fill=tk.X, pady=10, anchor=tk.N)

        # --- Seção de Carregamento de Dados ---
        data_frame = ttk.LabelFrame(config_frame, text="1. Carregamento de Dados")
        data_frame.pack(side=tk.LEFT, fill=tk.X, padx=(0, 10), expand=True)

        #--- Widget para selecionar número de classes ---
        num_classes_frame = ttk.Frame(data_frame)
        num_classes_frame.pack(fill=tk.X, padx=5, pady=(0, 10))
        ttk.Label(num_classes_frame, text="Número de Classes:").pack(side=tk.LEFT)
        self.num_classes_var = tk.IntVar(value=4)
        num_classes_combo = ttk.Combobox(
            num_classes_frame,
            textvariable=self.num_classes_var,
            values=[2, 3, 4, 5],
            state="readonly",
            width=5
        )
        num_classes_combo.pack(side=tk.LEFT, padx=5)
        num_classes_combo.bind("<<ComboboxSelected>>", self.update_class_inputs)

        # Container para os inputs das classes
        inputs_container = ttk.Frame(data_frame)
        inputs_container.pack(fill=tk.X)

        for i in range(5):
            # Usar um frame para cada linha para melhor alinhamento
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

        # --- Seção de Otimização ---
        params_frame = ttk.LabelFrame(config_frame, text="2. Otimização")
        params_frame.pack(side=tk.LEFT, fill=tk.BOTH)
        
        info_text = "Os melhores hiperparâmetros\n(C e Gamma) serão encontrados\nautomaticamente via GridSearch\ncom Validação Cruzada."
        info_label = ttk.Label(params_frame, text=info_text, justify=tk.CENTER, font=('Arial', 10, 'italic'))
        info_label.pack(expand=True, padx=20, pady=20)

        # Atualiza a visibilidade inicial dos inputs
        self.update_class_inputs()

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
            
            self.svm_system.prepare_data(X, y)
            self.log("Dados divididos em conjuntos de treino e teste e normalizados.")
            self.log(f"- Treino: {self.svm_system.X_train.shape[0]} amostras.")
            self.log(f"- Teste: {self.svm_system.X_test.shape[0]} amostras.")
            self.log("Pronto para treinar o modelo.\n")
            self.status_var.set("Dados carregados com sucesso. Pronto para treinar.")

        except Exception as e:
            messagebox.showerror("Erro no Carregamento", str(e))
            self.log(f"ERRO: {e}\n")
            self.status_var.set("Erro ao carregar dados.")

    def train_model(self):
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
            for i in range(5):
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