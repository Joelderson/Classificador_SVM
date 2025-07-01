#!/usr/bin/env python3
"""
Script de Setup para o Sistema de Classificação SVM - GVA/NAAT
Este script facilita a instalação e execução do sistema.
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Verifica se a versão do Python é compatível."""
    if sys.version_info < (3, 7):
        print("❌ Erro: Python 3.7 ou superior é necessário.")
        print(f"Versão atual: {sys.version}")
        return False
    print(f"✅ Python {sys.version.split()[0]} detectado.")
    return True

def install_requirements():
    """Instala as dependências do projeto."""
    print("\n📦 Instalando dependências...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependências instaladas com sucesso!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro ao instalar dependências: {e}")
        return False

def check_files():
    """Verifica se os arquivos necessários estão presentes."""
    required_files = [
        "classificador_svm_multiclasse.py",
        "requirements.txt",
        "README.md",
        "gva.jpg",
        "naat.jpg"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Arquivos ausentes: {', '.join(missing_files)}")
        return False
    
    print("✅ Todos os arquivos necessários estão presentes.")
    return True

def check_data_folders():
    """Verifica se as pastas de dados estão presentes."""
    data_folders = [
        "features_separadas_svm/Features_separadas_vetores_menores",
        "features_separadas_svm/Features_vetores_maiores"
    ]
    
    missing_folders = []
    for folder in data_folders:
        if not os.path.exists(folder):
            missing_folders.append(folder)
    
    if missing_folders:
        print(f"⚠️  Pastas de dados ausentes: {', '.join(missing_folders)}")
        print("   Os dados de exemplo podem não estar disponíveis.")
    else:
        print("✅ Pastas de dados de exemplo encontradas.")
    
    return True

def run_application():
    """Executa a aplicação principal."""
    print("\n🚀 Iniciando o Sistema de Classificação SVM...")
    try:
        subprocess.run([sys.executable, "classificador_svm_multiclasse.py"])
    except KeyboardInterrupt:
        print("\n👋 Aplicação encerrada pelo usuário.")
    except Exception as e:
        print(f"❌ Erro ao executar a aplicação: {e}")

def main():
    """Função principal do script de setup."""
    print("=" * 60)
    print("🔧 Setup do Sistema de Classificação SVM - GVA/NAAT")
    print("=" * 60)
    
    # Verificações iniciais
    if not check_python_version():
        sys.exit(1)
    
    if not check_files():
        print("\n❌ Setup falhou. Verifique se todos os arquivos estão presentes.")
        sys.exit(1)
    
    check_data_folders()
    
    # Instalação de dependências
    if not install_requirements():
        print("\n❌ Setup falhou. Erro na instalação das dependências.")
        sys.exit(1)
    
    print("\n✅ Setup concluído com sucesso!")
    print("\n📋 Próximos passos:")
    print("1. A aplicação será iniciada automaticamente")
    print("2. Configure o número de classes (2-7)")
    print("3. Selecione as pastas com os dados CSV")
    print("4. Escolha as features para treinamento")
    print("5. Execute o treinamento e avaliação")
    
    # Pergunta se deseja executar a aplicação
    response = input("\n🎯 Deseja executar a aplicação agora? (s/n): ").lower().strip()
    if response in ['s', 'sim', 'y', 'yes']:
        run_application()
    else:
        print("\n💡 Para executar manualmente, use:")
        print("   python classificador_svm_multiclasse.py")

if __name__ == "__main__":
    main() 