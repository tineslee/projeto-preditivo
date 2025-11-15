# Previsão de Turnover (RH) – IA Preditiva

Este projeto foi desenvolvido como parte de um estudo voltado para necessidades de predição em áreas de RH com grande fluxo. Ele utiliza aprendizado de máquina para prever a probabilidade de turnover (saída de funcionários) com base em dados sintéticos simulando cenários reais de RH.

![Dashboard Turnover RH](cover.png)

---

## Funcionalidades

- Geração de dados sintéticos simulando perfis de colaboradores
- Treinamento de modelo Random Forest com pipeline de pré-processamento
- Avaliação com métricas robustas: Accuracy, Precision, Recall, F1, ROC-AUC
- Dashboard interativo com Streamlit para upload e previsão
- Ambiente virtual para isolamento de dependências

---

## Tecnologias utilizadas

| Categoria         | Ferramentas e Bibliotecas                          |
|------------------|----------------------------------------------------|
| Linguagem         | Python                                             |
| Machine Learning  | Scikit-learn, Joblib                              |
| Manipulação de dados | Pandas                                         |
| Visualização      | Streamlit                                          |
| Ambiente          | Virtualenv, VS Code                                |
| Versionamento     | Git + GitHub                                       |

---

## Estrutura do projeto
projeto-preditivo/
├── data/ # Arquivos de dados sintéticos 
├── notebooks/ # Análises exploratórias (opcional)
├── src/ # Scripts principais │ 
├── 01_data_generation.py # Geração de dados sintéticos │ 
├── 03_model_ready_data.py # Treinamento e salvamento do modelo │ 
├── 05_app_streamlit.py # Dashboard interativo 
├── .gitignore 
├── README.md 
└── requirements.txt


---

## Como rodar localmente

1. **Clone o repositório**
   ```bash
   git clone https://github.com/seu-usuario/projeto-preditivo.git
   cd projeto-preditivo

2. **Crie o ambiente virtual**
   python3 -m venv venv
   source venv/bin/activate

3. **Instale as dependências**
pip install -r requirements.txt

4. **Gere os dados sintéticos**
python src/01_data_generation.py

5. **Treine e salve o modelo**
python src/03_model_ready_data.py

6. **Rode o dashboard**
streamlit run src/05_app_streamlit.py


## Exemplo de uso

Envie um CSV com os campos do dataset sintético (sem a coluna turnover) e visualize:

    Probabilidade de turnover (turnover_prob)

    Previsão binária (turnover_pred)

    Gráfico de distribuição

    Média agregada de risco

## Sobre mim

Desenvolvido por Thais Inês, apaixonada por IA e projetos com impacto real.



