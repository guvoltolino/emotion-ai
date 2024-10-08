# EmotiSense - Detector de Emoções

EmotiSense é uma aplicação que detecta emoções em textos ou falas utilizando um modelo de linguagem treinado. O projeto é desenvolvido com Streamlit e a API Google Generative AI, e inclui reconhecimento de voz para análise de emoções.

## Índice

- [Pré-requisitos](#pré-requisitos)
- [Instalação](#instalação)
- [Contribuição](#contribuição)

## Pré-requisitos

Antes de começar, certifique-se de ter o Python 3.7 ou superior instalado em seu sistema. Você também precisará de um arquivo `.env` com suas credenciais da API Google.

## Instalação

1. **Clone o Repositório**

   Primeiro, clone este repositório para o seu ambiente local.

2. **Crie um ambiente virtual utilizando o venv:**

    python -m venv venv

3. **Ative o Ambiente Virtual**

    Ative o ambiente virtual. O comando varia de acordo com o seu sistema operacional:

    Windows:

    venv\Scripts\activate

    macOS/Linux:

    source venv/bin/activate

4. **Instale as Dependências**

    Com o ambiente virtual ativado, instale as dependências do projeto:
    pip install -r requirements.txt

5. **Configure as Variáveis de Ambiente**

    Crie um arquivo .env na raiz do projeto e adicione suas credenciais da API Google:

    link para conseguir crendeciais: https://ai.google.dev/

    modelo utilizado: gemini-1.5-pro

6. **Iniciar a aplicação Streamlit com o seguinte comando:**

    streamlit run main.py
    Substitua main.py pelo nome do arquivo principal do seu projeto se for diferente.

7. **Acesse a Aplicação**

    Abra o navegador e acesse http://localhost:8501 para interagir com a aplicação EmotiSense.

## Contribuição

Se você encontrar algum problema ou tiver sugestões de melhorias, fique à vontade para abrir uma issue ou enviar um pull request.

Obrigado por usar o EmotiSense!