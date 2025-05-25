# NLP-Based SQL Chatbot with MySQL

I'm excited to share a project I’ve been working on that bridges the power of LLMs and real-time database interactions! I developed a smart Streamlit application that allows users to interact with a MySQL database using natural language queries. Powered by LangChain, the Mixtral-8x7B model via Groq API, and enhanced with Plotly for data visualization, this app dynamically interprets user questions, generates SQL queries, fetches results, and visualizes the data. Whether it's asking for available medications, pharmacy schedules, or detailed business insights, the app transforms technical data into accessible answers—all through an intuitive chat interface. This project demonstrates how generative AI can simplify internal data access, optimize knowledge sharing, and empower non-technical users.

## Installation

1. Clonez le repository
2. Installez les dépendances : `pip install -r requirements.txt`
3. Configurez votre fichier `.env`
4. Lancez l'application : `python src/app.py`

## Configuration

Créez un fichier `.env` avec les variables suivantes :

## Features
- **Natural Language Processing**: Uses GPT-4 to interpret and respond to user queries in natural language.
- **SQL Query Generation**: Dynamically generates SQL queries based on the user's natural language input.
- **Database Interaction**: Connects to a SQL database to retrieve query results, demonstrating practical database interaction.
- **Streamlit GUI**: Features a user-friendly interface built with Streamlit, making it easy for users of all skill levels.
- **Python-based**: Entirely coded in Python, showcasing best practices in software development with modern technologies.

## Brief Explanation of How the Chatbot Works

The chatbot works by taking a user's natural language query, converting it into a SQL query using GPT-4, executing the query on a SQL database, and then presenting the results back to the user in natural language. This process involves several steps of data processing and interaction with the OpenAI API and a SQL database, all seamlessly integrated into a Streamlit application.


```bash
pip install -r requirements.txt
```

Create your own .env file with the necessary variables, including your OpenAI API key:

```bash
OPENAI_API_KEY=[your-openai-api-key]
```

## Usage
To launch the Streamlit app and interact with the chatbot:

```bash
streamlit run app.py
```


**Note**: This project is intended for educational and research purposes. Please ensure compliance with the terms of use and guidelines of any APIs or services used.


Happy Coding! 🚀👨‍💻🤖

---

*If you find this project helpful, please consider giving it a star!*

---
