import streamlit as st
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker
import pandas as pd
import plotly.express as px
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import mysql.connector


# Charger les variables d'environnement
load_dotenv()

# Fonction pour initialiser la connexion à la base de données
def init_database(user: str, password: str, host: str, port: str, database: str):
    try:
        conn = mysql.connector.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database
        )
        st.success("Connexion réussie à la base de données MySQL.")
        return conn
    except mysql.connector.Error as err:
        st.error(f"Erreur lors de la connexion à la base de données : {err}")
        return None

# Fonction pour créer le moteur SQLAlchemy
def create_sqlalchemy_engine(user, password, host, port, database):
    try:
        conn_str = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
        engine = create_engine(conn_str)
        st.success("Moteur SQLAlchemy créé avec succès.")
        return engine
    except Exception as err:
        st.error(f"Erreur lors de la création du moteur SQLAlchemy : {err}")
        return None


# Fonction pour obtenir les noms des tables dans la base de données
def get_table_names(engine):
    inspector = inspect(engine)
    return inspector.get_table_names()

# Fonction pour obtenir les noms des colonnes dans une table
def get_column_names(engine, table_name):
    inspector = inspect(engine)
    columns = inspector.get_columns(table_name)
    return [column['name'] for column in columns]

# Fonction pour tronquer l'historique des conversations
def truncate_chat_history(chat_history, max_length=5):
    return chat_history[-max_length:]

def get_response(user_query: str, engine, chat_history: list):
    truncated_chat_history = truncate_chat_history(chat_history)

    # Requêtes SQL spécifiques basées sur la question de l'utilisateur
    if "nom et prix des médicaments" in user_query.lower():
        sql_query = """
            SELECT name, prix
            FROM medicaments
        """
    elif "pharmacies en garde" in user_query.lower():
        sql_query = """
            SELECT nom, phone_number, region, zone
            FROM pharmacies
            WHERE garde = TRUE
        """
    elif "catégories des médecins" in user_query.lower():
        sql_query = """
            SELECT name
            FROM categories
        """
    else:
        # Modèle général pour générer des requêtes SQL basées sur la question
        sql_template = """
            Vous êtes un analyste de données dans une entreprise de pharmacie. Vous interagissez avec un utilisateur qui pose des questions sur la base de données de la pharmacie.
            Basé sur le schéma de la table ci-dessous, écrivez une requête SQL qui répondrait à la question de l'utilisateur. Prenez en compte l'historique de la conversation.

            <SCHEMA>{schema}</SCHEMA>

            Historique de la conversation : {chat_history}

            Écrivez uniquement la requête SQL et rien d'autre. Ne pas encapsuler la requête SQL dans un autre texte, même pas des backticks.

            Question : {question}
            Requête SQL :
        """

        sql_prompt = ChatPromptTemplate.from_template(sql_template)

        # Utilisation de l'API Groq pour générer les requêtes SQL
        groq_api_key = os.getenv("GROQ_API_KEY")

        llm_sql = ChatGroq(
            model="mixtral-8x7b-32768",
            temperature=0,
            groq_api_key=groq_api_key
        )

        def get_schema(_):
            tables = get_table_names(engine)
            schema = ""
            for table in tables:
                columns = get_column_names(engine, table)
                schema += f"{table}: {', '.join(columns)}\n"
            return schema

        # Chaine pour générer les requêtes SQL via le LLM
        sql_chain = (
            RunnableMap({
                "schema": get_schema,
                "chat_history": RunnablePassthrough(),
                "question": RunnablePassthrough()
            })
            | sql_prompt
            | llm_sql
            | StrOutputParser()
        )

        sql_query = sql_chain.invoke({
            "question": user_query,
            "chat_history": truncated_chat_history,
        })

    # Exécuter la requête SQL générée et retourner les résultats
    try:
        with engine.connect() as connection:
            result = connection.execute(text(sql_query))

            # Affichage de la requête SQL exécutée
            st.info(f"Requête SQL exécutée :\n{sql_query}")

            # Si vous êtes sûr des colonnes, vous pouvez les récupérer explicitement
            rows = result.fetchall()

            if not sql_query:
                st.error("Aucune requête SQL n'a été générée.")
                return

            if rows:
                # Extraire les noms de colonnes à partir des métadonnées
                column_names = result.keys() if result.keys() else [f"column_{i}" for i in range(len(rows[0]))]

                # Convertir en DataFrame pour un affichage plus facile
                df = pd.DataFrame(rows, columns=column_names)
                st.write(df)

                # Affichage du graphique si deux colonnes sont sélectionnées
                if df.shape[1] == 2:
                    x_axis = df.columns[0]  # Utilisation de la première colonne comme axe X
                    y_axis = df.columns[1]  # Utilisation de la deuxième colonne comme axe Y
                    fig = px.line(df, x=x_axis, y=y_axis, title=f"Graphique de {y_axis} en fonction de {x_axis}")
                    st.plotly_chart(fig)

                # Affichage du graphique circulaire si 3 colonnes ou plus
                if df.shape[1] >= 3:
                    fig_pie = px.pie(df, values=df.columns[2], names=df.columns[0], title="Répartition")
                    st.plotly_chart(fig_pie)

            else:
                st.write("La requête n'a retourné aucun résultat.")
    except Exception as e:
        st.write(f"Erreur lors de l'exécution de la requête SQL : {e}")

        # Affichage de la requête SQL exécutée
        st.info(f"Requête SQL exécutée :\n{sql_query}")



# Interface utilisateur avec Streamlit
st.title("Chat avec la base de données SQL")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'db' not in st.session_state:
    st.session_state.db = None

# Formulaire de connexion à la base de données
with st.sidebar.form(key='connexion_form'):
    user = st.text_input("Nom d'utilisateur")
    password = st.text_input("Mot de passe", type="password")
    host = st.text_input("Hôte")
    port = st.text_input("Port", value="3306")
    database = st.text_input("Base de données")
    submit_button = st.form_submit_button(label='Se connecter')
    
    if submit_button:
        conn = init_database(user, password, host, port, database)
        if conn:
            st.session_state.db = create_sqlalchemy_engine(user, password, host, port, database)

# Si la base de données est connectée, afficher l'interface de chat SQL
if st.session_state.db:
    user_query = st.text_input("Entrez votre requête SQL")

    if st.button("Exécuter la requête"):
        response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        if response:
            st.session_state.chat_history.append(AIMessage(content=str(response)))
            df = pd.DataFrame(response)
            st.write(df)
            # Affichage graphique
            if len(df.columns) == 2:
                fig = px.line(df, x=df.columns[0], y=df.columns[1], title="Graphique des résultats")
                st.plotly_chart(fig)
            elif len(df.columns) > 2:
                fig_pie = px.pie(df, names=df.columns[0], values=df.columns[2], title="Répartition des données")
                st.plotly_chart(fig_pie)

    # Affichage de l'historique de la conversation
    st.write("Historique de la conversation :")
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            st.text(f"Utilisateur : {message.content}")
        else:
            st.text(f"IA : {message.content}")
    
    # Option de réinitialiser l'historique des conversations
    if st.sidebar.button("Réinitialiser l'historique des conversations"):
        st.session_state.chat_history = []
        st.success("Historique réinitialisé.")
