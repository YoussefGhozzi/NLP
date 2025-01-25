# import streamlit as st
# from sqlalchemy import create_engine, text, inspect
# from sqlalchemy.orm import sessionmaker
# import pandas as pd
# import plotly.express as px
# from langchain_core.messages import AIMessage, HumanMessage
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough, RunnableMap
# from langchain_core.output_parsers import StrOutputParser
# from langchain_groq import ChatGroq
# from dotenv import load_dotenv
# import os
# import mysql.connector


# # Charger les variables d'environnement
# load_dotenv()

# # # Mapping des noms de colonnes
# # column_mapping = {
# #     "client_folder_number": "NumDoss",
# #     "client_room_number": "NumCha",
# #     "type_service": "ServiceHospitalisation",
# #     "hospital_service": "ServiceHospitalisation",
# #     "BYMotifAdmission": "ServiceHospitalisation",
# #     "medical_service": "ServiceHospitalisation",
# #     "reason_MotifAdmission": "ServiceHospitalisation",
# #     "MotifAdmissionFROM": "ServiceHospitalisation",
# #     "hospitalization_service": "ServiceHospitalisation",
# #     "department_pec": "ServiceHospitalisation",
# #     "gender": "sex avec 1 pour male et 0 pour female",
# #     "birth_date": "DatNai",
# #     "nationality": "Nationalite",
# #     "patient_Nationalite": "Nationalite",
# #     "national_card_number": "NumCIN",
# #     "phone_number": "NumTel",
# #     "attending_physician": "MedecinTraitant",
# #     "physician_name": "MedecinTraitant",
# #     "doctor_name": "MedecinTraitant",
# #     "client_name": "NomPatient",
# #     "customer": "NomPatient",
# #     "NomPatient_name": "NomPatient",
# #     "attendance_reason": "NomPatient",
# #     "customer_name": "NomPatient",
# #     "arrival_date": "DatArr",
# #     "departure_date": "DatDep",
# #     "exit": "TypeSortie",
# #     'invoice_total_without_tax': "TotFactureHT",
# #     "salary": "TotFactureHT",
# #     "revenue_total_without_tax": "TotFactureHT",
# #     "fee_total": "TotalHonoraire",
# #     "tva_fee": "TotTVAFacture",
# #     "admission_type": "NatureAdmission",
# #     "admission_reason": "MotifAdmission",
# #     "demographic_reason": "MotifAdmission",
# #     "admission": "MotifAdmission",
# #     "type_reason": "MotifAdmission",
# #     "reason_reason": "MotifAdmission",
# #     "attending_reason": "MotifAdmission",
# #     "MotifAdmissions": "MotifAdmission",
# #     "MotifAdmission": "MotifAdmission",
# #     "Admission_reason": "MotifAdmission",
# #     "MotifAdmissionFROM": "MotifAdmission",
# #     "invoice_total_with_tax": "TotFactureTTC",
# #     "corresponding_doctor": "MedecinCorrespondant",
# #     "company_pec": "SocietePEC",
# #     "client": "VPatientAI",
# #     "demographic_name": "MedecinTraitant",
# #     "client_state": "EtaCli",
# # }

# # Fonction pour initialiser la connexion à la base de données
# def init_database(user: str, password: str, host: str, port: str, database: str):
#     try:
#         conn = mysql.connector.connect(
#             host=host,
#             port=port,
#             user=user,
#             password=password,
#             database=database
#         )
#         st.success("Connexion réussie à la base de données MySQL.")
#         return conn
#     except mysql.connector.Error as err:
#         st.error(f"Erreur lors de la connexion à la base de données : {err}")
#         return None
    
# # Fonction pour créer le moteur SQLAlchemy
# def create_sqlalchemy_engine(user, password, host, port, database):
#     try:
#         conn_str = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
#         engine = create_engine(conn_str)
#         st.success("Moteur SQLAlchemy créé avec succès.")
#         return engine
#     except Exception as err:
#         st.error(f"Erreur lors de la création du moteur SQLAlchemy : {err}")
#         return None


# # Fonction pour obtenir les noms des tables dans la base de données
# def get_table_names(engine):
#     inspector = inspect(engine)
#     return inspector.get_table_names()

# # Fonction pour obtenir les noms des colonnes dans une table
# def get_column_names(engine, table_name):
#     inspector = inspect(engine)
#     columns = inspector.get_columns(table_name)
#     return [column['name'] for column in columns]

# # Fonction pour ajuster la requête SQL pour la syntaxe SQL Server
# def adjust_query_for_sql_server(query):
#     if "LIMIT" in query:
#         limit_value = query.split("LIMIT")[1].strip().replace(";", "")
#         query = query.split("LIMIT")[0] + f" TOP {limit_value}"
#     return query

# # Fonction pour tronquer l'historique des conversations
# def truncate_chat_history(chat_history, max_length=5):
#     return chat_history[-max_length:]

# import os
# from sqlalchemy import create_engine

# def get_response(user_query: str, engine, chat_history: list):
#     truncated_chat_history = truncate_chat_history(chat_history)
    
#     # Requêtes SQL spécifiques basées sur la question de l'utilisateur
#     if "nom et prix des médicaments" in user_query.lower():
#         sql_query = """
#             SELECT name, prix
#             FROM medicaments
#         """
#     elif "pharmacies en garde" in user_query.lower():
#         sql_query = """
#             SELECT nom, phone_number, region, zone
#             FROM pharmacies
#             WHERE garde = TRUE
#         """
#     elif "catégories des médecins" in user_query.lower():
#         sql_query = """
#             SELECT name
#             FROM categories
#         """
#     else:
#         # Modèle général pour générer des requêtes SQL basées sur la question
#         sql_template = """
#             Vous êtes un analyste de données dans une entreprise de pharmacie. Vous interagissez avec un utilisateur qui pose des questions sur la base de données de la pharmacie.
#             Basé sur le schéma de la table ci-dessous, écrivez une requête SQL qui répondrait à la question de l'utilisateur. Prenez en compte l'historique de la conversation.
            
#             <SCHEMA>{schema}</SCHEMA>
            
#             Historique de la conversation : {chat_history}
            
#             Écrivez uniquement la requête SQL et rien d'autre. Ne pas encapsuler la requête SQL dans un autre texte, même pas des backticks.
            
#             Question : {question}
#             Requête SQL :
#         """
        
#         sql_prompt = ChatPromptTemplate.from_template(sql_template)
        
#         # Utilisation de l'API Groq pour générer les requêtes SQL
#         groq_api_key = os.getenv("GROQ_API_KEY")
        
#         llm_sql = ChatGroq(
#             model="mixtral-8x7b-32768",
#             temperature=0,
#             groq_api_key=groq_api_key
#         )
        
#         def get_schema(_):
#             tables = get_table_names(engine)
#             schema = ""
#             for table in tables:
#                 columns = get_column_names(engine, table)
#                 schema += f"{table}: {', '.join(columns)}\n"
#             return schema
        
#         # Chaine pour générer les requêtes SQL via le LLM
#         sql_chain = (
#             RunnableMap({
#                 "schema": get_schema,
#                 "chat_history": RunnablePassthrough(),
#                 "question": RunnablePassthrough()
#             })
#             | sql_prompt
#             | llm_sql
#             | StrOutputParser()
#         )
        
#         sql_query = sql_chain.invoke({
#             "question": user_query,
#             "chat_history": truncated_chat_history,
#         })
    
#     # Exécuter la requête SQL générée et retourner les résultats
#     result = engine.execute(sql_query).fetchall()
#     return result


    
#     # Nom de la base de données
#     db_name = "test"  # Remplacez ceci par le nom de la base de données dynamique si disponible
    
#     # Assurer le format correct de la requête SQL
#     if 'FROM ' in sql_query:
#         parts = sql_query.split('FROM ')
#         table_part = parts[1].split(' ')[0].strip()
#         condition_part = parts[1].split('WHERE ')[-1] if 'WHERE' in parts[1] else ''
        
#         sql_query = f"SELECT * FROM {table_part} WHERE {condition_part}" if condition_part else f"SELECT * FROM {table_part}"
    
#     # Exécution de la requête
#     try:
#         Session = sessionmaker(bind=engine)
#         session = Session()
#         result = session.execute(text(sql_query))
#         response = result.fetchall()
#         session.close()
        
#         # Convertir le résultat en DataFrame pour un meilleur affichage
#         if result.returns_rows:
#             df = pd.DataFrame(response, columns=result.keys())
#             response_str = df.to_string(index=False)
            
#             # Affichage du tableau des résultats
#             st.write(df)
            
#             # Affichage du graphique si plus de 2 colonnes sont sélectionnées
#             if df.shape[1] == 2:
#                 x_axis = df.columns[0]  # Utilisation de la première colonne comme axe X
#                 y_axis = df.columns[1]  # Utilisation de la deuxième colonne comme axe Y
#                 fig = px.line(df, x=x_axis, y=y_axis, title=f"Graphique de {y_axis} en fonction de {x_axis}")
#                 st.plotly_chart(fig)
            
#             # Ajout du graphique circulaire si 3 colonnes ou plus
#             if df.shape[1] >= 3:
#                 fig_pie = px.pie(df, values=df.columns[2], names=df.columns[0], title="Répartition")
#                 st.plotly_chart(fig_pie)
                
#         else:
#             response_str = "La requête n'a retourné aucun résultat."
#             st.write(response_str)
#     except Exception as e:
#         response_str = f"Erreur lors de l'exécution de la requête SQL : {e}"
#         st.write(response_str)
    
#     # Affichage de la requête SQL exécutée
#     st.info(f"Requête SQL exécutée :\n{sql_query}")
    
#     response_template = """
#         Vous êtes un analyste de données dans une entreprise. Vous interagissez avec un utilisateur qui pose des questions sur la base de données de l'entreprise.
#         Vous avez exécuté la requête suivante :
        
#         {sql_query}
        
#         Résultats :
        
#         {response}
#     """
    
#     return response_template.format(
#         sql_query=sql_query,
#         response=response_str
#     )



# # Interface utilisateur avec Streamlit
# st.title("Chat avec la base de données SQL Server")

# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []

# if 'show_db_form' not in st.session_state:
#     st.session_state.show_db_form = False

# # Icône pour afficher/cacher le formulaire de connexion
# if st.sidebar.button("🔌 Connexion à la DB"):
#     st.session_state.show_db_form = not st.session_state.show_db_form

# # Formulaire de connexion à la base de données
# if st.session_state.show_db_form:
#     with st.sidebar.form(key='connexion_form'):
#         user = st.text_input("Nom d'utilisateur")
#         password = st.text_input("Mot de passe", type="password")
#         host = st.text_input("Hôte")
#         port = st.text_input("Port")
#         database = st.text_input("Base de données")
#         submit_button = st.form_submit_button(label='Se connecter')
    
#         if submit_button:
#             conn = init_database(user, password, host, port, database)
#             if conn:
#                 st.session_state.db = create_sqlalchemy_engine(user, password, host, port, database)

# if st.session_state.get('db'):
#     st.sidebar.title("Options")
#     if st.sidebar.button("Se déconnecter"):
#         st.session_state.pop('db', None)
#         st.session_state.pop('chat_history', None)
#         st.success("Déconnexion réussie.")
    
#     user_query = st.text_input("Entrez votre requête SQL")
    
#     if st.button("Exécuter la requête"):
#         response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
#         st.session_state.chat_history.append(HumanMessage(content=user_query))
#         st.session_state.chat_history.append(AIMessage(content=response))
    
#     if st.sidebar.button("Réinitialiser l'historique des conversations"):
#         st.session_state.chat_history = []
#         st.success("Historique des conversations réinitialisé.")
    
#     st.sidebar.header("Structure de la base de données")
#     tables = get_table_names(st.session_state.db)
#     for table in tables:
#         st.sidebar.subheader(table)
#         columns = get_column_names(st.session_state.db, table)
#         st.sidebar.write(", ".join(columns))


# # import spacy
# # from transformers import pipeline
# # import streamlit as st
# # from sqlalchemy import create_engine, text, inspect
# # from sqlalchemy.orm import sessionmaker
# # import pandas as pd
# # import plotly.express as px
# # from langchain_core.messages import AIMessage, HumanMessage
# # from langchain_core.prompts import ChatPromptTemplate
# # from langchain_core.runnables import RunnablePassthrough, RunnableMap
# # from langchain_core.output_parsers import StrOutputParser
# # from langchain_groq import ChatGroq
# # from dotenv import load_dotenv
# # import os
# # import pyodbc


# # # Charger les variables d'environnement
# # load_dotenv()

# # # Configuration de la connexion à la base de données
# # DATABASE_URL = os.getenv('DATABASE_URL', 'mssql+pyodbc://user:password@server/database?driver=ODBC+Driver+17+for+SQL+Server')
# # engine = create_engine(DATABASE_URL)

# # # Charger un modèle NLP pré-entraîné
# # nlp_pipeline = pipeline('question-answering')

# # def analyser_question_avec_nlp(question, context):
# #     result = nlp_pipeline(question=question, context=context)
# #     return result


# # def fetch_initial_data(engine):
# #     query = """
# #     SELECT 
# #         NumDoss, client.NumCha, Services.des_service AS ServiceHospitalisation, EtaCli, 
# #         RTRIM(REPLACE(NomCli + ' ' + Prenom, '  ', ' ')) AS NomPatient, DatNai, 
# #         Nationnalite.libnat AS Nationalite, NumCIN, NumTel, AdrCli, 
# #         motif.Des AS MotifAdmission, natadm.Des AS NatureAdmission, Diagnost,
# #         CASE client.NumSoc 
# #             WHEN '' THEN 'PAYANT' 
# #             ELSE client.NumSoc 
# #         END AS CodePEC, 
# #         ISNULL(societe.dessoc, 'PAYANT') AS SocietePEC, TypReg, client.Plafond, 
# #         Medecin.nommed AS MedecinTraitant, MedSpec, NomPac, AdrPac, TelPac, 
# #         NomEng, CINEng, TelEng, client.Observ, DatArr, HeuArr, DatDep, HeuDep, 
# #         NumFac, DatFac, TypSortie.dessortie AS TypeSortie, MntCli AS TotFactureHT, 
# #         MntHo AS TotalHonoraire, MntCliPEC AS TotPECHT, MntHoPEC AS TotHonorairePEC, 
# #         client.TVAPEC, MntTva AS TotTVAFacture, MntRem AS TotRemise,
# #         MntCli + MntHo + MntTva AS TotFactureTTC, Payer, datepay, ModReg,
# #         TypArr.DesTyp AS TypeArrive, Avance, NumInt, MedRad, MedChir,
# #         ISNULL(T1.nommed, '') AS MedecinCorrespondant, EtabOrg, Kiné, Ergo, Ortho, 
# #         client.Timbre, Avoir, RefPEC, DatePEC, NumVir, MntRecu, client.TimbrePEC, 
# #         UserCre, UserFac, RetenuPEC, Eta_Recouv, DatCIN, DatCinEng, Datf_Plaf, 
# #         Profession, CodCat, TypPCE, Matricule, typconv, pere, RemHO, MntAvoir, 
# #         DatAvoir, NumPec, Identifiant, DatDConv, DatFConv, DateEnt, Lieu, 
# #         client.Resident, Autoris, DatAutoris, HeuAutoris, PER_PEC, Rem_autoris, 
# #         User_autoris, CodBurReg, AnPriseCh, NumOrdPriseCh, NumBordCNAM, Duplication, 
# #         OrgEmpl, Archive, DatRecep, Classer, NumCarte, Memo, P_Plafond, Cha_Bloc, 
# #         num_rdv, sex, audit, Date_audit, Heure_audit, User_audit, DatBordCnam, 
# #         Copie_Pas, client.usermodif, Pr_Plafond, Libelle_Avance, MatPers, CodPat, 
# #         client.HOPATPEC, Libelle_Appurement, Etat_civil, Bord_PER_PEC, has_piece_joint, 
# #         Epous, EpousVeuve, codMedRecommande, Nature_Heberg, delegation, motif_urgence, 
# #         CliniqueCorr, CodeCliniqueCorr, patientAdmisPMA, Port_Taxation, Gouvernorat, 
# #         CodePostale, Tel2, AdresseLocale, LIEN_PER_PEC, NATURE_PER_PEC, Code_Reservation, 
# #         MedUrg, PersAContacter, TElPersAContacter, AdrPersAContacter, 
# #         client.Intervention_Bloc, photo, avoirPhoto, num_sous_soc, date_DebCarte, 
# #         date_FinCarte, codAdherent, Pays, Num_Bordereau, NumConv, UserRecep, 
# #         NomCliAr, PrenomAr, Prenom2Ar, EpousVeuveAr, EpousAr, pereAr, userAutorisModifAv, 
# #         NumCheque, client.email, Accompagnant2, NomPac2, AdrPac2, TelPac2, 
# #         LienPac2, ANES, Oeuil, EXCEPTION, LienPac, PEC_Non_Parvenue, 
# #         Date_Autois_per, Heure_Autois_per, Code_Region, TypAjusRadio, 
# #         Code_Prest_Cnam, date_env, Num_Carte, Accompagnant, VIP, 
# #         Code_Med_Charge, Code_Emp_Charge, VLD_EXCEPTION, ancienID, 
# #         DatSortiePrevue, lienPersAContacter, numDemandeBloc, numSoc2, 
# #         Date_Dep_Prevu, Heure_Dep_Prevu, Autorise_Per, Recette, code_TypePrest, 
# #         Sequence_OPD, UserInstance, num_cabinet, ModeConsultation, TypAjusPayant, 
# #         TypAjusOrganisme, Nbre_seanceReeducation, Prix_unitaire, 
# #         MntPatient_AlaCharge, Vld_Contentieux, Eta_Recouv_Patient, 
# #         NumBord_Transf_Cont, Etat_Facture, A_recep_par, 
# #         NumBord_Transf_Cont_Pat, ImprimeBS, plafond_PER_PEC, CoursDollar, 
# #         CoursEuro, verseEsp, ObservationNutrition, CodePrestation, Date_Acte, 
# #         Date_Dece, Heure_Dece, Medecin_Dece, service_Dece, NumDevis, 
# #         Renseignement, Num_CNAM_Recouv, client.date_depot, 
# #         Vld_Contentieux_patient, Etat_Cont_Patient, Etat_Cont_PEC, 
# #         Cont_recep_Patient, Cont_recep_PEC, client.NomArb, PrenomArb, 
# #         client.Per_PEC_Personnel, client.Nature_Per_PEC_Personnel, 
# #         client.Numadmission, client.Comute, client.MedTrait2, 
# #         client.autorisConsultDMIcentral, client.CINPac, client.DocManquant, 
# #         newIdent, client.NumSocMutuelle 
# #     FROM client 
# #     LEFT OUTER JOIN Medecin ON Client.MedTrait = Medecin.CodMed
# #     LEFT OUTER JOIN medecin T1 ON client.MedCorr = T1.CodMed
# #     INNER JOIN chambre ON Client.NumCha = Chambre.NumCha
# #     INNER JOIN motif ON Client.TypAdm = Motif.Cod
# #     INNER JOIN Nationnalite ON client.nation = Nationnalite.codnat
# #     INNER JOIN natadm ON client.natadm = natadm.cod
# #     LEFT OUTER JOIN societe ON client.numsoc = societe.numsoc
# #     LEFT OUTER JOIN TypSortie ON client.TypSortie = TypSortie.CodSortie
# #     INNER JOIN TypArr ON client.TypArr = TypArr.codtyp
# #     INNER JOIN Services ON Chambre.NumSer = Services.Num_Service
# #     """

# #     try:
# #         # Utiliser pd.read_sql_query pour exécuter la requête et obtenir les résultats en DataFrame
# #         df = pd.read_sql_query(query, engine)
# #         return df
# #     except Exception as e:
# #         st.error(f"Erreur lors de l'exécution de la requête SQL initiale : {e}")
# #         return None

# # # Exemple d'une fonction pour détecter l'intention de l'utilisateur
# # def detect_intent(user_query: str):
# #     # Utilisation d'un modèle NLP plus moderne pour la détection d'intention
# #     nlp_pipeline = pipeline("zero-shot-classification")
# #     candidate_labels = ["compter", "somme", "filtrer"]
# #     result = nlp_pipeline(user_query, candidate_labels)
# #     return result["labels"][0]
    
# # # Fonction pour analyser la requête et en extraire les informations importantes
# # def analyze_user_query(user_query: str):
# #     # Amélioration de l'analyse avec le modèle de transformateurs
# #     nlp_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
# #     entities = nlp_pipeline(user_query)
    
# #     # Détection simplifiée pour le filtrage
# #     if "service" in user_query.lower():
# #         return "Filtrer par service"
# #     elif "montant" in user_query.lower():
# #         return "Filtrer par montant"
# #     else:
# #         return "Aucune action spécifiée"

# # def generate_chart(df, selected_column):
# #     if selected_column not in df.columns:
# #         st.error(f"Colonne sélectionnée {selected_column} non trouvée dans les données.")
# #         return None
    
# #     fig = px.histogram(df, x=selected_column)
# #     st.plotly_chart(fig)

# # # Fonction pour initialiser la connexion à SQL Server et récupérer les données
# # def init_database_sql_server(user, password, host, port, database):
# #     try:
# #         engine = create_sqlalchemy_engine_sql_server(user, password, host, port, database)
# #         return engine
# #     except Exception as e:
# #         st.error(f"Erreur lors de la connexion à la base de données : {e}")
# #         return None


# # Fonction pour initialiser la connexion à la base de données
# def init_database(user: str, password: str, host: str, port: str, database: str):
#     try:
#         conn = mysql.connector.connect(
#             host=host,
#             port=port,
#             user=user,
#             password=password,
#             database=database
#         )
#         st.success("Connexion réussie à la base de données MySQL.")
#         return conn
#     except mysql.connector.Error as err:
#         st.error(f"Erreur lors de la connexion à la base de données : {err}")
#         return None
# # Fonction pour créer le moteur SQLAlchemy
# def create_sqlalchemy_engine(user, password, host, port, database):
#     try:
#         conn_str = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
#         engine = create_engine(conn_str)
#         st.success("Moteur SQLAlchemy créé avec succès.")
#         return engine
#     except Exception as err:
#         st.error(f"Erreur lors de la création du moteur SQLAlchemy : {err}")
#         return None


# # def create_sqlalchemy_engine_sql_server(user, password, host, port, database):
# #     connection_string = (
# #         f"mssql+pyodbc://{user}:{password}@{host}:{port}/{database}"
# #         f"?driver=ODBC+Driver+17+for+SQL+Server"
# #     )
# #     return create_engine(connection_string)


# # # Fonction pour obtenir les noms des tables dans la base de données
# # def get_table_names(engine):
# #     query = """
# #     SELECT [TABLE_NAME]
# #     FROM [INFORMATION_SCHEMA].[TABLES]
# #     WHERE [TABLE_SCHEMA] = :schema AND [TABLE_TYPE] = :type
# #     ORDER BY [TABLE_NAME]
# #     """
    
# #     # Obtenez une connexion à partir de l'engine
# #     with engine.connect() as connection:
# #         result = connection.execute(text(query), {'schema': 'dbo', 'type': 'BASE TABLE'})
# #         return [row[0] for row in result]



# # # Fonction pour obtenir les noms des colonnes dans une table
# # def get_column_names(engine, table_name):
# #     inspector = inspect(engine)
# #     columns = inspector.get_columns(table_name)
# #     return [column['name'] for column in columns]

# # # Fonction pour tronquer l'historique des conversations
# # def truncate_chat_history(chat_history, max_length=5):
# #     return chat_history[-max_length:]

# # def get_response(user_query: str, df: pd.DataFrame, chat_history: list):
# #     analysis = analyze_user_query(user_query)
# #     intention = analysis["intention"]

# #     if intention == "compter":
# #         result_df = df.groupby(['ServiceHospitalisation'])['NumDoss'].nunique().reset_index(name='nombre_d_admissions')

# #     elif intention == "somme":
# #         result_df = df.groupby(['ServiceHospitalisation'])[analysis.get("entité", 'TotFactureTTC')].sum().reset_index(name='TotalFacture')
    
# #     elif intention == "filtrer":
# #         # Filtrer par service, nationalité ou médecin
# #         if "service" in analysis:
# #             service = analysis['service'][0] if analysis['service'] else None
# #             result_df = df[df['ServiceHospitalisation'] == service]
# #         elif "nationalité" in analysis:
# #             nationalite = analysis['nationalité'][0] if analysis['nationalité'] else None
# #             result_df = df[df['Nationalite'] == nationalite]
# #         elif "entité" in analysis:
# #             # Filtrer par médecin traitant ou autre entité
# #             entité = analysis['entité']
# #             result_df = df[df['MedecinTraitant'] == entité]
# #         else:
# #             result_df = df
# #     else:
# #         result_df = pd.DataFrame()  # Aucun résultat si l'intention est inconnue

# #     # Affichage des résultats
# #     if not result_df.empty:
# #         st.dataframe(result_df)
# #         response_str = result_df.to_string(index=False)

# #         # Affichage graphique
# #         if result_df.shape[1] == 2:
# #             x_axis = result_df.columns[0]
# #             y_axis = result_df.columns[1]
# #             fig = px.line(result_df, x=x_axis, y=y_axis, title=f"Graphique de {y_axis} en fonction de {x_axis}")
# #             st.plotly_chart(fig)
# #         elif result_df.shape[1] >= 3:
# #             fig_pie = px.pie(result_df, values=result_df.columns[2], names=result_df.columns[0], title="Répartition")
# #             st.plotly_chart(fig_pie)
# #     else:
# #         st.warning("Aucun résultat correspondant à votre requête.")

# #     return result_df

# # # def analyze_user_query(user_query: str):
# # #     doc = nlp(user_query.lower())
    
# # #     entities = [(ent.text, ent.label_) for ent in doc.ents]

# # #     # Amélioration de l'analyse pour détecter les médecins
# # #     if "médecin" in user_query.lower() or "docteur" in user_query.lower():
# # #         return {"intention": "filtrer", "entité": "MedecinTraitant"}
    
# # #     # Détection de la nationalité
# # #     if "nationalité" in user_query.lower():
# # #         nationalite = [token.text for token in doc if token.ent_type_ == "NORP"]
# # #         return {"intention": "filtrer", "nationalité": nationalite}
    
# # #     # Détection du montant total
# # #     if "montant" in user_query.lower() or "total" in user_query.lower():
# # #         return {"intention": "somme", "entité": "TotFactureTTC"}

# # #     return {"intention": "inconnu", "détails": entities}

# # # # Interface Streamlit pour l'utilisateur
# # # def main():
# # #     st.title("Système de gestion des données hospitalières")
    
# # #     # Charger les données
# # #     df = fetch_initial_data(engine)
    
# # #     if df is not None:
# # #         # Requête utilisateur
# # #         user_query = st.text_input("Posez votre question :", "")
        
# # #         if st.button("Rechercher"):
# # #             # Historique simplifié
# # #             chat_history = [user_query]
# # #             response = get_response(user_query, df, chat_history)
# # #             st.write(response)
# # #     else:
# # #         st.error("Impossible de charger les données.")

# # # if __name__ == "__main__":
# # #     main()
# # # # Interface utilisateur avec Streamlit
# # # st.title("Chat avec la base de données SQL Server")

# # # if 'chat_history' not in st.session_state:
# # #     st.session_state.chat_history = []

# # # if 'show_db_form' not in st.session_state:
# # #     st.session_state.show_db_form = False

# # # # Icône pour afficher/cacher le formulaire de connexion
# # # if st.sidebar.button("🔌 Connexion à la DB"):
# # #     st.session_state.show_db_form = not st.session_state.show_db_form

# # # # Formulaire de connexion à la base de données SQL Server
# # # if st.session_state.show_db_form:
# # #     with st.sidebar.form(key='connexion_form'):
# # #         user = st.text_input("Nom d'utilisateur")
# # #         password = st.text_input("Mot de passe", type="password")
# # #         host = st.text_input("Hôte")
# # #         port = st.text_input("Port")
# # #         database = st.text_input("Base de données")
# # #         submit_button = st.form_submit_button(label='Se connecter')
    
# # #         if submit_button:
# # #             conn = init_database_sql_server(user, password, host, port, database)
# # #             if conn:
# # #                 st.session_state.db = create_sqlalchemy_engine_sql_server(user, password, host, port, database)
# # #                 # Exécuter la requête SQL initiale et stocker les résultats
# # #                 st.session_state.initial_data = fetch_initial_data(st.session_state.db)
# # #                 if st.session_state.initial_data is not None:
# # #                     st.success("Données initiales chargées avec succès.")


# # # if st.session_state.get('db'):
# # #     st.sidebar.title("Options")
# # #     if st.sidebar.button("Se déconnecter"):
# # #         st.session_state.pop('db', None)
# # #         st.session_state.pop('chat_history', None)
# # #         st.success("Déconnexion réussie.")
    
# # #     user_query = st.text_input("Entrez votre requête SQL")
    
# # #     if st.button("Exécuter la requête"):
# # #         response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
# # #         st.session_state.chat_history.append(HumanMessage(content=user_query))
# # #         st.session_state.chat_history.append(AIMessage(content=response))
    
# # #     # if st.sidebar.button("Réinitialiser l'historique des conversations"):
# # #     #     st.session_state.chat_history = []
# # #     #     st.success("Historique des conversations réinitialisé.")
    
# # #     # st.sidebar.header("Structure de la base de données")
# # #     # tables = get_table_names(st.session_state.db)
# # #     # for table in tables:
# # #     #     st.sidebar.subheader(table)
# # #     #     columns = get_column_names(st.session_state.db, table)
# # #     #     st.sidebar.write(", ".join(columns))

# # #     # # Afficher l'historique des conversations
# # #     # st.header("Historique des conversations")
# # #     # for message in st.session_state.chat_history:
# # #     #     if isinstance(message, HumanMessage):
# # #     #         st.write(f"**Utilisateur :** {message.content}")
# # #     #     else:
# # #     #         st.write(f"**Assistant :** {message.content}")
# # # else:
# # #     st.warning("Veuillez vous connecter à la base de données pour exécuter des requêtes.")
# # # Interface utilisateur avec Streamlit
# # def main():
# #     st.title("Chat avec la base de données SQL Server")

# #     if 'df' not in st.session_state:
# #         st.session_state.df = None

# #     if 'chat_history' not in st.session_state:
# #         st.session_state.chat_history = []

# #     if 'show_db_form' not in st.session_state:
# #         st.session_state.show_db_form = False

# #     # Icône pour afficher/cacher le formulaire de connexion
# #     if st.sidebar.button("🔌 Connexion à la DB"):
# #         st.session_state.show_db_form = not st.session_state.show_db_form

# #     # Formulaire de connexion à la base de données SQL Server
# #     if st.session_state.show_db_form:
# #         with st.sidebar.form(key='connexion_form'):
# #             user = st.text_input("Nom d'utilisateur")
# #             password = st.text_input("Mot de passe", type="password")
# #             host = st.text_input("Hôte")
# #             port = st.text_input("Port")
# #             database = st.text_input("Base de données")
# #             submit_button = st.form_submit_button(label='Se connecter')
        
# #             if submit_button:
# #                 engine = init_database_sql_server(user, password, host, port, database)
# #                 if engine:
# #                     st.session_state.df = fetch_initial_data(engine)
# #                     if st.session_state.df is not None:
# #                         st.success("Données initiales chargées avec succès.")
# #                     else:
# #                         st.error("Impossible de charger les données.")
    
# #     if st.session_state.df is not None:
# #         st.sidebar.title("Options")
# #         if st.sidebar.button("Se déconnecter"):
# #             st.session_state.pop('df', None)
# #             st.session_state.pop('chat_history', None)
# #             st.success("Déconnexion réussie.")
# #        # Filtrage basé sur l'intention de l'utilisateur
# #         user_query = st.text_input("Entrez votre question :")
# #         intent = detect_intent(user_query)
# #         st.write(f"Intention détectée : {intent}")
        
# #         if st.button("Exécuter la requête"):
# #             response = get_response(user_query, st.session_state.df, st.session_state.chat_history)
# #             st.session_state.chat_history.append({"role": "user", "content": user_query})
# #             st.session_state.chat_history.append({"role": "assistant", "content": response})
        
# #         # Afficher l'historique des conversations
# #         st.header("Historique des conversations")
# #         for message in st.session_state.chat_history:
# #             if message["role"] == "user":
# #                 st.write(f"**Utilisateur :** {message['content']}")
# #             else:
# #                 st.write(f"**Assistant :** {message['content']}")
# #     else:
# #         st.warning("Veuillez vous connecter à la base de données pour exécuter des requêtes.")

# # if __name__ == "__main__":
# #     main()









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





# import streamlit as st
# from sqlalchemy import create_engine, text, inspect
# from sqlalchemy.orm import sessionmaker
# import pandas as pd
# import plotly.express as px
# from langchain_core.messages import AIMessage, HumanMessage
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough, RunnableMap
# from langchain_core.output_parsers import StrOutputParser
# from langchain_groq import ChatGroq
# from dotenv import load_dotenv
# import os
# import mysql.connector
# from transformers import pipeline
# import pandas as pd
# from sqlalchemy import text

# # Charger les variables d'environnement
# load_dotenv()

# # Fonction pour initialiser la connexion à la base de données
# def init_database(user: str, password: str, host: str, port: str, database: str):
#     try:
#         conn = mysql.connector.connect(
#             host=host,
#             port=port,
#             user=user,
#             password=password,
#             database=database
#         )
#         st.success("Connexion réussie à la base de données MySQL.")
#         return conn
#     except mysql.connector.Error as err:
#         st.error(f"Erreur lors de la connexion à la base de données : {err}")
#         return None

# # Fonction pour créer le moteur SQLAlchemy
# def create_sqlalchemy_engine(user, password, host, port, database):
#     try:
#         conn_str = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
#         engine = create_engine(conn_str)
#         st.success("Moteur SQLAlchemy créé avec succès.")
#         return engine
#     except Exception as err:
#         st.error(f"Erreur lors de la création du moteur SQLAlchemy : {err}")
#         return None

# # Fonction pour obtenir les noms des tables
# def get_table_names(engine):
#     inspector = inspect(engine)
#     return inspector.get_table_names()

# # Fonction pour obtenir les noms des colonnes dans une table
# def get_column_names(engine, table_name):
#     inspector = inspect(engine)
#     columns = inspector.get_columns(table_name)
#     return [column['name'] for column in columns]

# # Fonction pour tronquer l'historique des conversations
# def truncate_chat_history(chat_history, max_length=5):
#     return chat_history[-max_length:]

# # Exemple de schéma structuré pour comprendre le besoin de l'utilisateur
# def get_user_input():
#     # Demander à l'utilisateur de remplir un formulaire
#     st.write("### Formulaire de demande")
    
#     emotion = st.selectbox("Comment vous sentez-vous aujourd'hui ?", ["Heureux", "Triste", "Stressé", "Neutre"])
#     specific_concern = st.text_input("Y a-t-il quelque chose de spécifique qui vous inquiète ?")
#     problem_type = st.selectbox("Quel type de problème souhaitez-vous traiter ?", ["Problème émotionnel", "Problème physique", "Problème médical général"])
    
#     return {
#         "emotion": emotion,
#         "specific_concern": specific_concern,
#         "problem_type": problem_type
#     }
# # Exemple de mapping des champs lexicaux aux sentiments

# # Fonction pour améliorer la compréhension du besoin
# def get_response(user_query: str, engine, chat_history: list):
#     if 'db_engine' not in st.session_state or st.session_state.db_engine is None:
#         st.error("La connexion à la base de données n'est pas établie.")
#         return None
    
#     user_input = get_user_input()
#     emotion = user_input["emotion"]
#     specific_concern = user_input["specific_concern"]
#     problem_type = user_input["problem_type"]
    
#     emotion_query = """
#         SELECT emotion FROM feeling 
#         WHERE emotion = :emotion
#     """
#     doctor_query = """
#         SELECT d.id, d.name, d.speciality, d.sponsor, d.experience
#         FROM doctor d
#         JOIN feeling f ON f.iddoctor = d.id
#         WHERE f.emotion = :emotion
#         ORDER BY d.sponsor DESC, d.experience DESC
#     """
    
#     try:
#         with engine.connect() as connection:
#             # Vérification de l'émotion dans la base de données
#             result = connection.execute(text(emotion_query), {"emotion": emotion})
#             db_emotion = result.scalar()

#             if not db_emotion:
#                 follow_up = follow_up_question(chat_history)
#                 chat_history.append(AIMessage(content=follow_up))
#                 return follow_up

#             # Recherche des médecins correspondant à l'émotion
#             doctors_result = connection.execute(text(doctor_query), {"emotion": db_emotion})
#             doctors = doctors_result.fetchall()

#             if not doctors:
#                 return f"Aucun médecin trouvé pour traiter l'émotion '{emotion}'."

#             # Information supplémentaire basée sur le type de problème
#             if problem_type == "Problème émotionnel":
#                 additional_info = "Nous vous recommandons de consulter un spécialiste en santé mentale."
#             elif problem_type == "Problème physique":
#                 additional_info = "Vous pouvez consulter un médecin généraliste pour des conseils médicaux."
#             else:
#                 additional_info = "Nous vous suggérons de prendre rendez-vous avec un spécialiste."

#             # Conversion des résultats en DataFrame pour affichage
#             column_names = doctors_result.keys()
#             df = pd.DataFrame(doctors, columns=column_names)
            
#             # Retourner les résultats et recommandations
#             response = {
#                 "doctors": df.to_dict(orient="records"),
#                 "recommendation": additional_info
#             }
#             return response
#     except Exception as e:
#         st.error(f"Erreur lors de l'exécution de la requête SQL : {e}")
#         return None

# def follow_up_question(chat_history):
#     predefined_questions = [
#         "Pouvez-vous en dire plus sur ce que vous ressentez ?",
#         "Est-ce que vous êtes triste, stressé ou heureux aujourd'hui ?",
#         "Quelque chose de spécifique vous préoccupe-t-il ?"
#     ]
#     return predefined_questions[len(chat_history) % len(predefined_questions)]



# def follow_up_question(chat_history):
#     predefined_questions = [
#         "Pouvez-vous en dire plus sur ce que vous ressentez ?",
#         "Est-ce que vous êtes triste, stressé ou heureux aujourd'hui ?",
#         "Quelque chose de spécifique vous préoccupe-t-il ?"
#     ]
#     # Retourner une question du pool, en fonction de la longueur de l'historique
#     return predefined_questions[len(chat_history) % len(predefined_questions)]


# # Interface utilisateur avec Streamlit
# st.title("Chat avec la base de données SQL")

# # Initialisation des variables dans session_state si elles n'existent pas
# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []

# if 'db_engine' not in st.session_state:
#     st.session_state.db_engine = None  # Initialisation de la clé db_engine

# # Formulaire de connexion à la base de données
# with st.sidebar.form(key="connexion_form"):
#     user = st.text_input("Nom d'utilisateur")
#     password = st.text_input("Mot de passe", type="password")
#     host = st.text_input("Hôte")
#     port = st.text_input("Port", value="3306")
#     database = st.text_input("Base de données")
#     submit_button = st.form_submit_button(label="Se connecter")

# if submit_button:
#     # Tentative de connexion à la base de données
#     engine = init_database(user, password, host, port, database)
#     if engine:
#         st.session_state.db_engine = engine

# # Interface principale après connexion
# if st.session_state.db_engine:
#     st.success("Base de données connectée.")
#     user_query = st.text_input("Entrez votre requête SQL")
#     if st.button("Exécuter la requête"):
#         response = get_response(user_query, st.session_state.db_engine, st.session_state.chat_history)
#         st.session_state.chat_history.append({"user": user_query, "response": response})

#         if isinstance(response, pd.DataFrame):  # Résultats sous forme de tableau
#             st.write(response)
#             if len(response.columns) >= 2:
#                 fig = px.line(response, x=response.columns[0], y=response.columns[1])
#                 st.plotly_chart(fig)
#         else:  # Résultats sous forme de message
#             st.info(response)

# # Affichage de l'historique
# if st.session_state.chat_history:
#     st.write("### Historique des conversations")
#     for chat in st.session_state.chat_history:
#         st.text(f"Utilisateur : {chat['user']}")
#         st.text(f"Réponse : {chat['response']}")
