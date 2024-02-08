import streamlit as st

def exec_page0(self):
  st.image("src/ressources/Logo_pulmoscanCapture.PNG")

  st.subheader('Détection du COVID-19 et autres maladies pulmonaires à partir de radiographies du thorax, grâce à l’apprentissage profond ("deep learning")')

  st.write("")


  st.markdown(
    """
    "PulmoScan" est un projet réalisé entre novembre 2023 et janvier 2024 dans le cadre du bootcamp Data Scientist, cohorte d'octobre 2023. Ce projet a été réalisé à titre d'apprentissage uniquement ; les résultats affichés n'ont pas été validés par protocole scientifique. Aucune utilisation externe ne doit en être faite.
    """)


  st.divider()

  st.markdown(
  """

  #### Contexte :health_worker: 
    - **Apparition du COVID-19** : tension forte sur les capacités de diagnostic et d'orientation dans un contexte de pénurie de médecins généralistes et spécialistes en France
  - Des techniques **d'imagerie médicale** éprouvées (Radiologie, Scanner, IRM...) mais qui reposent fortement sur l'interprétation humaine par un praticien qualifié
  - **Avantages de la radiologie vs. autres techniques** : facilité d'utilisation, plus faible coût, plus faible dose de rayonnement et meilleure accessibilité dans les hôpitaux publics
  - Avancées majeures dans la **reconnaissance d'image par apprentissage automatique** au cours des 20 dernières années
  ---
  #### Objectifs :dart: 
  1. Développer un modèle d’apprentissage automatique capable de **prédire de manière fiable un diagnostic de maladie pulmonaire à partir d'une radiographie du thorax**.
  2. Explorer les **perspectives et limites** offertes par l’apprentissage profond dans le domaine médical
  3. Pour l'équipe : mettre en application les acquis de la formation DataScientest 
   
 --- 
  ####  Données source :open_file_folder:
  - **[COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database/)** : banque contenant 21 165 images de radiographies du thorax, mise à disposition libre d'accès par une équipe de chercheurs de l'université du Qatar et de l'université de Dhaka (Bangladesh). 
  ---

  ####  Méthodologie :microscope: 
    """
  )
  st.image("src/ressources/Methodo.png", caption = "Schéma d'ensemble de la méthodologie employée pour le projet")
  st.write("")
  st.image("src/ressources/logos_outils.png", caption = "Outils utilisés")

  st.markdown(
  """
  ---
  #### Résultats obtenus :trophy: 
  - Classification juste à près de **90%** obtenue grâce à plusieurs modèles d'apprentissage par transfert
  - **Plus de 15 modèles** de deep learning expérimentés et comparés
  - Mise en oeuvre de techniques **d'interprétabilité** pour analyser les résultats
  - **5 approfondissements menés** pour tenter d'améliorer la performance
  - Analyse de la qualité du jeu de données
  ---
  """
  )

  st.markdown(
  """
  #### Equipe projet (cohorte octobre 2023) :people_holding_hands: 
  
  """
  )
  st.write("")


  col1, col2, col3, col4, col5  = st.columns([3,3, 3, 3, 6])

  with col1:
    st.image("src/ressources/photo_SC.jpg", width = 120, caption = "Steve Costalat")

  with col2:
    st.image("src/ressources/photo_TG.jpg", width = 120, caption = "Thibaut Gazagnes")
  
  with col3:
    st.image("src/ressources/photo_NG.jpg", width = 120, caption = "Nicolas Gorgol")

  with col4:
    st.image("src/ressources/photo_GP.jpg", width = 120, caption = "Mentor projet : Gaël Penessot")
