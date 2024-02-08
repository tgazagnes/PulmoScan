import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import altair as alt
import cv2

def exec_exp1_page3(self):
  
  

  st.markdown(
  """
  ##### :test_tube: **Hypothèse testée** : Les vecteurs de densité de probabilité de l'intensité des images masquées 1x256 seraient une information suffisantes pour entraîner un modèle de classification.""")
  
  col1, col2, col3, col4, col5 = st.columns([0.25,4,0.5,3,0.25])

  container = col2.container(border=True)
  container.markdown(
    """
      **Expérimentation:** Classification à deux classes Patient "Normal" vs "Malade" (atteint d'une pathologie pulmonaire Covid, pneumonie virale, opacité)""")
  
  col4.image(cv2.imread('src/ressources/Model_reduction.png'),caption = 'Meilleur Modèle entraîné',width=200)

  col2.markdown( """
  

  
""")
  confusion_matrix = pd.DataFrame({
      "" : ["Prédiction MALADE", "Prédiction NORMAL"],
      "MALADE réel" : ['VP = 764', 'FN = 43'],
      "NORMAL réel" : ['FP = 279', 'VN = 528']
    }   )

  expander = col2.expander("Détails de l'expérimentation")
  expander.write(
      """
      1. Equilibrage des données sur deux classes: 
        * Classe "Normal" (patient sain) : 
          * 1345 x 3 images provenant de Normal
        * Classe "Malade" présentant une pathologie pulmonaire: 
          * 1345 images provenant de COVID, 
          * 1345 provenant de 'Lung_Opacity', 
          * et 1345 provenant de 'Viral_Pneumonia'
      2. Transformation des images en vecteur 1x256
      3. Création d'un jeu d'entrainement 6456 images et d'un jeu de Test de 1614 images
      4. Evaluation d'un premier modèle de machine learning
      5. Evaluation d'un modèle de Deep Learning inspiré de LeNet
      6. Optimisation de l'architecture et des hyperparamètres avec Keras Tuner
      7. Différents entraînement en jouant sur la métrique évaluée : fonction de perte ou score de justesse      
      """)     

  col2.image(cv2.imread('src/ressources/reduite_train.png'),caption = 'Entraînement optimisé du meilleur modèle',width=600)
  
  col2.dataframe(data=confusion_matrix, use_container_width=True,hide_index=True)
  col2.markdown( """
       ##### Résultats : justesse atteinte de 80%, avec une sensibilité de 94,5%""")

  col2.divider()
  #Conclusions
  col2.markdown(
  """
  ##### :clipboard: **Conclusions :**
  * Cette approche présente des résultats intéressants avec l'avantage d'être très peu gourmande en ressources de calcul.
  * L'explicabilité est aussi facilitée puisque le modèle est entraîné sur une information plus simple.
  * Les performances atteintes sont toutefois inférieures aux modèles de deep learning appliqués directement sur les images.
  * Approfondissements possibles:  combinaison à d'autres modèles, évaluation de modèle de transfer learning adapté pour les signaux 1D, expérimentation de réseaux de neurones récurrents.
  """)

def exec_exp3_page3(self):

  st.markdown(
  """
  ##### :test_tube: **Hypothèse testée** : Une classification à 3 classes, sans "Lung Opacity", serait plus performante.""")

  
  
  expander = st.expander(":interrobang: **Observation:** La classe **autres pathologies (LUNG_OPACITY)** n'est pas homogène.")
  expander.markdown(
  """
  Au contraire des autres classes du jeu de données qui représentent des pathologies spécifiques, celle-ci contient des radiographies de **pathologies diverses**, allant la simple infection pulmonaire à l'oedeme pulmonaire. La nature et la répartition de ces autres pathologies n'est pas renseignée dans le jeu de données. """)
  
  container = st.container(border=True)
  container.markdown(
  """
   **Expérimentation:** Comparaisons des performances de classification d'un même modèle d'apprentissage par transfert basé sur ResNet152v2 sur des jeux d'images à 3 classes et 4 classes.»
""") 

  col1, col2, col3, col4, col5 = st.columns([0.25,2.8,2.8,2.8,0.25])

    # 1ère ligne dans un container : expérimentation 1
  container = col2.container(border=True)
  container.markdown(
    """
    **Etape :one: : Apprentissage sans dégel de couches**

    """)

  chart_data = pd.DataFrame({
      "Tests" : ["4 classes", "3 classes"],
      "Résultats" : [0.68, 0.80]
    }   )
      
  c = (alt.Chart(chart_data, height = 150)
      .encode(y= alt.Y("Tests", sort = None, title=""), 
              x = alt.X("Résultats", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='%')).title("Justesse sur jeu de validation"),
              text = alt.Text("Résultats", format = '.0%'),
              opacity=alt.value(0.8),
              color = alt.Color("Tests").legend(None)
             )
      )
  
  graph = c.mark_bar() + c.mark_text(align = "left", dx = 6, size = 14)

  col2.altair_chart(graph, use_container_width=True)
  col2.markdown(""" ##### + 12 pts de justesse""")


  container = col3.container(border=True)
  container.markdown(
    """
    **Etape :two: : Réapprentissage avec dégel de 21 couches**

    """)  
  
  chart_data = pd.DataFrame({
      "Tests" : ["4 classes", "3 classes"],
      "Résultats" : [0.69, 0.81]
    }   )
      
  c = (alt.Chart(chart_data, height = 150)
      .encode(y= alt.Y("Tests", sort = None, title=""), 
              x = alt.X("Résultats", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='%')).title("Justesse sur jeu de validation"),
              text = alt.Text("Résultats", format = '.0%'),
              opacity=alt.value(0.8),
              color = alt.Color("Tests").legend(None)
             )
      )
  graph = c.mark_bar() + c.mark_text(align = "left", dx = 6, size = 14)

  col3.altair_chart(graph, use_container_width=True)
  col3.markdown(""" ##### + 12 pts de justesse""")
 

  container = col4.container(border=True)
  container.markdown(
    """
    **Etape :three: : Réapprentissage avec dégel de 61 couches**
    """)  

  chart_data = pd.DataFrame({
      "Tests" : ["4 classes", "3 classes"],
      "Résultats" : [0.74, 0.87]
    }   )
      
  c = (alt.Chart(chart_data, height = 150)
      .encode(y= alt.Y("Tests", sort = None, title=""), 
              x = alt.X("Résultats", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='%')).title("Justesse sur jeu de validation"),
              text = alt.Text("Résultats", format = '.0%'),
              opacity=alt.value(0.8),
              color = alt.Color("Tests").legend(None)
             )
      )
  graph = c.mark_bar() + c.mark_text(align = "left", dx = 6, size = 14)

  col4.altair_chart(graph, use_container_width=True)
  col4.markdown(""" ##### + 13 pts de justesse""")
    
  st.divider()
  #Conclusions
  st.markdown(
  """
  ##### :clipboard: **Conclusions :**
  * La reduction du jeu de données à 3 classes améliore significativement la performance de classification.	
  * Raisons possibles (non démontrées) :  
    - **Forte hétérogénéité** des pathologies regroupées dans la classe Lung_Opacity
    - Réduction des biais de classification par le retrait de cette classe

  """
  )
 