import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import altair as alt

def exec_exp2_page3(self):
  st.markdown(
  """
  ##### :test_tube: **Hypothèse testée** : l'augmentation de données permet d'améliorer la performance de classification en réduisant les biais liés au jeu d'entraînement.	

  """
  )
  st.write("")

  # 1ère ligne dans un container : expérimentation 1
  container = st.container(border=True)

  container.markdown(
    """
    **Expérimentation :one: : Diversification du jeu d’entraînement à taille égale**

    """)

  # 5 colonnes avec colonnes 2 et 4 utilisées, pour laisser un peu de marge sur les côtés et au centre 
  col1, col2, col3, col4, col5 = st.columns([0.25,4,0.5,4,0.25])

  # Explication de l'expérimentation
  with col2:
    st.write("")
    st.markdown(
      """
       **Transformations appliquées aux images** :
      > - Translation aléatoire +/-10% 
      > - Rotation aléatoire +/-10°
      > - Retournement horizontal aléatoire
      > - Zoom aléatoire +/-15%
      > - Variation d’intensité aléatoire +/-10%

             
      """)

  # Graphique avec les résultats comparés
  with col4:
    st.write("")
    st.write("**Performance comparée avec ResNetV2 :**")
    chart_data = pd.DataFrame({
      "Tests" : ["Sans transformations", "Avec tranformations"],
      "Résultats" : [0.82, 0.77]
    }   )
      
    c = (
      alt.Chart(chart_data, height = 150)
      .encode(y= alt.Y("Tests", sort = None, title = ""), 
              x = alt.X("Résultats", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='%')).title("Justesse sur jeu de validation"),
              text = alt.Text("Résultats", format = '.0%'),
              opacity=alt.value(0.8),
              color = alt.Color("Tests").legend(None)
              )
      )
    graph = c.mark_bar() + c.mark_text(align = "left", dx = 6, size = 14)

    st.altair_chart(graph, use_container_width=True)

  #En dessous : un champ déroulant pour en savoir plus
  #expander = st.expander("Détails de l'expérimentation")
  #expander.write(
    


  st.divider()

# 2è ligne dans un container : expérimentation 2
  container2 = st.container(border=True)

  container2.markdown(
    """
    **Expérimentation :two: : Augmentation du nombre d’images du jeu d’entraînement (x 2,5)**

    """)
  
  # 5 colonnes avec colonnes 2 et 4 utilisées, pour laisser un peu de marge sur les côtés et au centre 
  col1, col2, col3, col4, col5 = st.columns([0.25,4,0.5,4,0.25])

  # Explication de l'expérimentation
  with col2:
    st.write("")
    st.markdown(
  """
    **Augmentation du nombre d'images dans le jeu d'entraînement** :
        
  """)
    #créer le dataframe
    data_nbimg = pd.DataFrame({
      "Classes" : ["Normal", "Lung Opacity", "COVID", "Viral Pneumonia", "Total"], 
      "Données source" : [10192, 6012, 3616, 1345, 21165],
      "Jeu d'images préprocessé" : [1345, 1345, 1345, 1345, 5380],
      "Jeu d'images augmenté" : [3616, 3616, 3616, 1345, 12193]
    }
    )
    st.dataframe(data = data_nbimg, use_container_width = True, hide_index = True)


  # Graphique avec les résultats comparés    
  with col4:
    st.write("")
    st.write("**Performance comparée avec VGG19 :**")

    chart_data = pd.DataFrame({
      "Dataset" : ["4304 images", "12193 images"],
      "Résultats" : [0.91, 0.90]
    }   )
      
    c = (
      alt.Chart(chart_data, height = 150)
      # .mark_bar()
      .encode(y= alt.Y("Dataset", sort = None, title = ""), 
              x = alt.X("Résultats", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='%')).title("Justesse sur jeu de validation"),
              text = alt.Text("Résultats", format = '.0%'),
              opacity=alt.value(0.8),
              color = alt.Color("Dataset").legend(None)
              )
      )
    graph = c.mark_bar() + c.mark_text(align = "left", dx = 6, size = 14)

    st.altair_chart(graph, use_container_width=True)
  
  #En dessous : un champ déroulant pour en savoir plus
  #expander = st.expander("Détails de l'expérimentation")
  #expander.write(
    
  st.divider()



# 3è ligne dans un container : expérimentation 3
  container3 = st.container(border=True)

  container3.markdown(
    """
    **Expérimentation :three: : Pré-traitement supplémentaire visant la réduction des biais**

    """)

  # 5 colonnes avec colonnes 2 et 4 utilisées, pour laisser un peu de marge sur les côtés et au centre 
  col1, col2, col3, col4, col5 = st.columns([0.25,4,0.5,4,0.25])
  
  #Explication de lexpérimentation
  with col2:
    st.write("")
    st.markdown(
      """
       **Transformations appliquées aux images d'entraînement et de test** :
      > - Egalisation d'histogramme (pour éliminer les variations d'intensité dûes à la technique de prise d'image)
      > - Flou gaussien (pour réduire le bruit et les artefacts sur les images - tubes, catheters...)
      
             
      """)
  
  # Graphique avec les résultats  
  with col4:
    st.write("")
    st.write("**Performance comparée avec VGG19 :**")
  
    chart_data = pd.DataFrame({
      "Tests" : ["Sans transformation", "Avec transformation"],
      "Résultats" : [0.90, 0.88]
    }   )
      
    c = (
      alt.Chart(chart_data, height = 150)
      # .mark_bar()
      .encode(y= alt.Y("Tests", sort = None, title = ""), 
              x = alt.X("Résultats", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='%')).title("Justesse sur jeu de validation"),
              text = alt.Text("Résultats", format = '.0%'),
              opacity=alt.value(0.8),
              color = alt.Color("Tests").legend(None)
              )
      )
    graph = c.mark_bar() + c.mark_text(align = "left", dx = 6, size = 14)

    st.altair_chart(graph, use_container_width=True)

  #En dessous : un champ déroulant pour en savoir plus
  #expander = st.expander("Détails de l'expérimentation")
  #expander.write(
    

  st.divider()


  #Conclusions
  st.markdown(
  """
  ##### :clipboard: **Conclusions :**
  * **L'augmentation de données ne permet pas d'améliorer la performance de classification dans notre cas**.	
  * Raisons possibles (non démontrées) :  
    - **Forte variabilité pré-existante** dans le jeu de données initial
    - Transformations appliquées **non pertinentes** d'un point de vue métier 
    - Biais non identifiés dans le jeu de données, et non corrigés par les traitements choisis

  """
  )



#Onglet "Influence du masque"
def exec_exp4_page3(self):

  st.markdown(
  """
  ##### :test_tube: **Hypothèse testée** : les masques fournis excluent une partie des informations utiles pour la performance des modèles.	

  
  """
  )


  # 1ère ligne dans un container : expérimentation 1
  container = st.container(border=True)

  container.markdown(
    """
    **Expérimentation : comparaison de la performance de classification sur le même jeu d'images POUMONS SEULS, POUMONS MASQUES, et IMAGES COMPLETES**

    """)     
     
    # 5 colonnes avec colonnes 2 et 4 utilisées, pour laisser un peu de marge sur les côtés et au centre 
  col1, col2, col3, col4, col5 = st.columns([0.25,4,0.25,4,0.25])

  # Explication de l'expérimentation
  with col2:
    st.write("")
    st.markdown(
  """
    **3 filtres appliqués au même jeu d'images** :
        
  """)
    #afficher l 'image
    st.image("src/ressources/page3b_masques.png", width = 500)

  # Graphique avec les résultats comparés    
  with col4:
    st.write("")
    st.write("**Performance comparée avec ResNet152V2 :**")
    st.write("")
    st.write("")

    chart_data = pd.DataFrame({
      "Dataset" : ["Poumons seuls", "Images complètes","Poumons masqués"],
      "Résultats" : [0.84, 0.87, 0.89]
    }   )
      
    c = (
      alt.Chart(chart_data, height = 250)
      # .mark_bar()
      .encode(y= alt.Y("Dataset", sort = None, title = ""), 
              x = alt.X("Résultats", scale=alt.Scale(domain=[0, 1.1]), axis=alt.Axis(format='%')).title("Justesse sur jeu de validation"),
              text = alt.Text("Résultats", format = '.0%'),
              opacity=alt.value(0.8),
              color = alt.Color("Dataset").legend(None)
              )
      )
    graph = c.mark_bar() + c.mark_text(align = "left", dx = 6, size = 14)

    st.altair_chart(graph, use_container_width=True)
  
  #Conclusions
  st.markdown(
  """
  ##### :clipboard: **Conclusions :**
  * Le modèle atteint une meilleure performance sur les images complètes et les images « poumons masqués » ! 
  * Les masques excluent donc une partie de l’information utile. 
  * Le score sur les images « poumons masqués » n’est pas expliqué, et pourrait être dû à des **biais non identifiés** dans les données sources.

  """
  )

  st.divider() 
  
   #En dessous : un champ déroulant pour en savoir plus
  expander = st.expander("Pour aller plus loin : analyse comparée du Grad-CAM")
  expander.markdown(
    """
    ###### Cartes d’activations Grad-CAM du modèle entraîné sur les images avec poumons, sans poumons, et complètes (IMAGE COVID – 1517)
    * Prédiction correcte du modèle dans les trois cas : COVID
    * Les 3 Grad-CAM montrent une zone activée dans la partie inférieure droite du poumon et sur la zone du cœur
    * Ces zones sont à la limite des masques, qui peuvent donc exclure des informations utiles (par exemple : opacité sur le contour du coeur) 
    * Sur d'autres images testées, le grad-CAM montre une activation sur des zones non pertinentes dans le cas de ces pathologies (clavicule, cou...)
  
"""
  )
  expander.image("src/ressources/page3b_gradcam_masques_COVID.png")