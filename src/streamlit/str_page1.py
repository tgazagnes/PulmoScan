import matplotlib.pyplot as plt
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import dataexploration as dexpl


def exec_page1(self):
####### Initialisation de la page
  data = dexpl.init_raw_dataset()
  
  if 'file' not in st.session_state:
    st.session_state.file = dexpl.any_random_file(data)

############################################
  st.subheader('Exploration et préparation des données')
  col1, col2 = st.columns([1,20])

##  if col1.button(':arrows_counterclockwise:','b1'):
##    st.session_state.file = dexpl.any_random_file(data)
    

  tab1, tab2, tab3 , tab4 , tab5 = st.tabs(["Données brutes", "Classes", "Données utiles et masques", "Données réduites", "Pré-traitement"])

  with tab1:

    st.markdown(
    """
#### 	 Origine des données :seedling:  

Le jeu de données utilisé, disponible sur [Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database/) est une banque de **21 165 images** de radiographies pulmonaires. Compilées par une équipe de chercheurs de l'Université de Doha (Qatar) et l'Université de Dhaka (Bangladesh), elle a été constituée à partir de diverses sources telles que:
  - Italian Society of Medical and Interventional Radiology (SIRM) 
  - Valencia Region Image Bank (BIMCV)
  - Novel Corona Virus 2019 Dataset (Joseph Paul Cohen, Paul Morrison)
  - Chest X-Ray Images (pneumonia) Kaggle database 
  - Radiological Society of North America (RSNA) Kaggle database

  """
    )
    container1 = st.container(border=True)
    if container1.button(":gray[Afficher un exemple d'image et masque associé] :arrows_counterclockwise:",'b1'):
      st.session_state.file = dexpl.any_random_file(data)
    file = st.session_state.file
    

  
    fig = plt.figure(figsize=(8,8))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.set_axis_off()
    ax2.set_axis_off()
    ax1.set_title('image : '+ st.session_state.file.split('/')[-1],
                 {'fontsize': 'small',
                 'color':'black'})
    ax2.set_title('masque : '+ st.session_state.file.split('/')[-1],
                 {'fontsize': 'small',
                 'color':'black'})

    ax1.imshow(dexpl.load(file), cmap='gray')
    ax2.imshow(dexpl.load_mask(file), cmap='gray')

    container1.pyplot(fig)



 


    st.write("Arborescence du jeu de données sur Kaggle :")

    st.image(cv2.imread('src/ressources/arborescence_data2.png'))



  #Onglet n°2
    
  with tab2:


    st.markdown(
     """
#### Equilibre du jeu de données :scales: 

Ce jeu de données est fortement deséquilibré entre les différentes classes : 
- la classe “Normal” (porteur sain) représente **48%** des images, 
- la classe "Lung Opacity" **28%**,
- la classe "Covid" **17%**, 
- la classe "Viral Pneumonia" **6%**.
"""
     )
 
    st.image(cv2.imread('src/ressources/Data_Pie.png'),caption = 'Proportion de radiographie par classe dans le jeu de données de Kaggle', width = 500)
    st.markdown(
    """
      Pour le pré-traitement, nous appliquerons un **équilibrage** du jeu de données se par sous-échantillonage sans remise avec pour etalon la taille de la classe minoritaire de pneumonie virale (1345 images).
    """)  

    st.divider()

    st.markdown(
    """
#### **Analyse des images moyennes par classe :camera:**

L'analyse des images moyennes de chaque classe permet de distinguer un contraste plus marqué pour les patients sains. 
Nous repérerons également une opacité plus importante pour les diagnostics de pathologies pulmonaires (Covid, pneumonie virale, opacité pulmonaire) comparé aux patients sains. Les différences entre les images moyennes de chaque pathologie pulmonaire sont moins évidentes à repérer à l'œil nu.

""")  
    st.image(cv2.imread('src/ressources/Mean_class.png'),caption = 'Images moyennes par classe', width = 500)



  with tab3:

    st.markdown(
    """
    #### **Postulat** :bulb: : les informations pertinentes pour la classification se situent au niveau des poumons.

La moyenne de la surface pulmonaire par image est de *23%*. 

En appliquant les masques, nous choisissons de concentrer nos modèles sur les informations utiles à l'intérieur de la zone pulmonaire uniquement. 

Nous espérons ainsi réduire la quantité d'informations à traiter, et optimiser les performances de traitements des modèles.
""") 
  
    file = st.session_state.file  
    if st.button(':gray[Afficher une image au hasard] :arrows_counterclockwise:','b2'):
      st.session_state.file = dexpl.any_random_file(data)


    fig = plt.figure(figsize=(8,8))

    ax2 = fig.add_subplot(121)
    ax3 = fig.add_subplot(122)

    ax2.set_axis_off()
    ax2.set_title('Image avec masque : '+ st.session_state.file.split('/')[-1],
                {'fontsize': 'xx-small',
                'color':'gray'})

    ax2.imshow(dexpl.apply_mask(file), cmap='gray')
      

    ax3.set_axis_off()
    ax3.set_title('Ratio de surface pulmonaire : '+ st.session_state.file.split('/')[-1],
                {'fontsize': 'xx-small',
                'color':'gray'})
    

    masque = dexpl.load_mask(file)
    pulm = masque.sum()//255
    not_pulm = masque.shape[0] * masque.shape[1] - pulm

    ax3.pie( [pulm, not_pulm ], labels = ['surface des poumons','surface masquée'], colors = ['green','orange'],explode = [0.1, 0],
      autopct = lambda x: str(round(x, 2)) + '%', pctdistance = 0.5, labeldistance = 1.2,
          textprops={'fontsize': 'x-small',
                'color':'black'})

    st.pyplot(fig)

    st.image(cv2.imread("src/ressources/distrib_utile.png"), caption = "Histogramme des ratio de surface pulmonaires sur l'ensemble des images")
    



  with tab4:

 #    col1, col2 = st.columns([3,2])
    st.markdown(
  """ 
  #### **Distribution d'intensité des images masquées :low_brightness:**
  
  En observant la distribution des niveaux de gris de chaque image (avec masque), nous observons que leurs profils varient significativement (voir ci-dessous). Il semble toutefois y avoir des similitudes de forme par classe.

  L'objectif étant d'expérimenter la classification par apprentissage profond, nous n'avons pas creusé plus loin les variations entre classes.
  
  Nous avons toutefois mené une expérimentation dite **approche réduite** consistant en une classification des images réduites à un simple vecteur 1x256.
    """)  

    st.markdown("""
  Ici nous affichons une image masquée de chaque classe avec sa distribution d'intensité correspondante : 
  """
  )
       
    image_dict = {0:'src/ressources/benchmark_distrib1.png',
                1:'src/ressources/benchmark_distrib2.png',
                2:'src/ressources/benchmark_distrib3.png',
                3:'src/ressources/Benchmark_distrib4.png'}
    
    st.image(image_dict.get(np.random.choice(range(4))))
        
    file = st.session_state.file  
    distribution, _ = dexpl.get_distrib(dexpl.apply_mask(file))
    if st.button(':gray[Afficher un exemple] :arrows_counterclockwise:','b3'):
      st.session_state.file = dexpl.any_random_file(data)

    fig = plt.figure(figsize=(16,8))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.set_axis_off()
    ax1.set_title('Image masquée : '+ st.session_state.file.split('/')[-1],
                {'fontsize': 'medium',
                'color':'gray'})
    ax2.set_title('histogramme d\'intensité de '+ st.session_state.file.split('/')[-1],
                {'fontsize': 'medium',
                'color':'gray'})

    ax1.imshow(dexpl.apply_mask(file), cmap='gray')

    ax2.hist(distribution, bins=64, range = (1,256), density = True, color='orange',rwidth=1)
    st.pyplot(fig)

  
    






     
  with tab5:

    st.markdown(
      """
#### Opérations appliquées lors du pré-traitement :gear: 
En conclusion, nous avons appliqué les opérations suivantes à chaque image pour préparer le jeu de données à utiliser dans les modèles de classification : 
-	Conversion en niveaux de gris et redimensionnement des images en 256*256 pour correspondre à la taille des masques
-	Application des masques aux images pour focaliser l'apprentissage sur la zone impactée (intérieur des poumons)
-	Pour l'approche dite "réduite" : conversion de chaque image en un vecteur 1x256

Nous avons également équilibré le jeu de données grâce à la méthode du sous-échantillonnage sans remise.
Nous avons choisi d’expérimenter sans augmentation de données à ce stade, en accord avec le mentor projet.

Exemple sur une image (cliquer sur le 1, puis le 2, puis le 3 pour afficher le traitement)

"""
)

    file = st.session_state.file
    fig = plt.figure(figsize=(8,8))


  
    col1, col2, col3 = st.columns([1,1,2])
    c1_off, c2_off, c3_off = (False,True,True)
    c1_display, c2_display, c3_display = (False,False,False)

    img1 = dexpl.load_and_resize(file)
    img2 = dexpl.apply_mask(file)
    distrib, dfrow = dexpl.get_distrib(dexpl.apply_mask(file))
    df = pd.DataFrame(dfrow,range(256)) 
    df = df.apply(lambda x: np.array(x)/(np.array(x).sum()))
  
    fig1 = plt.figure(figsize=(8,8))
    ax1 = fig1.add_subplot(111)
    ax1.set_axis_off()
    ax1.imshow(img1,cmap='gray')
    
    fig2 = plt.figure(figsize=(8,8))
    ax2 = fig2.add_subplot(111)
    ax2.set_axis_off()
    ax2.imshow(img2,cmap='gray')

    fig3 = plt.figure(figsize=(8,4))
    ax3 = fig3.add_subplot(111)
    ax3.hist(distribution, bins=64, range = (1,256), density = True, color='orange',rwidth=1)


    str1 =     """
  :gear:
  * Chargement des images
  * Conversion en niveaux de gris
  * Redimensionnement en 256*256
  ___
  """

    str2 =     """
  :gear:
  * Application du masque
  * Création d'un jeu de donnée équilibré
  * Création d'un jeu d'entraînement
  * Création d'un jeu de validation
  * Création d'un jeu de test
  ___
  """

    str3 =     """
  :gear:
  * Reduction de dimension 256x256 à 1x256
  * Création d'un jeu de donnée équilibré binaire
  * Création d'un jeu d'entrainement
  * Création d'un jeu de validation
  * Création d'un jeu de test
  ___
  """
  
    if col1.button('1 CONVERTIR ET REDIMENSIONNER','c1',disabled=c1_off):
      c1_display = True 
      c1_off = True
      c2_off = False 
      
    if col2.button('2 APPLIQUER LE MASQUE','c2',disabled=c2_off):
      c2_display = True
      c2_off = True
      c3_off = False 
  
    if col3.button('3 CONVERTIR EN ARRAY 1D','c3',disabled=c3_off):
      c3_display = True
      c3_off = True

    if c1_display:
      col1.pyplot(fig1)
      col1.markdown(str1)

    if c2_display:
      col1.pyplot(fig1)
      col2.pyplot(fig2)
      col1.markdown(str1)
      col2.markdown(str2)
      col2.image("src/ressources/Adirecte.png", width=200)

    if c3_display:
      col1.pyplot(fig1)
      col2.pyplot(fig2)
      col3.pyplot(fig3)
      col1.markdown(str1)
      col2.markdown(str2)
      col3.markdown(str3)
      col2.image("src/ressources/Adirecte.png", width=200)
      col3.image("src/ressources/Areduite.png", width=200)
      

    if st.button(':gray[Choisir une autre image] :arrows_counterclockwise:','b4'):
      st.session_state.file = dexpl.any_random_file(data)




