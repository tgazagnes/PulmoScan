import streamlit as st

def exec_page4(self):
  st.subheader("Résultats et perspectives")
  st.write("")




  st.markdown(
  """

  #### Atteinte des objectifs :trophy: 
  - :heavy_check_mark: Développement et test de modèles de Deep Learning et d'apprentissage par transfert permettant d'obtenir une classification d'images médicales juste à près de 90%
  - :heavy_check_mark: Mise en oeuvre de techniques d'interprétation des résultats (Grad-CAM) et confrontation des résultats avec une expertise métier
  - :heavy_check_mark: Expérimentations complémentaires et identification de facteurs affectant la performance des modèles (réduction de dimensions, augmentation de données, réduction des biais par pré-traitement des images, masquage différenciés avec/sans poumons)
  - :heavy_check_mark: Montée en compétence rapide sur le Deep Learning et mise en application concrète des acquis de la formation  
  ---

  #### Limites identifiées :warning: 
  - **Interprétabilité** : peu de lien clair entre les zones activées par les modèles et les signes pathologiques recherchés par les professionnels => frein à l'acceptabilité de la solution
  - **Qualité des données** : absence de métadonnées sur les patients et présence probable de biais "invisibles", liés aux sources, qui peuvent expliquer la bonne performance de classification sur les images "sans poumons".
  """
  )

  container1 = st.container(border=True)
  container1.markdown(
    """

  Un article scientifique publié sur [Nature.com](https://www.nature.com/articles/s41598-023-30174-1) met en avant les biais non testés dans les bases d'images publiques de radios du thorax publiques (dont la nôtre) :
  - Biais liés au profil des patients (non connus dans notre dataset) : sexe, âge, autres caractéristiques démographiques
  - Biais liés aux conditions de capture d'image : appareil mobile ou non, consignes données au patient, cathéters ou tubes d'intubation...
  
  **Sans contrôle sur la représentativité des différentes populations et techniques de prise d'image, il est possible que le modèle apprenne à reconnaître les caractéristiques des patients ou des machines, et non de la maladie**.
   Dans notre dataset, on constate par exemple que la classe Viral Pneumonia est issue d'une seule source, différente des autres classes. Le score élevé de classification sur cette classe pourrait être expliqué par des spécificités liées à cette source.
  
  """
  )

  container1.image("src/ressources/page4_sourcedonnees.PNG", caption = "Répartition des classes par sources des images")
  
  st.divider()

  st.markdown(
    """
  #### Perspectives et potentiel de généralisation :medical_symbol: 
  - **Bénéfices potentiels** : Aide au triage rapide des patients en situation d'urgence (justesse similaire au test PCR pour un délai réduit)
  - **Pertinence de la solution** : 
    - Nécessité de solutions **interprétables** conçues pour **assister/compléter** l'expertise du spécialiste, soumis aux enjeux de responsabilité médicale
    - Nécessité d'un travail approfondi sur la **qualité des données d'entraînement** et d'une validation scientifique robuste 
  - Frein à la **disponibilité** des données étiquetées (coût/ressources)
    
  """
  )

  st.markdown(
  """
  ---
  #### Remerciements :pray: 
  Nous remercions l’équipe DataScientest pour l’opportunité de travailler sur ce sujet, Gaël Penessot qui a été notre mentor pour l’accompagnement tout au long du projet, et Martine 
Mattei (radiologue) pour ses éclairages médicaux et son regard critique sur les cartes d’activation Grad-CAM issues des modèles testés.

  """
  )

  st.divider()
  st.markdown(
  """
  ####  Bibliographie :books:
-	Yadav, Ruchi & Sahoo, Debasis & Graham, Ruffin. (2020). Thoracic imaging in COVID–19. Cleveland Clinic Journal of Medicine. 87. 10.3949/ccjm.87a.ccc032.
-	Wong HYF, Lam HYS, Fong AH, et al. Frequency and distribution of chest radiographic ﬁndings in COVID-19 positive patients [published online ahead of print, 2019 Mar 27]. Radiology 2019;201160.doi:10.1148/radiol.2020201160
-	Narin A., Kaya C., Pamuk Z. Automatic detection of coronavirus disease (covid-19) using x-ray images and deep convolutional neural networks. arXiv preprint arXiv:2003.10849. 2020 [Google Scholar](https://scholar.google.com/scholar_lookup?journal=arXiv+preprint+arXiv:2003.10849&title=Automatic+detection+of+coronavirus+disease+(covid-19)+using+x-ray+images+and+deep+convolutional+neural+networks&author=A.+Narin&author=C.+Kaya&author=Z.+Pamuk&publication_year=2020&)
-	Appasami G, Nickolas S. A deep learning-based COVID-19 classification from chest X-ray image: case study. Eur Phys J Spec Top. 2022;231(18-20):3767-3777. doi: 10.1140/epjs/s11734-022-00647-x. Epub 2022 Aug 18. PMID: 35996535; PMCID: PMC9386662.
-	Sadre, R., Sundaram, B., Majumdar, S. et al. Validating deep learning inference during chest X-ray classification for COVID-19 screening. Sci Rep 11, 16075 (2021). https://doi.org/10.1038/s41598-021-95561-y
-	Talaat, M.; Si, X.; Xi, J. Multi-Level Training and Testing of CNN Models in Diagnosing Multi-Center COVID-19 and Pneumonia X-ray Images. Appl. Sci. 2023, 13, 10270. https://doi.org/10.3390/app131810270 
-	Heidari M, Mirniaharikandehei S, Khuzani AZ, Danala G, Qiu Y, Zheng B. Improving the performance of CNN to predict the likelihood of COVID-19 using chest X-ray images with preprocessing algorithms. Int J Med Inform. 2020 Dec;144:104284. doi: 10.1016/j.ijmedinf.2020.104284. Epub 2020 Sep 23. PMID: 32992136; PMCID: PMC7510591. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7510591/ 
-	Wang L, Lin ZQ, Wong A. COVID-Net: a tailored deep convolutional neural network design for detection of COVID-19 cases from chest X-ray images. Sci Rep. 2020 Nov 11;10(1):19549. doi: 10.1038/s41598-020-76550-z. PMID: 33177550; PMCID: PMC7658227.
- Arias-Garzón, D., Tabares-Soto, R., Bernal-Salcedo, J. et al. Biases associated with database structure for COVID-19 detection in X-ray images. Sci Rep 13, 3477 (2023). https://doi.org/10.1038/s41598-023-30174-1

  """
  )