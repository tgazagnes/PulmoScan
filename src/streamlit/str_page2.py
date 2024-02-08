import streamlit as st
import pathlib
import random
import cv2
from tensorflow import keras
import numpy as np
import os
import subprocess
import urllib.request

def exec_page2(self):

  model_file_path = pathlib.Path("model.keras")

  @st.cache_resource
  def load_model():

    if not model_file_path.exists():
      with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
        api_key = "AIzaSyDHqY68XsiDtgOvgJ4cOpGLQ3K8H1mDy1U"
        #file_id = "1bL0HsiKScMjHGSMJkDzKOfVVeIpSkTvY" # first_model.keras
        file_id = '1S__wVx0sz74MOCHA57mYkI9QGMk31s0f' # best_model_VGG16.keras
        source = "https://www.googleapis.com/drive/v3/files/%s?alt=media&key=%s" % (file_id, api_key)
        destination_file = "model.keras"
        urllib.request.urlretrieve(source, destination_file)

    return keras.models.load_model("model.keras")

  model = load_model()

  # suppression du fichier .keras téléchargé pour optimiser la mémoire, plus besoin une fois que le modèle est chargé
  if model_file_path.exists():
    model_file_path.unlink()
  
  st.title("Modélisation")
  
  tab1, tab2, tab3 = st.tabs(["Datasets", "Modèle LeNet", "Transfer learning"])
  
  with tab1 :

    col001, col002 = st.columns(2)

    with col001 :
      st.image("src/ressources/page2/nbr_images_1.png")
    with col002 :
      st.image("src/ressources/page2/nbr_images_2.png")

  with tab2 :

    st.subheader("Tuning manuel")

    container1 = st.container(border=True)

    with container1 :
    
      st.button("Sélectionner un tuning ci-dessous pour afficher les paramètres modifiés et les métriques du modèle associé")

      col1, col2, col3, col4, col5, col6 = st.columns(6)
      with col1 : 
        tuning0 = st.checkbox('TUNING 0')
      with col2 : 
        tuning1 = st.checkbox('TUNING 1')
      with col3 : 
        tuning2 = st.checkbox('TUNING 2')
      with col4 : 
        tuning3 = st.checkbox('TUNING 3')
      with col5 : 
        tuning4 = st.checkbox('TUNING 4')
      with col6 : 
        tuning5 = st.checkbox('TUNING 5')
      
      st.write("  \n")
      st.write("  \n")

      st.image("src/ressources/page2/tuning0/tuning0.png")

      col_width_conv2D = 0.75
      col_width_maxpol = 0.85
      col_width_dropout = 0.7
      col_width_flatten = 0.6
      col_width_dense = 0.75

      col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12 = \
        st.columns([2.8,\
                    col_width_conv2D,col_width_maxpol,\
                    col_width_conv2D,col_width_maxpol,\
                    col_width_conv2D,col_width_maxpol,\
                    col_width_dropout,\
                    col_width_flatten,\
                    col_width_dense,\
                    col_width_dense,\
                    0.8])


      def write_model_parameters(model_par):
    
        with col1 : # paramètres globaux
          st.markdown (model_par[0])
        with col2 : # CONV1
          st.markdown(model_par[1])
        with col3 : # MAXPOOL 1
          st.write (model_par[2])
        with col4 : # CONV2
          st.write (model_par[3])
        with col5 : # MAXPOOL 2
          st.write (model_par[4])
        with col6 : # CONV3
          st.write (model_par[5])
        with col7 : # MAXPOOl 3
          st.write (model_par[6])
        with col8 : # DROPOUT
          st.write (model_par[7])
        with col9 : # FLATTEN
          st.write (model_par[8])
        with col10 : # DENSE 1
          st.write (model_par[9])
        with col11 : # DENSE 2
          st.write (model_par[10])

      model_param0 = ["**paramètres globaux** :  \n" + "BATCH_SIZE 32  \n" + "EPOCHS 10",
                      "", "", "", "", "", "", "", "", "", ""]
      model_param1 = ["**paramètres globaux** :  \n" + "BATCH_SIZE 64",
                      "", "", "", "", "", "", "", "", "", ""]
      model_param2 = ["", 
                      "","", "", "", "", "", "drop  \nout  \n 0.1", "", "", ""]
      model_param3 = ["", 
                      "32 filters","", "64 filters", "", "128 filters", "", "", "", "", ""]
      model_param4 = ["", 
                      "kernel 5*5","", "kernel 5*5", "", "kernel 5*5", "", "", "", "", ""]
      model_param5 = ["pre-processing :  \n" + "data augmentation  \n" + "zoom / flip / rotation",
                      "","", "", "", "", "", "", "", "", ""]
      #st.divider()

    container2 = st.container(border=True)

    with container2 :

      st.write("  \n")

      col01, col02, col03 = st.columns(3)

      if tuning0:
        write_model_parameters(model_param0)
        with col01 :
          st.image("src/ressources/page2/tuning0/tuning0_training.png")
        with col02 :
          st.image("src/ressources/page2/tuning0/tuning0_class_report.png")
        with col03 :
          st.image("src/ressources/page2/tuning0/tuning0_conf_matrix.png")
      if tuning1:
        write_model_parameters(model_param1)
        with col01 :
          st.image("src/ressources/page2/tuning1/tuning1_training.png")
        with col02 :
          st.image("src/ressources/page2/tuning1/tuning1_class_report.png")
        with col03 :
          st.image("src/ressources/page2/tuning1/tuning1_conf_matrix.png")
      if tuning2:
        write_model_parameters(model_param2)
        with col01 :
          st.image("src/ressources/page2/tuning2/tuning2_training.png")
        with col02 :
          st.image("src/ressources/page2/tuning2/tuning2_class_report.png")
        with col03 :
          st.image("src/ressources/page2/tuning2/tuning2_conf_matrix.png")   
      if tuning3:
        write_model_parameters(model_param3)
        with col01 :
          st.image("src/ressources/page2/tuning3/tuning3_training.png")
        with col02 :
          st.image("src/ressources/page2/tuning3/tuning3_class_report.png")
        with col03 :
          st.image("src/ressources/page2/tuning3/tuning3_conf_matrix.png") 
      if tuning4:
        write_model_parameters(model_param4)
        with col01 :
          st.image("src/ressources/page2/tuning4/tuning4_training.png")
        with col02 :
          st.image("src/ressources/page2/tuning4/tuning4_class_report.png")
        with col03 :
          st.image("src/ressources/page2/tuning4/tuning4_conf_matrix.png") 
      if tuning5:
        write_model_parameters(model_param5)
        with col01 :
          st.image("src/ressources/page2/tuning5/tuning5_training.png")
        with col02 :
          st.image("src/ressources/page2/tuning5/tuning5_class_report.png")
        with col03 :
          st.image("src/ressources/page2/tuning5/tuning5_conf_matrix.png")    

    st.subheader("Analyse des images mal prédites")

    bad_prediction_dir = "src/ressources/page2/bad_predictions"

  # ['COVID', 'Lung_Opacity', 'Normal', 'Viral_Pneumonia']

    if st.button('Cliquer pour afficher des images mal prédites'):

      col0001, col0002, col0003, col0004 = st.columns(4)

      with col0001:

        bad_prediction_dir_COVID = pathlib.Path(bad_prediction_dir, "COVID")
        bad_prediction_COVID_files = list(bad_prediction_dir_COVID.iterdir())
        bad_prediction_COVID_file = str(pathlib.Path(random.choice(bad_prediction_COVID_files)))
        st.image (bad_prediction_COVID_file)

      with col0002:

        bad_prediction_dir_Lung_Opacity = pathlib.Path(bad_prediction_dir, "Lung_Opacity")
        bad_prediction_Lung_Opacity_files = list(bad_prediction_dir_Lung_Opacity.iterdir())
        bad_prediction_Lung_Opacity_file = str(pathlib.Path(random.choice(bad_prediction_Lung_Opacity_files)))
        st.image (bad_prediction_Lung_Opacity_file)

      with col0003:

        bad_prediction_dir_Normal = pathlib.Path(bad_prediction_dir, "Normal")
        bad_prediction_Normal_files = list(bad_prediction_dir_Normal.iterdir())
        bad_prediction_Normal_file = str(pathlib.Path(random.choice(bad_prediction_Normal_files)))
        st.image (bad_prediction_Normal_file)

      with col0004:

        bad_prediction_dir_Viral_Pneumonia = pathlib.Path(bad_prediction_dir, "Viral_Pneumonia")
        bad_prediction_Viral_Pneumonia_files = list(bad_prediction_dir_Viral_Pneumonia.iterdir())
        bad_prediction_Viral_Pneumonia_file = str(pathlib.Path(random.choice(bad_prediction_Viral_Pneumonia_files)))
        st.image (bad_prediction_COVID_file)
      
    st.subheader("Tuning automatique") 

    st.write("Utilisation du Keras tuner avec 2 algorithmes : *RandomSearch* et *Hyperband*  \n\n\n")
    st.write("Choix de l'algorithme *Hyperband*  :  \n \
    :heavy_minus_sign: *RandomSearch* : Combinaisons de valeurs d'hyperparamètres choisies au hasard peuvent conduire à des résultats pas optimaux.  \n \
    :heavy_plus_sign: *Hyperband* :     Algo optimisé qui s'éxecute sur un nombre limité d'epochs pour sélectionner les meilleures combinaisons.") 

    container3 = st.container(border=True)

    with container3:
      st.write("ESPACE DE RECHERCHE DES HYPERPARAMETRES")
      st.image("src/ressources/page2/tuning0/tuning0.png")
      col00001, col00002, col00003, col00004, col00005, col00006, col00007, col00008, col00009, col000010, col000011, col000012 = \
      st.columns([2.8,\
                  col_width_conv2D,col_width_maxpol,\
                  col_width_conv2D,col_width_maxpol,\
                  col_width_conv2D,col_width_maxpol,\
                  col_width_dropout,\
                  col_width_flatten,\
                  col_width_dense,\
                  col_width_dense,\
                  0.8])           
      with col00001 : # paramètres globaux
        st.markdown ("**paramètres globaux** :  \n" + "BATCH_SIZE 32  \n" + "EPOCHS  10  \n" + \
                  "LEARNING RATE  \n" + "[0.1, 0.01, 0.001, 0.0001, 1e-05]")
      with col00002 : # CONV1
        st.markdown("**filters**:  \nmin 16  \nmax  \n128  \nstep 32  \n**kernel**:  \nmin 3  \nmax 5  \nstep 1")
      with col00003 : # MAXPOOL 1
        st.write("")
      with col00004 : # CONV2
        st.markdown("**filters**:  \nmin 16  \nmax  \n128  \nstep 32  \n**kernel**:  \nmin 3  \nmax 5  \nstep 1")
      with col00005 : # MAXPOOL 2
        st.write("")  
      with col00006 : # CONV3
        st.markdown("**filters**:  \nmin 16  \nmax  \n128  \nstep 32  \n**kernel**:  \nmin 3  \nmax 5  \nstep 1")
      with col00007 : # MAXPOOl 3
        st.write("") 
      with col00008 : # DROPOUT
        st.markdown ("**drop**  \n **out**:  \nmin  \n0  \nmax  \n0.5  \nstep  \n0.1")
      with col00009 : # FLATTEN
        st.write("") 
      with col000010 : # DENSE 1
        st.markdown ("**units**:  \n[32,   \n64,  \n128,  \n256]")
      with col000011 : # DENSE 2
        st.write("") 

    container4 = st.container(border=True)

    with container4:

      st.write("RESULTATS TUNING AUTOMATIQUE  \n 10 combinaisons d'hyperparamètres avec les meilleures accuracy de validation")
      st.image("src/ressources/page2/keras_tuner_hyperband_best_results.png")

    container5 = st.container(border=True)

    with container5:

      st.text("METRIQUES MEILLEUR MODELE")
              
      col000001, col000002 = st.columns(2)

      with col000001:
        st.image("src/ressources/page2/keras_tuner_hyperband_class_report.png")

      with col000002:
        st.image("src/ressources/page2/keras_tuner_hyperband_conf_matrix.png")

  with tab3 :

    st.subheader("Concept")

    st.markdown("Réutilisation de la partie convolutionnelle d'un modèle pré-entrainé sur une autre tache de classification")

    st.markdown("**Etape 1** : Choix d'un modèle pré-entrainé")

    st.image("src/ressources/page2/transfer_learning/vgg16_full.jpg")

    st.markdown("**Etape 2** : Suppression des couches denses et de sortie du modèle pré-entrainé")

    st.image("src/ressources/page2/transfer_learning/vgg16_remove_top.jpg")

    st.markdown("**Etape 3** : Insertion de nouvelles couches denses et de sortie spécifiques à notre modèle à entrainer  \n\
                Optionnel : Réentrainement de couches convolutionnelles du modèle pré-entrainé ('fine-tuning')")

    st.image("src/ressources/page2/transfer_learning/transfer-learning-fine-tuning.jpg")

    st.subheader("Résultats")

    st.image("src/ressources/page2/transfer_learning/results.png")

    st.subheader("Meilleur modèle")

    test_set_dir = "src/ressources/page2/predictions"
    class_ar =  ['COVID', 'Lung_Opacity', 'Normal', 'Viral_Pneumonia']

    def img_display_pred(dir, real_class):
      test_set_files = list(dir.iterdir())
      img_test_file = str(pathlib.Path(random.choice(test_set_files)))
      img_test = cv2.imread(img_test_file, cv2.IMREAD_GRAYSCALE)
      cimg_test=cv2.cvtColor(img_test,cv2.COLOR_GRAY2RGB)
      cimg_test_preproc = keras.applications.vgg16.preprocess_input(cimg_test)
      cimg_test_preproc = np.expand_dims(cimg_test_preproc, axis=0)
      pred_prob = model.predict(cimg_test_preproc)
      pred_class = np.argmax(pred_prob[0])
      st.image(img_test_file)
      if pred_class != real_class:
        st.write(f"Classe prédite :  \n:red[{class_ar[pred_class]}]")
      else:
        st.write(f"Classe prédite :  \n:green[{class_ar[pred_class]}]")
        
    container6 = st.container(border=True)

    with container6:

      if st.button("Cliquer pour afficher une image de chaque classe et la classe prédite par le meilleur modèle"):   

        # ['COVID', 'Lung_Opacity', 'Normal', 'Viral_Pneumonia']
        col0000001, col0000002, col0000003, col0000004 = st.columns(4)

        with col0000001 :
          st.write('Classe réelle :  \n' + class_ar[0])
          test_dir = pathlib.Path(test_set_dir, class_ar[0])
          img_display_pred(test_dir,0)
        with col0000002 : 
          st.write('Classe réelle :  \n' + class_ar[1])
          test_dir = pathlib.Path(test_set_dir, class_ar[1])
          img_display_pred(test_dir,1)
        with col0000003 : 
          st.write('Classe réelle :  \n' + class_ar[2])
          test_dir = pathlib.Path(test_set_dir, class_ar[2])
          img_display_pred(test_dir,2)
        with col0000004 :         
          st.write('Classe réelle :  \n' + class_ar[3])
          test_dir = pathlib.Path(test_set_dir, class_ar[3])
          img_display_pred(test_dir,3)
          
        

  








   








