import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from str_page0 import exec_page0
from str_page1 import exec_page1
from str_page2 import exec_page2
from str_page3 import exec_page3
from str_page4 import exec_page4

st.set_page_config(layout="wide")

#st.title("Oct23 Bootcamp DS  Radios pulmonaires ")

st.sidebar.image("src/ressources/Logo_pulmoscanCapture.PNG")
st.sidebar.write("**Projet PulmoScan : Classification de radios pulmonaires par deep learning**")
st.sidebar.write("")

pages=["1\. Contexte et objectifs", "2\. Exploration et préparation des données","3\. Modélisation","4\. Approfondissements","5\. Résultats et perspectives"]
page=st.sidebar.radio("Aller vers", pages)
st.sidebar.write("")
st.sidebar.image("src/ressources/logo_datascientest.png")

st.sidebar.info(
"""
A propos : \n
Cette application a été développée par l'équipe OCT23_DS_Projet radio dans le cadre de la formation bootcamp [Data Scientist](http://datascientest.com) :
* [Steve Costalat](https://www.linkedin.com/in/stevecostalat)
* [Thibaut Gazagnes](https://www.linkedin.com/in/thibautgazagnes)
* [Nicolas Gorgol](https://www.linkedin.com/in/nicolas-gorgol-53a329ba/)

Mentor projet Datascientest :
* [Gaël Penessot](https://www.linkedin.com/in/gael-penessot)

Retrouvez le projet sur [Github](https://github.com/DataScientest-Studio/OCT23_BDS_Radios_Poumons).
"""
    )

if page == pages[0] :
  exec_page0(st)
 
if page == pages[1] :
  exec_page1(st)

if page == pages[2] :
  exec_page2(st)


if page == pages[3] :
  exec_page3(st)



if page == pages[4] :
  exec_page4(st)
