import streamlit as st
from str_page3_bis import exec_exp2_page3, exec_exp4_page3
from str_page3_ter import exec_exp1_page3, exec_exp3_page3

def exec_page3(self):
  st.subheader('Approfondissements')
  
  tab1, tab2, tab3 , tab4  = st.tabs(["Classification à 3 classes","Réduction de dimension", "Augmentation de données",  "Influence du masque"])
  with tab1:
    exec_exp3_page3(st) 
  with tab2:
    exec_exp1_page3(st)
  with tab3:
    exec_exp2_page3(st)
  with tab4:
    exec_exp4_page3(st)
     
     
     
