import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report



st1 = st.header(" Major Project :-  Spam Detection On Sparse Text ")

st2 = st.container()

col1,col2 = st.columns(2)

with st2:
 st.write("The first row lists features like ""'the'"", ""'to'"", ""'ect'"", etc., and each email row has counts of these words. The last column is ""'Prediction'"", which we assume is the class label (spam or not spam). The data seems to have word frequencies for each email.")    
 df = pd.read_csv("E:\Venv\emails.csv")
 st.write(df.head(10))
 check = st.checkbox("Click here to Train the Dataset with Naive Bayes CLassifiction Model ")
 but = st.button('submit', type='primary')


if  check and but:
    X = df.drop(['Email No.','Prediction'],axis=1)
    y = df['Prediction']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    nb = MultinomialNB(alpha=1)
    nb.fit(X_train, y_train)
    yPred = nb.predict(X_test)
    st.write("Accuracy:- " , accuracy_score(y_test,yPred))
    st.write("Classification Report:-  " , classification_report(y_test,yPred))
    st.write("Confusion Matrix:-  " ,pd.crosstab(y_test,yPred))
    x= y_test
    y=yPred
    big=  plt.bar(x,y)
    st.write("The FIGURES SHOWCASING TRAINED DATASET IN GRAPHICAL REPRESENTATION")
    st.bar_chart(big)
    
    st.scatter_chart(big)
    
    st.line_chart(big)

st3 = st.container()


with st3 , col2:
   st.info('Completed By :- Bhoomika Sharma ')
   

  
  
   





  






   
