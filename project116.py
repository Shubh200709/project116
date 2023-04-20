import pandas as pd
import matplotlib.pyplot as plp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import accuracy_score
import plotly.express as px
import plotly.graph_objects as go

data = pd.read_csv('Admission_Predict.csv')
toefl = data['TOEFL Score'].to_list()
gre = data['GRE Score'].to_list()
chance = data['Chance of admit'].to_list()

scatter = px.scatter(x=toefl,y=gre)
#scatter.show()

colors = []
for i in chance:
    if(i == 0):
        colors.append('red')
    else:
        colors.append('green')

graph = go.Figure(go.Scatter(x=toefl,y=gre,marker=dict(color=colors),mode = 'markers'))

score = data[['TOEFL Score','GRE Score']]
result = data['Chance of admit']

score_test,score_train,result_test,result_train = train_test_split(score,result,test_size=0.75,random_state=0)

logreg = LogisticRegression()
logreg.fit(score_train,result_train)

result_predict = logreg.predict(score_test)
accuracy = accuracy_score(result_test,result_predict)

print('accuracy:',accuracy)