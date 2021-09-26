from datetime import date, timedelta 
import streamlit as st
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 
import datetime

st.markdown("# Cases and Testing")
import pandas as pd 
CMalaysia = pd.read_csv('cases_malaysia.csv') 
CState    = pd.read_csv('cases_state.csv')
clusters  = pd.read_csv('clusters.csv') 
TMalaysia = pd.read_csv('tests_malaysia.csv') 
TState    = pd.read_csv('tests_state.csv')

st.markdown("# Healthcare")
pkrc          = pd.read_csv('pkrc.csv') 
hospital      = pd.read_csv('hospital.csv') 
icu           = pd.read_csv('icu.csv')

st.markdown("# Deaths") 
DMalaysia= pd.read_csv("deaths_malaysia.csv") 
DState   = pd.read_csv("deaths_state.csv")

st.markdown("# Static data")
population    = pd.read_csv("population.csv")

st.markdown("# Cleaning")
selectedData = [TState, CState, hospital] 
selectedState = ["Pahang", "Kedah", "Johor", "Selangor"] 
sState = [] 

for i in selectedState: 
  sData = pd.DataFrame() 
  for j in selectedData: 
    aData = j[j["state"] == i] 
    if sData.empty: sData = aData 
    else:           sData = pd.merge(sData, aData, how = "outer", on = "date") 
    sData = sData.drop("state", 1) 
    sData = sData.dropna() 
  sData = sData.drop("date", 1) 
  sState.append(sData)

st.markdown("# Binning") 
columns = list(sState[0].columns.values.tolist()) 
groupNames = ["Low", "Medium", "High"] 
binState = [] 

for i in sState: 
  aBin = pd.DataFrame() 
  for j in columns: 
    bins = np.linspace(min(i[j]), max(i[j]), 4) 
    aBin[j + "_binned"] = pd.cut(i[j], bins, labels = groupNames, include_lowest = True) 
  binState.append(aBin)

"""#### (i) 
Discuss the exploratory data analysis steps you have conducted including detection of outliers and missing values? 
<font color="red"> (2m) </font>
"""

# The dataset used is "pkrc.csv"
sns.set(rc = {"figure.figsize" : (50, 25)}) 
sns.boxplot(x = "state", y = "beds", data = pkrc)

pkrc.isna().sum()

"""#### (ii) 
What are the states that exhibit strong correlation with (i) Pahang, and (ii) Johor? <font color="red"> (2m) </font>
"""

states = ["Johor", "Kedah", "Kelantan", "Melaka", "Negeri Sembilan", "Pahang", "Perak", "Perlis", "Pulau Pinang", "Sabah", "Sarawak", "Selangor", "Terengganu", "W.P. Kuala Lumpur", "W.P. Labuan", "W.P. Putrajaya"]

eachCS = [] 

for i in states: 
  aState = CState[CState["state"] == i] 
  eachCS.append(aState)

for i, j in enumerate(eachCS): 
  st.write ("State:", states[i]) 
  fig, ax = plt.subplots(figsize = [16,12]) 
  lineAState = sns.lineplot(x = "date", y = "cases_new", ax = ax, data = j, legend = "brief", label = states[i]) 
  plt.show()

"""Based on the graphs shown above, the states that exhibit strong correlation with Pahang is Kelantan, while the state that exihibit strong correlation with Johor is Terengganu. This can be observed from the trend of the daily cases graph. The peak of the line graph indicates an outbreak going on in the state, and the state that have a strong correlation also has a similar trend line.

#### (iii) 
What are the strong features/indicators to daily cases for (i) Pahang, (ii) Kedah,(iii) Johor, and (iv) Selangor? <font color="red"> (3m) </font>

[Note: you must use at least 2 methods to justify your findings]
"""

st.markdown("# Chi-Square Test")
#Manual ordinal encoding where low = 0, medium = 1, and high = 2
encBin = [] 

for i in binState: 
  X = i.drop("cases_new_binned", axis = 1) 
  y = i["cases_new_binned"] 

  for k, l in enumerate(groupNames): 
    X = X.replace(l, k) 
    y = y.replace(l, k) 
  
  encBin.append([X, y])

st.markdown("# Training Data")
from sklearn.model_selection import train_test_split 

trainSetCS = [] 

for (X, y) in encBin: 
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.75, random_state = 1) 
  trainSetCS.append([X_train, X_test, y_train, y_test])

st.markdown("# Chi Square Test")
from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import chi2 

cSquare = [] 
xTrainCS = [] 
xTestCS = [] 

for (X_train, X_test, y_train, y_test) in trainSetCS: 
  chiSq = SelectKBest(score_func = chi2, k = "all") 
  chiSq.fit(X_train, y_train) 
  X_train_chiSq = chiSq.transform(X_train) 
  X_test_chiSq = chiSq.transform(X_test) 

  cSquare.append(chiSq) 
  xTrainCS.append(X_train_chiSq) 
  xTestCS.append(X_test_chiSq)

st.markdown("# Display Results")
cRef = pd.DataFrame(encBin[0][0].columns) 
cRef = cRef.rename(columns = {0 : "Column Reference"}) 
cRef

for i, j in enumerate(cSquare): 
  st.write ("\nState:", selectedState[i]) 
  for k, l in enumerate(j.scores_): 
    st.write (encBin[i][0].columns[k], "=", l) 
  plt.bar([k for k in range(len(j.scores_))], j.scores_) 
  st.pyplot()
   

st.markdown("# Boruta")

from sklearn.ensemble import RandomForestClassifier 
from boruta import BorutaPy 
from sklearn.preprocessing import MinMaxScaler 

def ranking(ranks, names, order = 1): 
  minmax = MinMaxScaler() 
  ranks = minmax.fit_transform(order * np.array([ranks]).T).T[0] 
  ranks = map(lambda x: round(x, 2), ranks) 
  return dict(zip(names, ranks))

bScore = [] 

for i in sState: 
  X = i.drop("cases_new", axis = 1) 
  y = i["cases_new"] 

  colNames = X.columns 

  rf = RandomForestClassifier(n_jobs = -1, class_weight = "balanced", max_depth = 5) 
  featSelector = BorutaPy(rf, n_estimators = "auto", random_state = 1) 

  featSelector.fit(X.values, y.values.ravel()) 

  borutaScore = ranking(list(map(float, featSelector.ranking_)), colNames, order = -1) 
  borutaScore = pd.DataFrame(list(borutaScore.items()), columns = ["Features", "Score"]) 
  borutaScore = borutaScore.sort_values("Score", ascending = False) 

  bScore.append(borutaScore)

for i, j in enumerate(bScore): 
  st.write("\nState:", selectedState[i]) 
  st.write(j.nlarge
           (3, ["Score"])) 
  borutaPlot = sns.catplot(x = "Score", y = "Features", data = j, kind = "bar", height = 14, aspect = 1.9, palette = 'coolwarm') 
  plt.title("Boruta all features for " + selectedState[i]) 
  st.pyplot()

for i, j in enumerate(bScore): 
  j = j[j["Score"] >= 0.5]

"""#### (iv) 

Comparing regression and classification models, what model performs well in predicting the daily cases for (i) Pahang, (ii) Kedah, (iii) Johor, and (iv) Selangor? <font color="red"> (3m) </font>

Requirements:
1. Use TWO(2) regression models and TWO(2) classification models
2. Use appropriate evaluation metrics.
"""

st.markdown("# Random Forest")
from sklearn.model_selection import train_test_split 

regTrainSet = [] 

for i, j in enumerate(sState): 
  X = j.drop("cases_new", axis = 1) 
  Xb = X[bScore[i]["Features"]] 
  y = j["cases_new"] 

  X_train, X_test, y_train, y_test = train_test_split(Xb, y, test_size = 0.3, random_state = 4) 
  regTrainSet.append([X_train, X_test, y_train, y_test])

from sklearn.ensemble import RandomForestRegressor 

regRF = [] 
regRFDiff = [] 

for (X_train, X_test, y_train, y_test) in regTrainSet: 
  rrf = RandomForestRegressor(max_depth = 5, random_state = 4, min_samples_leaf = 8, min_weight_fraction_leaf = 0.1) 
  rrf.fit(X_train, y_train) 
  y_pred = rrf.predict(X_test) 
  regRF.append(rrf) 

  aDiff = pd.DataFrame({"Original New Cases" : y_test, "Predicted New Cases" : y_pred}, columns = ["Original New Cases", "Predicted New Cases"]) 
  aDiff["Differences"] = aDiff["Predicted New Cases"].astype("int64") - aDiff["Original New Cases"].astype("int64") 
  aDiff.reset_index(drop = True, inplace = True) 
  regRFDiff.append(aDiff)

for i, (X_train, X_test, y_train, y_test) in enumerate(regTrainSet): 
  st.write("\nState:", selectedState[i]) 
  rrf = regRF[i] 
  st.write("Accuracy of Random Forest on training set: {:.3f}".format(rrf.score(X_train, y_train))) 
  st.write("Accuracy of Random Forest on test set: {:.3f}".format(rrf.score(X_test, y_test))) 
  display(regRFDiff[i])

rrfResult = pd.DataFrame(columns = ["State", "Training Set Accuracy", "Test Set Accuracy"]) 
for i, (X_train, X_test, y_train, y_test) in enumerate(regTrainSet): 
  rrf = regRF[i] 
  rrfResult.loc[len(rrfResult.index)] = [selectedState[i], rrf.score(X_train, y_train), rrf.score(X_test, y_test)] 

rrfResult

st.markdown("# Decision Tree")
from sklearn.tree import DecisionTreeRegressor 

regDT = [] 
regDTDiff = [] 

for (X_train, X_test, y_train, y_test) in regTrainSet: 
  rdt = DecisionTreeRegressor(max_depth = 5, random_state = 4, min_samples_leaf = 8, min_weight_fraction_leaf = 0.1) 
  rdt.fit(X_train, y_train) 
  y_pred = rdt.predict(X_test) 
  regDT.append(rdt) 

  aDiff = pd.DataFrame({"Original New Cases" : y_test, "Predicted New Cases" : y_pred}, columns = ["Original New Cases", "Predicted New Cases"]) 
  aDiff["Differences"] = aDiff["Predicted New Cases"].astype("int64") - aDiff["Original New Cases"].astype("int64") 
  aDiff.reset_index(drop = True, inplace = True) 
  regDTDiff.append(aDiff)

for i, (X_train, X_test, y_train, y_test) in enumerate(regTrainSet): 
  print("\nState:", selectedState[i]) 
  rdt = regDT[i] 
  st.write("Accuracy of Decision Tree on training set: {:.3f}".format(rdt.score(X_train, y_train))) 
  st.write("Accuracy of Decision Tree on test set: {:.3f}".format(rdt.score(X_test, y_test))) 
  display(regDTDiff[i])

rdtResult = pd.DataFrame(columns = ["State", "Training Set Accuracy", "Test Set Accuracy"]) 
for i, (X_train, X_test, y_train, y_test) in enumerate(regTrainSet): 
  rdt = regDT[i] 
  rdtResult.loc[len(rdtResult.index)] = [selectedState[i], rdt.score(X_train, y_train), rdt.score(X_test, y_test)] 

rdtResult

st.markdown("# Classification")
from sklearn.model_selection import train_test_split 

classTrainSet = [] 

for i, j in enumerate(binState): 
  X = j.drop("cases_new_binned", axis = 1) 
  Xd = pd.get_dummies(X) 
  y = j["cases_new_binned"] 

  X_train, X_test, y_train, y_test = train_test_split(Xd, y, test_size = 0.3, random_state = 4) 
  classTrainSet.append([X_train, X_test, y_train, y_test])

st.markdown("# Random Forest")
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score 

classRF = [] 
classRFDiff = [] 
scoreRF = [] 

for (X_train, X_test, y_train, y_test) in classTrainSet: 
  crf = DecisionTreeClassifier(max_depth = 5, random_state = 4, min_samples_leaf = 8, min_weight_fraction_leaf = 0.1) 
  crf.fit(X_train, y_train) 
  y_pred = crf.predict(X_test) 
  classRF.append(crf) 

  aDiff = pd.DataFrame({"Original New Cases" : y_test, "Predicted New Cases" : y_pred}, columns = ["Original New Cases", "Predicted New Cases"]) 
  aDiff.reset_index(drop = True, inplace = True) 
  classRFDiff.append(aDiff) 

  aScore = accuracy_score(y_test, y_pred) 
  scoreRF.append(aScore)

for i, (X_train, X_test, y_train, y_test) in enumerate(classTrainSet): 
  print("\nState:", selectedState[i]) 
  crf = classRF[i] 
  st.write("Accuracy of Random Forest on training set: {:.3f}".format(crf.score(X_train, y_train))) 
  st.write("Accuracy of Random Forest on test set: {:.3f}".format(crf.score(X_test, y_test))) 
  st.write('Accuracy score of Random Forest: {:.3f}'.format(scoreRF[i])) 
  display(classRFDiff[i])

crfResult = pd.DataFrame(columns = ["State", "Training Set Accuracy", "Test Set Accuracy", "Accuracy Score"]) 
for i, (X_train, X_test, y_train, y_test) in enumerate(classTrainSet): 
  crf = classDT[i] 
  crfResult.loc[len(crfResult.index)] = [selectedState[i], crf.score(X_train, y_train), crf.score(X_test, y_test), crf.score(X_test, y_test)] 

crfResult

st.markdown("# Decision Tree")
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 

classDT = [] 
classDTDiff = [] 
scoreDT = [] 

for (X_train, X_test, y_train, y_test) in classTrainSet: 
  cdt = DecisionTreeClassifier(max_depth = 5, random_state = 4, min_samples_leaf = 8, min_weight_fraction_leaf = 0.1) 
  cdt.fit(X_train, y_train) 
  y_pred = cdt.predict(X_test) 
  classDT.append(cdt) 

  aDiff = pd.DataFrame({"Original New Cases" : y_test, "Predicted New Cases" : y_pred}, columns = ["Original New Cases", "Predicted New Cases"]) 
  aDiff.reset_index(drop = True, inplace = True) 
  classDTDiff.append(aDiff) 

  aScore = accuracy_score(y_test, y_pred) 
  scoreDT.append(aScore)

for i, (X_train, X_test, y_train, y_test) in enumerate(classTrainSet): 
  st.write("\nState:", selectedState[i]) 
  cdt = classDT[i] 
  st.write("Accuracy of Decision Tree on training set: {:.3f}".format(cdt.score(X_train, y_train))) 
  st.write("Accuracy of Decision Tree on test set: {:.3f}".format(cdt.score(X_test, y_test))) 
  st.write('Accuracy score of Decision Tree: {:.3f}'.format(scoreDT[i])) 
  display(classDTDiff[i])

cdtResult = pd.DataFrame(columns = ["State", "Training Set Accuracy", "Test Set Accuracy", "Accuracy Score"]) 
for i, (X_train, X_test, y_train, y_test) in enumerate(classTrainSet): 
  cdt = classDT[i] 
  cdtResult.loc[len(cdtResult.index)] = [selectedState[i], cdt.score(X_train, y_train), cdt.score(X_test, y_test), cdt.score(X_test, y_test)] 

cdtResult
