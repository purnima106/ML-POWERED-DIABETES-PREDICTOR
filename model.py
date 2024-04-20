import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay 
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")
import pickle

 

# Import Dataset
dataset = pd.read_csv('diabetes.csv')
print(dataset.shape)
print(dataset.isnull().sum())

feature = dataset[["Glucose", "Insulin", "BMI", "Age"]]
target = dataset["Outcome"]



x_train2, x_test2, y_train2, y_test2 = train_test_split(feature, target)  # LR

x_train3, x_test3, y_train3, y_test3 = train_test_split(feature, target)  # nb

x_train4, x_test4, y_train4, y_test4 = train_test_split(feature, target)  # KNC
k = int(len(dataset) ** 0.5)
if k % 2 == 0:
    k = k + 1

x_train5, x_test5,  y_train5, y_test5 = train_test_split(feature, target) #DT

x_train6, x_test6,  y_train6, y_test6 = train_test_split(feature, target) #RFC


# Train-test split for SVM
dataset_X = dataset.iloc[:, [1, 4, 5, 7]].values
dataset_Y = dataset.iloc[:, 8].values

scaler = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = scaler.fit_transform(dataset_X)
X = pd.DataFrame(dataset_scaled)
Y = dataset_Y

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42, stratify=dataset['Outcome'])
#---------------------------------------------------------------------------------------------------------------------------

model_lr = LogisticRegression()
model_lr.fit(x_train2, y_train2)

model_nb = GaussianNB()
model_nb.fit(x_train3, y_train3)

model_knc = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
model_knc.fit(x_train4, y_train4)

model_dt = DecisionTreeClassifier()
model_dt.fit(x_train5, y_train5)

model_rn = RandomForestClassifier()
model_rn.fit(x_train6, y_train6)
#---------------------------------------------------------------------------------------------------------------------
# classification_report_for_LogRes
cr1 = classification_report(y_test2, model_lr.predict(x_test2))
print((cr1) + "This is the Classification Report  for Logistic Regression")

print("    ")

# classification_report_for NBG
cr2 = classification_report(y_test3, model_nb.predict(x_test3))
print((cr2) + "This is the Classification Report for GuassianNB")

print("    ")

# classification_report_for_KNC
cr3 = classification_report(y_test4, model_knc.predict(x_test4))
print((cr3) + "This is the Classification Report for KNeighborClassifier")

print("    ")

# classification_report_for_SVM
svc = SVC(kernel='linear', random_state=42)
svc.fit(X_train, Y_train)
cr4 = classification_report(Y_test, svc.predict(X_test))
print((cr4) + "This is the Classification Report for Support Vector Machine")

print("    ")

# classification_report_for_DT
cr5 = classification_report(y_test5, model_dt.predict(x_test5))
print((cr5) + "This is the Classification Report for DecisionTreeClassifier")

print("    ")

# classification_report_for_RFC
cr6 = classification_report(y_test6, model_dt.predict(x_test6))
print((cr6) + "This is the Classification Report for RandomForestClassifier")

print("    ")
#---------------------------------------------------------------------------------------------------------------------------

disp1 = ConfusionMatrixDisplay.from_estimator(model_lr, x_test2, y_test2)  # For LR
print(str(disp1.confusion_matrix)+ "Logistic Regression" )

print("    ")

disp2 = ConfusionMatrixDisplay.from_estimator(model_nb, x_test3, y_test3)  # For NB
print(str(disp2.confusion_matrix) + "Naive Bayes GaussianNB")

print("    ")


disp3 = ConfusionMatrixDisplay.from_estimator(model_knc, x_test4, y_test4)  # For KNC
print(str(disp3.confusion_matrix) + "KNieghborClassifier")

print("    ")


disp4 = ConfusionMatrixDisplay.from_estimator(svc, X_test, Y_test)  # For SVM
print(str(disp4.confusion_matrix) + "SupportVectorMachine")

print("    ")

disp5 = ConfusionMatrixDisplay.from_estimator(model_dt, x_test5, y_test5) #for dt
print(str(disp5.confusion_matrix) + "Decision Tree")

print("    ")


disp6 = ConfusionMatrixDisplay.from_estimator(model_rn, x_test6, y_test6) #for RFC
print(str(disp6.confusion_matrix) + "RandomForestClassifier")

print("    ")

#---------------------------------------------------------------------------------------------------------------------------
sl1 = model_lr.score(x_train2, y_train2)
sl2 = model_lr.score(x_test2, y_test2)
print("Logistic Regression accuracy on training set is: ", sl1)
print("Logistic Regression accuracy on testing set is: ", sl2)

print("    ")


sb1 = model_nb.score(x_train3,y_train3)
sb2 =model_nb.score(x_test3,y_test3)
print("Naive Bayes Guassian  accuracy on training set is: ", sb1)
print("Naive Bayes Guassian accuracy on testing set is: ", sb2)

print("    ")


sk1 = model_knc.score(x_train4,y_train4)
sk2 =model_knc.score(x_test4,y_test4)
print("KNeighbor Classifier  accuracy on training set is: ", sk1)
print("KNeighbor Classifier accuracy on testing set is: ", sk2)

print("    ")


sd1 = model_dt.score(x_train5, y_train5)
sd2 = model_dt.score(x_test5, y_test5)
print("Decision Tree Classifier accuracy on training set is: ", sd1)
print("Decision Tree Classifier accuracy on testing set is: ", sd2)

print("    ")


sr1 = model_rn.score(x_train6,y_train6)
sr2 =model_rn.score(x_test6,y_test6)
print("RandomForestClassifier  accuracy on training set is: ", sr1)
print("RandomForestClassifier accuracy on testing set is: ", sr2)

print("    ")

score1 = svc.score(X_test, Y_test)
score2 = svc.score(X_train,Y_train)
print("SVM accuracy on training set is:", score1)
print("SVM accuracy on testing set is:", score2)

#-------------------------------------------------------------------------------------------------------------

#method 1: Z-score
#z_scores = np.abs(stats.zscore(dataset))
#outliers_z = np.where(z_scores > 3)

#outliers = np.unique(np.concatenate([outliers_z[0]]))
#print("Detected Outliers:", outliers)"""

#-------------------------------------------------------------------------------------------------------------

# Saving the model
with open('model.pkl', 'wb') as model_file:
    pickle.dump(svc, model_file)

# Example prediction (commented out)
# model = pickle.load(open('model.pkl', 'rb'))
# print(model.predict(scaler.transform([[86, 66, 26.6, 31]])))



