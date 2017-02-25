# Created on Fri Oct 28 00:43:02 2016
# author: Luis Campos Garrido
# Image Recognition via scikit-learn

from scipy import misc
import matplotlib.pyplot as plt
import glob
import pandas as pd


#
# Data Preparation
#

# Prueba con 1 imagen
img = misc.imread('fcbdata/iniesta/andresiniesta62.jpg')
type(img)
print img.shape
plt.figure().add_subplot(111).imshow(img)

label = []
iniesta = []
for filename in glob.glob("fcbdata/iniesta/*.jpg"):
    img = misc.imread(filename)
    # Scale colors from (0-255) to (0-1), then reshape to 3D array per pixel to preserve all color channels with .reshape(-1,3)
    img = (img / 255.0).reshape(-1)
    iniesta.append(img)
    label.append(1)
iniesta = pd.DataFrame(iniesta)

messi = []
for filename in glob.glob("fcbdata/messi/*.jpg"):
    img = misc.imread(filename)
    img = (img / 255.0).reshape(-1)
    messi.append(img)
    label.append(2)
messi = pd.DataFrame(messi)

neymar = []
for filename in glob.glob("fcbdata/neymar/*.jpg"):
    img = misc.imread(filename)
    img = (img / 255.0).reshape(-1)
    neymar.append(img)
    label.append(3)
neymar = pd.DataFrame(neymar)

X = pd.concat([iniesta, messi, neymar])
y = pd.DataFrame(label)[0]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)


#
# Data Modeling
#

# K-Neighbors
from sklearn.neighbors import KNeighborsClassifier
knmodel = KNeighborsClassifier()
knmodel.fit(X_train, y_train)
print knmodel.score(X_test, y_test)

# Support Vector Machines
from sklearn.svm import SVC
svcmodel = SVC()
svcmodel.fit(X_train, y_train) 
print svcmodel.score(X_test, y_test)

# Decision Tree
from sklearn import tree
treemodel = tree.DecisionTreeClassifier()
treemodel.fit(X_train,y_train)
print treemodel.score(X_test, y_test)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
forestmodel = RandomForestClassifier()
forestmodel.fit(X_train, y_train)
print forestmodel.score(X_test, y_test)
