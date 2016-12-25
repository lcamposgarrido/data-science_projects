# Created on Fri Dec 16 2016
# Luis Campos Garrido
# Investors behaviour analysis

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing
pl.style.use('ggplot')
  
def drawVectors(transformed_features, components_, columns, plt):
  num_columns = len(columns)
  # This funtion will project the original feature
  # onto the principal component feature-space, so that we can
  # visualize how important each one was in the
  # multi-dimensional scaling
  xvector = components_[0] * max(transformed_features[:,0])
  yvector = components_[1] * max(transformed_features[:,1])
 important_features = { columns[i] : math.sqrt(xvector[i]**2 + yvector[i]**2) for i in range(num_columns) }
  important_features = sorted(zip(important_features.values(), important_features.keys()), reverse=True)
  print "Features by importance:\n", important_features
  ax = plt.axes()
  for i in range(num_columns):
    plt.arrow(0, 0, xvector[i], yvector[i], color='b', width=0.0005, head_width=0.02, alpha=0.75)
    plt.text(xvector[i]*1.2, yvector[i]*1.2, list(columns)[i], color='b', alpha=0.75)
  return ax
  
def doPCA(data, dimensions=2):
 from sklearn.decomposition import PCA
  model = PCA(n_components=dimensions)
  model.fit(data)
  return model

def scale(data, method):
    # Feature scaling
    if method == 'standard':
        scaled = preprocessing.StandardScaler().fit_transform(data)
    elif method == 'minmax':
        scaled = preprocessing.MinMaxScaler().fit_transform(data)
    elif method == 'maxabs':
        scaled = preprocessing.MaxAbsScaler().fit_transform(data)
    elif method == 'normal':
        scaled = preprocessing.Normalizer().fit_transform(data)
    else:
        scaled = data
    return scaled

    
    
#
# Load, cleansing and transform data
#
  
df = pd.read_csv('C:/Users/lcamp/Documents/user_to_1st_time_investor.csv', index_col=0)

# Take a look!
"""
print df.shape
print df.head()
print df.var()
print df.describe()
print df.dtypes
df.plot.hist()
"""

nclients = df.shape[0]
ninvestors = sum(df.investor_status=='investor')
nnoninvestors = sum(df.investor_status=='non-investor')
print "Number of investors: %d" %ninvestors
print "Number of non-investors: %d" % nnoninvestors

# Transform 'time_to_sign_up_from_first_visit_days' to a categorical binary feature
df['registered'] = df.time_to_sign_up_from_first_visit_days != '-'
df = df.drop(['time_to_sign_up_from_first_visit_days'], axis=1)
# Feature only present in investors
df = df.drop(['first_visit_to_first_investment_days'], axis=1)

# Separate investors and non-investors and clean up

# Drop label, days to sign up and days to first investment columns
inv = df.iloc[:ninvestors,1:]
noinv = df.iloc[ninvestors:,1:]

# Check all features are numeric (except registered)
inv.dtypes
noinv.dtypes

# Remove outliers +-3 std dev
for i in inv.columns:
    if inv.dtypes[i]=='int64':
        inv.loc[:,i] = inv.loc[:,i][np.abs(inv.loc[:,i] - inv.loc[:,i].mean())<=(3*inv.loc[:,i].std())]
        noinv.loc[:,i] = noinv.loc[:,i][np.abs(noinv.loc[:,i] - noinv.loc[:,i].mean())<=(3*noinv.loc[:,i].std())]
inv = inv.dropna()
noinv = noinv.dropna()



#
# Exploratory Analysis
#

# Differences between investors and non-investors
inv.describe()
noinv.describe()

# Box Plot investors vs non-investors
for i in range(0,len(inv.columns)-1):
    plt.figure(i)
    plt.subplot(211)
    plt.title(inv.columns[i])
    plt.ylabel('Investors')
    plt.boxplot(inv[inv.columns[i]], vert=False)
    plt.subplot(212)
    plt.ylabel('Non-investors')
    plt.boxplot(noinv[noinv.columns[i]], vert=False)

plt.figure(i+1)

# Drop non-investors rows so the number of samples of each class is of the same order of magnitude for training
np.random.seed(5)
noinv = noinv.iloc[np.random.permutation(len(noinv))]
noinv = noinv.iloc[:len(inv)*10,:]
                   
# Add label again
inv['investor_status'] = 'investor'
noinv['investor_status'] = 'non-investor'

# Put them together again
X = inv
X = inv.append(noinv)



#
# Dimensionality Reduction
#

# Colors to plot investors vs non-investors
sample_colors = []           
sample_colors = ['green' if i=='investor' else 'red' for i in X.investor_status]

# Save label in a separate variable
y = X.investor_status#.map({'non-investor':0, 'investor':1})
# Drop labels and '1st visit to 1st investment days' because it only affects to investors
X = X.drop(['investor_status'], axis=1)

# Feature scaling
T = scale(X, 'normal')
#T = X # No Change

# Run PCA to reduce dimensionality and plot. This way we might be able to identify most important features
pca = doPCA(T)
T = pca.transform(T)
ax = drawVectors(T, pca.components_, X.columns.values, plt)
T = pd.DataFrame(T)
T.columns = ['component1', 'component2']
T.plot.scatter(x='component1', y='component2', marker='o', c=sample_colors, alpha=0.5, ax=ax)



#
# Train models
#

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

# K-Neighbors
from sklearn.neighbors import KNeighborsClassifier
knmodel = KNeighborsClassifier()
knmodel.fit(X_train, y_train)
kn_score = knmodel.score(X_test, y_test)*100
print "K-Neighbors score: %f %%" % kn_score

# Support Vector Machines
from sklearn.svm import SVC
svcmodel = SVC()
svcmodel.fit(X_train, y_train) 
svc_score = svcmodel.score(X_test, y_test)*100
print "Support Vector Machines score: %f %%" % svc_score

# Decision Tree
from sklearn import tree
treemodel = tree.DecisionTreeClassifier()
treemodel.fit(X_train,y_train)
tree_score = treemodel.score(X_test, y_test)*100
print "Decision Tree score: %f %%" % tree_score

# Random Forest
from sklearn.ensemble import RandomForestClassifier
forestmodel = RandomForestClassifier()
forestmodel.fit(X_train, y_train)
forest_score = forestmodel.score(X_test, y_test)*100
print "Random Forest score: %f %%" % forest_score

# Lets see most important features to know what allow us to discern investors from non-investors
importances = forestmodel.feature_importances_
indices = np.argsort(importances)[::-1]
# Print the feature ranking
print("Random Forest feature ranking:")
for f in range(X.shape[1]):
    print("%d. %s (%f)" % (f + 1, X.columns[indices[f]], importances[indices[f]]))

    
