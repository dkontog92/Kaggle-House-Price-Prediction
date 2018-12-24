# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from datetime import date
from scipy.stats.stats import pearsonr

train_data = pd.read_csv('train.csv',sep=',')
test_data = pd.read_csv('test.csv', sep = ',')
X_data_orig = pd.concat([train_data.iloc[:,:-1], test_data], ignore_index = True)
Y_train = train_data.iloc[:,-1]

print('\n\n\n')

''' 
----------------FEATURE ENGINEERING FOR ALL DATA--------------------
'''
X_data = pd.DataFrame(index=X_data_orig.index)
X_data_orig['KitchenQual'].fillna('TA', inplace=True)
X_data_orig['MasVnrType'].fillna('None', inplace=True)
X_data_orig['BsmtExposure'].fillna('No',inplace=True)
X_data_orig['BsmtFinType1'].fillna('No',inplace=True)

#LotShape has a logical increase thus we replace strings with values 0 1 2 and 3
X_data['LotShape'] = X_data_orig['LotShape'].replace(to_replace={'IR3': 0, 'IR2': 1, 'IR1': 2, 'Reg': 3})

#Replace 'Street', 'Alley' and 'LandContour' columns with indicator columns
street_cols = pd.get_dummies(X_data_orig['Street'],prefix = 'Street')
alley_cols = pd.get_dummies(X_data_orig['Alley'],prefix = 'Alley')
land_cols = pd.get_dummies(X_data_orig['LandContour'],prefix='LandContour')
X_data = pd.concat([X_data, street_cols, alley_cols, land_cols], axis = 1)

#Categorical to numerical the Utilities column
X_data['Utilities'] = X_data_orig['Utilities'].replace(to_replace={'AllPub': 3, 'NoSewr': 2, 'NoSeWa': 1, 'ELO': 0})
#Categorical to numeric the LandSlope column
X_data['LandSlope'] = X_data_orig['LandSlope'].replace(to_replace={'Gtl': 2, 'Mod': 1, 'Sev': 0})

X_data['YearBuilt'] = 2018 - X_data_orig['YearBuilt']
X_data['YearRemodAdd'] = 2018 - X_data_orig['YearRemodAdd']
X_data['ExterQual'] = X_data_orig['ExterQual'].replace(to_replace={'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0})
X_data['ExterCond'] = X_data_orig['ExterCond'].replace(to_replace={'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0})
X_data['BsmtQual'] = X_data_orig['BsmtQual'].replace(to_replace={'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0})
X_data['BsmtCond'] = X_data_orig['BsmtCond'].replace(to_replace={'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0})
X_data['KitchenQual'] = X_data_orig['KitchenQual'].replace(to_replace={'Ex': 3, 'Gd': 2, 'TA': 1, 'Fa': 0})


#Fill nan values in X_data
X_data['BsmtQual'].fillna(0, inplace=True)
X_data['BsmtCond'].fillna(0, inplace=True)

X_data = pd.concat([X_data, X_data_orig[['LotArea','OverallQual','OverallCond','TotalBsmtSF','BedroomAbvGr','KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageArea']]], axis = 1)
 

###-------Exploration for encoding of the remaining variables-----------

def plot_count_box(feature,y_name, data):
    '''Function to plot count and box plots for data exploratio '''
    fig = plt.figure()

    ax1 = fig.add_subplot(2,1,1) 
    sns.countplot(data = data, x = feature, ax = ax1)

    ax2 = fig.add_subplot(2,1,2) 
    sns.boxplot(data = data, x=feature, y=y_name , ax = ax2)
    plt.grid(True)
    plt.show()


#Plots for encoding insight about the "Neighborhood" feature
#plot_count_box('Neighborhood','SalePrice',train_data)
g = train_data.groupby(['Neighborhood'])['SalePrice'].mean()
g.sort_values(ascending=False)

#Encode Neigborhood into a scale from 0 to 6 based on average price per area
neiborhoodDict = {
                  'NoRidge': 6, 'NridgHt': 6, 'StoneBr': 6, 'Timber': 4,
                  'Veenker': 4, 'Somerst': 4, 'ClearCr': 4, 'Crawfor': 4, 
                  'CollgCr': 3, 'Blmngtn': 3, 'Gilbert': 3, 'NWAmes': 3, 
                  'SawyerW': 3, 'Mitchel': 2, 'NAmes': 2, 'NPkVill': 2, 
                  'SWISU': 2, 'Blueste': 2, 'Sawyer': 2, 'OldTown': 1, 
                  'Edwards': 1, 'BrkSide': 1, 'BrDale': 0, 'IDOTRR': 0, 
                  'MeadowV': 0 
                  }
X_data['Neighborhood'] = X_data_orig['Neighborhood'].replace(to_replace=neiborhoodDict)

#MSZoning Variable
#plot_count_box('MSZoning','SalePrice',train_data)

MSZoningDict = {'RL': 3, 'FV': 3, 'RM': 2, 'RH': 2, 'C (all)': 0 }
X_data['MSZoning'] = X_data_orig['MSZoning'].replace(to_replace=MSZoningDict)
X_data[X_data.isnull().any(axis=1)] # Print rows with null/nan values
X_data['MSZoning'] = X_data['MSZoning'].fillna(0)
X_data = X_data.fillna(X_data.mean())


#Roofstyle
#plot_count_box('Exterior1st','SalePrice',train_data)
#HeatingQC
#plot_count_box('HeatingQC','SalePrice',train_data)
X_data['HeatingQC'] = X_data_orig['HeatingQC'].replace(to_replace={'Ex': 3, 'Gd': 2, 'TA': 1, 'Fa': 1, 'Po':0})
#CentralAir
#plot_count_box('CentralAir','SalePrice',train_data)
X_data['CentralAir'] = X_data_orig['CentralAir'].replace(to_replace={'Y': 1, 'N': 0})
#FireplaceQu
#plot_count_box('FireplaceQu','SalePrice',train_data)
X_data['FireplaceQu'] = X_data_orig['FireplaceQu'].replace(to_replace={'Ex': 3, 'Gd': 2, 'TA': 1.5, 'Fa': 1, 'Po':0})
X_data['FireplaceQu'].fillna(0, inplace=True)
#Foundation
#plot_count_box('Foundation','SalePrice',train_data)
foundation_dict = {'PConc': 3,'CBlock': 1.5,'BrkTil': 1,'Slab': 0,'Stone': 2,'Wood': 2}
X_data['Foundation'] = X_data_orig['Foundation'].replace(to_replace=foundation_dict)
#MasVnrType
#plot_count_box('MasVnrType','SalePrice',train_data)
X_data['MasVnrType'] = X_data_orig['MasVnrType'].replace(to_replace={'None': 0, 'BrkCmn': 0, 'BrkFace': 1, 'Stone': 2})
#BsmtExposure
X_data['BsmtExposure'] = X_data_orig['BsmtExposure'].replace(to_replace={'No': 0, 'Gd': 2, 'Mn': 1, 'Av': 1.2})
#BsmtFinType1: 
#plot_count_box('BsmtFinType1','SalePrice',train_data)
X_data['BsmtFinType1'] = X_data_orig['BsmtFinType1'].replace(to_replace={'Unf': 1, 'GLQ': 2, 'ALQ': 1, 'Rec': 1,'BLQ': 1,'LwQ': 1,'No': 0})

#YrSold an MoSold
year_sold = list(X_data_orig['YrSold'])
month_sold = list(X_data_orig['MoSold'])
timings = []
for i in range(len(year_sold)):
    timings.append(date(year_sold[i],month_sold[i],1).toordinal())
timings = np.array(timings) - min(timings)
X_data['SellDate'] = pd.Series(timings)


'''--------------------END OF FEATURE ENGINEERING --------------------------------'''

def RMSE(y,yhat,log=False):
    ''' Mean Squared Error function'''
    if log == False:
        return np.sqrt(np.sum((y-yhat)**2)/len(yhat))
    else:
        return np.sqrt(np.sum((np.log(y)-np.log(yhat))**2)/len(yhat))
        

#Data normalization
X_data_norm = X_data/X_data.max()
X_train_norm = X_data_norm.iloc[:1460,:]
X_test_norm = X_data_norm.iloc[1460:,:] #ONLY TO BE USED FOR SUBMISSION PREDICTIONS


#Split into train and validation sets
X_train2, X_val, Y_train2, Y_val = (train_test_split(X_train_norm, Y_train, 
                                test_size=0.20, random_state = 9))

#RANDOM FOREST
print('\n---Training the Forest----')
regr = RandomForestRegressor(max_depth=20, random_state=1,n_estimators=500,max_features = 'sqrt')
regr.fit(X_train2, Y_train2)
train_pred = regr.predict(X_train2)
val_pred = regr.predict(X_val)
print('\n---Training Finished----\n')
print('Training R^2: '+ str(regr.score(X_train2,Y_train2)))
print('Validation R^2: '+ str(regr.score(X_val,Y_val)))

feat_imp = regr.feature_importances_
ImpFeatures = X_train2.columns[feat_imp.argsort()[-10:][::-1]]  

print('\nTop 10 most important features are (in order):')
for feat in ImpFeatures:
    print(feat)

train_MSE = RMSE(Y_train2, train_pred,log=True)
val_MSE = RMSE(Y_val, val_pred, log=True)
print('Train log root MSE: {:.5f}'.format(train_MSE))
print('Validation log root MSE: {:.5f}'.format(val_MSE))




#----------------------------------------------------------------------------------------------------
#Parameter tuning - Depth of trees 

X_train2, X_val, Y_train2, Y_val = (
        train_test_split(X_train_norm, Y_train, test_size=0.25, random_state = 1))

trainMSE = []
valMSE = []
depths = list(range(1,20,2))

for dep in depths:
    regr = (RandomForestRegressor(max_depth=dep, random_state=1,
                                 n_estimators=500, max_features = 'sqrt'))
    regr.fit(X_train2, Y_train2)
    train_pred = regr.predict(X_train2)
    val_pred = regr.predict(X_val)
    trainMSE.append(RMSE(Y_train2, train_pred, log = True))
    valMSE.append(RMSE(Y_val, val_pred, log = True))

fig = plt.figure()
 
plt.plot(depths,trainMSE,label='Train MSE')
plt.plot(depths,valMSE,label='Test MSE')
plt.title('Root Log Mean Squared Error vs Depth of trees')
plt.legend()
plt.ylabel('RLMSE')
plt.xlabel('Max depth of trees in forest')
plt.grid(which='major')
plt.show()




#-------------------------------------------------------------------------------------
#GRADIENT BOOSTING ALGORITHM

print('\n---Training the Gradient Boosting algo----')
grad_regr = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=300,max_features = None)
grad_regr.fit(X_train2, Y_train2)
grad_train_pred = grad_regr.predict(X_train2)
grad_val_pred = grad_regr.predict(X_val)
print('\n---Training Finished----\n')
print('Training R^2: '+ str(grad_regr.score(X_train2,Y_train2)))
print('Validation R^2: '+ str(grad_regr.score(X_val,Y_val)))

feat_imp = regr.feature_importances_
ImpFeatures = X_train2.columns[feat_imp.argsort()[-10:][::-1]]  

print('\nTop 10 most important features are (in order):')
for feat in ImpFeatures:
    print(feat)

grad_train_MSE = RMSE(Y_train2, grad_train_pred,log=True)
grad_val_MSE = RMSE(Y_val, grad_val_pred, log=True)
print('Train log root MSE: {:.5f}'.format(grad_train_MSE))
print('Validation log root MSE: {:.5f}'.format(grad_val_MSE))


#Parameter tuning - learning rate gradient boosting 

X_train2, X_val, Y_train2, Y_val = (
        train_test_split(X_train_norm, Y_train, test_size=0.25, random_state = 3))

trainMSE = []
valMSE = []
learning_rate = list(np.arange(0.01,0.2,0.01))
depths = list(range(1,15,1))

#Change looping list to optimize the different parameters
for lr in learning_rate:
    grad_regr = GradientBoostingRegressor(loss='ls', learning_rate=lr, n_estimators=400, max_depth = 6)
    grad_regr.fit(X_train2, Y_train2)
    train_pred = grad_regr.predict(X_train2)
    val_pred = grad_regr.predict(X_val)
    trainMSE.append(RMSE(Y_train2, train_pred, log = True))
    valMSE.append(RMSE(Y_val, val_pred, log = True))

fig = plt.figure()
 
plt.plot(learning_rate,trainMSE,label='Train MSE')
plt.plot(learning_rate,valMSE,label='Test MSE')
plt.title('Root Log Mean Squared Error vs Depth of trees')
plt.legend()
plt.ylabel('RLMSE')
plt.xlabel('Learning rate')
plt.grid(which='major')
plt.show()


################################################
#CROSSVALIDATION RANDOM FOREST VS GRADIENT BOOSTING

from sklearn.model_selection import KFold

FOLDS = 10
kf = KFold(n_splits=FOLDS)
#trainMSE_RF = []
valMSE_RF = []
#trainMSE_GB = []
valMSE_GB = []
for train_index, test_index in kf.split(X_train_norm):
    
    #print("TRAIN:", train_index, "TEST:", test_index)
    
    X_train, X_test = X_train_norm.iloc[train_index], X_train_norm.iloc[test_index]
    y_train, y_test = Y_train[train_index], Y_train[test_index]
    regr_RF = (RandomForestRegressor(max_depth=12, random_state=1,
                                 n_estimators=500, max_features = 'sqrt'))
    
    regr_GB = GradientBoostingRegressor(loss='ls', learning_rate=0.12, n_estimators=400, max_depth = 6)
    
    regr_RF.fit(X_train,y_train)
    regr_GB.fit(X_train,y_train)
    
    val_RF_preds = regr_RF.predict(X_test)
    valMSE_RF.append(RMSE(y_test, val_RF_preds, log = True))    
    
    val_GB_preds = regr_GB.predict(X_test)
    valMSE_GB.append(RMSE(y_test, val_GB_preds, log = True))    
    

plt.plot(np.arange(0,FOLDS),valMSE_RF,label='Forest RLMSE')
plt.plot(np.arange(0,FOLDS),valMSE_GB,label='Gradient Boosting RLMSE')
plt.plot(np.arange(0,FOLDS), np.ones(FOLDS)*np.mean(valMSE_RF), label='Mean Forest')
plt.plot(np.arange(0,FOLDS), np.ones(FOLDS)*np.mean(valMSE_GB), label='Mean GradientBoosting')
plt.title(str(FOLDS)+' fold Validation Set RLMSE Forest vs Gradient Boosting')
plt.legend()
plt.ylabel('RLMSE')
plt.xlabel('Folds')
plt.grid(which='major')
plt.show() 
    
    


#----------- TEST DATA PREDICTIONS AND STORE IN submission.csv  ----------------------
#Since Gradient boosting shows better performance in cross validation, we will
#use it to make predictions for the test set. We train a GB algo on the whole
#training dataset.


print('\n---Training the Gradient Boosting algo----')
grad_regr = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=400,max_depth = 6,max_features = None)
grad_regr.fit(X_train_norm, Y_train)
print('\n---Training Finished----\n')


test_preds = pd.DataFrame(grad_regr.predict(X_test_norm))
test_preds.index += 1461
test_preds.columns = ['SalePrice']
test_preds.to_csv("submission.csv",sep=';')




########################################################################
#Important functions
'''
X_train['Street'] = X_train['Street'].astype('category')
X_train['MSSubClass'].value_counts()
pd.get_dummies(obj_df, columns=["body_style", "drive_wheels"], prefix=["body", "drive"]).head()

Check for null values X_train['LandContour'].isnull().values.any()

objects = train_data.select_dtypes(include=['object']).copy()

#Returns df with null values
objects[objects.isnull().any(axis=1)]


df_objects=X_data_orig.select_dtypes(include=['object']).copy()
for i in df_objects.columns:
    plot_count_box(i,'SalePrice',train_data)

'''





