#!/usr/bin/env python
# coding: utf-8




import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import contractions   #Fixes contractions such as `you're` to you `are`
import regex as re    #regular expression module
import nltk           #Natural Language Toolkit
from nltk.corpus import stopwords #such as “the”, “a”, “an”, “in”)
from nltk.stem import WordNetLemmatizer 
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer, minmax_scale
from scipy.sparse import hstack
from sklearn.linear_model import Ridge
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, Lasso
import lightgbm as lgb





#reading data
train  = pd.read_csv('./train.tsv',sep='\t')
test  = pd.read_csv('./test.tsv',sep='\t')




test.sort_values('id',inplace=True)
train.sort_values('train_id',inplace=True)





#drop all datapoint with price as 0
train.drop(train[train['price']==0.0].index,inplace=True)





train.info()




test.info()


# ### train test split



train, cv = train_test_split(train, test_size = 0.20, random_state = 21)


# ### Filling Null values



#fill item description as 'No description yet', category as 'others/others/others', brand_name as locals
train.item_description.fillna('No description yet',inplace=True)
test.item_description.fillna('No description yet',inplace=True)
cv.item_description.fillna('No description yet',inplace=True)
train.category_name.fillna('others/others/others',inplace=True)
test.category_name.fillna('others/others/others',inplace=True)
cv.category_name.fillna('others/others/others',inplace=True)
train.brand_name.fillna('locals',inplace=True)
test.brand_name.fillna('locals',inplace=True)
cv.brand_name.fillna('locals',inplace=True)



#splitting category by delimeter '/' to make new feature cat_level1, cat_level2, cat_level3
train['cat_level1'],train['cat_level2'],train['cat_level3'] = zip(*train.category_name.apply(lambda x: x.split('/')))
test['cat_level1'],test['cat_level2'],test['cat_level3'] = zip(*test.category_name.apply(lambda x: x.split('/')))
cv['cat_level1'],cv['cat_level2'],cv['cat_level3'] = zip(*cv.category_name.apply(lambda x: x.split('/')))


# ### Text Preprocessing



stop_words = set(stopwords.words('english'))-{"no","nor","not"} #such as “the”, “a”, “an”, “in”)





'''Text proceesing 
1) Converting all words to lowercase.
2) Removal of stop words
3) Removing punctuation and special characters.
4) Removing unwanted multiple spaces
5)Handling Alpha-numeric values and so on.
6) lemmatizing
'''
from nltk.stem import PorterStemmer
ps = PorterStemmer()
def text_prep(val):
    val = contractions.fix(val)
    val = val.replace('\\r',' ')
    val = val.replace('\\"',' ')
    val = val.replace('\\n',' ')
    val = re.sub('[^A-Za-z0-9]+',' ',val)
    val = ' '.join(ps.stem(v) for v in val.split() if v.lower() not in stop_words)
    val = val.lower().strip()
    return val





#text processing name and item description
train['proc_item_description'] = train.item_description.apply(text_prep)
test['proc_item_description'] = test.item_description.apply(text_prep)
cv['proc_item_description'] = cv.item_description.apply(text_prep)

train['proc_name'] = train.name.apply(text_prep)
test['proc_name'] = test.name.apply(text_prep)
cv['proc_name'] = cv.name.apply(text_prep)





#mergeing feature processed item description, name and three levels of category to make a new feature 'text' and we will produce ‘new name’ by merging name and brand name 
train["text"]=train["proc_item_description"]+" "+train["proc_name"]+' '+train['cat_level1']+" "+train['cat_level2']+" "+train['cat_level3']
train['newname']=train['proc_name']+' '+train['brand_name']
test["text"]=test["proc_item_description"]+" "+test["proc_name"]+' '+test['cat_level1']+" "+test['cat_level2']+" "+test['cat_level3']
test['newname']=test['proc_name']+' '+test['brand_name']
cv["text"]=cv["proc_item_description"]+" "+cv["proc_name"]+' '+cv['cat_level1']+" "+cv['cat_level2']+" "+cv['cat_level3']
cv['newname']=cv['proc_name']+' '+cv['brand_name']


# ### Word Embedding




#Embedding 'text' feature tfidf
vectorizer = TfidfVectorizer(ngram_range=(1,2),min_df=30,max_features=50000)
vectorizer.fit(train['text'].values)
train_text = vectorizer.transform(train['text'].values)
test_text = vectorizer.transform(test['text'].values)
cv_text = vectorizer.transform(cv['text'].values)




#Count vectorizing newname feature
vectorizer = CountVectorizer()
vectorizer.fit(train['newname'].values)
train_newname = vectorizer.transform(train['newname'].values)
test_newname = vectorizer.transform(test['newname'].values)
cv_newname = vectorizer.transform(cv['newname'].values)


# ### Encoding



#one hot encoding item_condition and shipping
train_item_cond = csr_matrix(pd.get_dummies(train[['item_condition_id','shipping']],
                                          sparse=True).values)
test_item_cond = csr_matrix(pd.get_dummies(test[['item_condition_id','shipping']],
                                          sparse=True).values)
cv_item_cond = csr_matrix(pd.get_dummies(cv[['item_condition_id','shipping']],
                                          sparse=True).values)


# ### Preparing Features

#hstack all data
X_train_data = hstack((train_newname,train_text,train_item_cond))
Y_train_data = np.log(train['price'].values+1)
Y_cv_data = np.log(cv['price'].values+1)
X_test_data = hstack((test_newname,test_text,test_item_cond))
X_cv_data = hstack((cv_newname,cv_text,cv_item_cond))



def rmsle(a,p):
    s = np.sqrt(np.sum(np.square(np.log(p+1)-np.log(a+1)))/p.shape[0])
    return s


# ### Ridge
print("tarining ridge regression")
#grid search for ridge best parameter 
parameters = {'alpha':[0.0001,0.001,0.01,0.1,1,10,100,1000,10000],
              'fit_intercept' : [False],
              'solver' : ['lsqr','sag']}

gs_ridge = GridSearchCV(estimator = Ridge(),
                        param_grid = parameters,
                        cv = 3, 
                        scoring = 'neg_mean_squared_error',
                        verbose = 100,
                        return_train_score = True,
                        n_jobs = -2)
gs_ridge.fit(X_train_data, Y_train_data)

print("Best Estimator ",gs_ridge.best_estimator_)


#training ridge
rig = Ridge(alpha = 1,solver = "sag", fit_intercept=False)
rig.fit(X_train_data, Y_train_data)
pred = rig.predict(X_cv_data)
print("RMSLE: ridge",rmsle(cv.price,np.expm1(pred)))


# ### Lasso
print("training lasso")
#grid search for best parameter
parameters = {'alpha':[0.0001,0.001,0.01,0.1,1,10,100,1000,10000],
              'fit_intercept' : [False],
              }

gs_lasso = GridSearchCV(estimator = Lasso(),
                        param_grid = parameters,
                        cv = 3, 
                        scoring = 'neg_mean_squared_error',
                        verbose = 100,
                        return_train_score = True,
                        n_jobs = -2)
gs_lasso.fit(X_train_data, Y_train_data)

print("Best Estimator ",gs_lasso.best_estimator_)


# training lasso
rig = Lasso(alpha = 0.0001, fit_intercept=False)
rig.fit(X_train_data, Y_train_data)
pred = rig.predict(X_cv_data)
print("RMSLE: lasso",rmsle(cv.price,np.expm1(pred)))


# ### lightbgm


print("training light gbm")

lgbm_params_1 = {'n_estimators': 5000,
                 'max_depth': 40,
                 'num_leaves':100,
                 'n_jobs': -1}
model = lgb.LGBMRegressor(**lgbm_params_1)




#training light gbm
model.fit(X_train_data, Y_train_data)
pred = model.predict(X_cv_data)
print("RMSLE: light gbm",rmsle(cv.price,np.expm1(pred)))




#predicting value
# Y_pred =  model.predict(X_test_data)
# sub = {"id":test.id.values,"price":np.expm1(Y_pred)}
# submission = pd.DataFrame(sub)
# submission.to_csv("submission431.csv",index=False,header=True)

