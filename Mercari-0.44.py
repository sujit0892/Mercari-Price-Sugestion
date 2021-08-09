#!/usr/bin/env python
# coding: utf-8

# ### IMPORTING


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)






import contractions
import regex as re
import nltk
from nltk.corpus import stopwords
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





#train test split
train, cv = train_test_split(train, test_size = 0.20, random_state = 21)


# ### Filling NA


#creating set of brand_name
brand_set = set(train['brand_name'].values)





#filling all null value as missing
#brand value we will use name to fill it and the value which remains null after this fill as ‘missing’. 
train.item_description.fillna('missing',inplace=True)
test.item_description.fillna('missing',inplace=True)
cv.item_description.fillna('missing',inplace=True)
train.category_name.fillna('missing/missing/missing',inplace=True)
test.category_name.fillna('missing/missing/missing',inplace=True)
cv.category_name.fillna('missing/missing/missing',inplace=True)
train.brand_name.fillna('missing',inplace=True)
test.brand_name.fillna('missing',inplace=True)
cv.brand_name.fillna('missing',inplace=True)
def missing_brand(features):
    brand = features[0]
    name = features[1]
    if brand == 'missing':
        for word in name.split():
            if word in brand_set:
                return word
    if name in brand_set:
        return name
    return brand
train['brand_name'] = train[['brand_name', 'name']].apply(missing_brand, axis=1)
test['brand_name'] = test[['brand_name', 'name']].apply(missing_brand, axis=1)
cv['brand_name'] = cv[['brand_name', 'name']].apply(missing_brand, axis=1)





#replacing No description yet with missing
train.item_description.replace("No description yet","missing",inplace=True)
test.item_description.replace("No description yet","missing",inplace=True)
cv.item_description.replace("No description yet","missing",inplace=True)




#split category in 3 level with delimeter '/'
train['cat_level1'],train['cat_level2'],train['cat_level3'] = zip(*train.category_name.apply(lambda x: x.split('/')))
test['cat_level1'],test['cat_level2'],test['cat_level3'] = zip(*test.category_name.apply(lambda x: x.split('/')))
cv['cat_level1'],cv['cat_level2'],cv['cat_level3'] = zip(*cv.category_name.apply(lambda x: x.split('/')))





#calculating length of name
train["name_len"]=train.name.apply(lambda x:len(x.split(" ")))
test["name_len"]=test.name.apply(lambda x:len(x.split(" ")))
cv["name_len"]=cv.name.apply(lambda x:len(x.split(" ")))





def text_len(x):
    if(x=='missing'):
        return 0
    else:
        return len(x.split(" "))





#creating length of description
train["des_len"]=train.item_description.apply(text_len)
test["des_len"]=test.item_description.apply(text_len)
cv["des_len"]=cv.item_description.apply(text_len)


# ### Sentiment Analysis




#calculating sentimate score
sid = SentimentIntensityAnalyzer()
train['negative'],train['neutral'],train['positive'],train['compound'] = zip(*train.item_description.apply(lambda x: sid.polarity_scores(x).values()))
test['negative'],test['neutral'],test['positive'],test['compound'] = zip(*test.item_description.apply(lambda x: sid.polarity_scores(x).values()))
cv['negative'],cv['neutral'],cv['positive'],cv['compound'] = zip(*cv.item_description.apply(lambda x: sid.polarity_scores(x).values()))


# ### Word Embedding




#count vectorizing brand name, cat_level1, cat_level2, cat_level_3, name,
# tfidf to item description




vectorizer = CountVectorizer()
vectorizer.fit(train['brand_name'].values)
train_brand_name = vectorizer.fit_transform(train['brand_name'].values)
test_brand_name = vectorizer.transform(test['brand_name'].values)
cv_brand_name = vectorizer.transform(cv['brand_name'].values)





vectorizer = CountVectorizer()
vectorizer.fit(train['cat_level1'].values)
train_cat_level1 = vectorizer.fit_transform(train['cat_level1'].values)
test_cat_level1 = vectorizer.transform(test['cat_level1'].values)
cv_cat_level1 = vectorizer.transform(cv['cat_level1'].values)




vectorizer = CountVectorizer()
vectorizer.fit(train['cat_level2'].values)
train_cat_level2 = vectorizer.fit_transform(train['cat_level2'].values)
test_cat_level2 = vectorizer.transform(test['cat_level2'].values)
cv_cat_level2 = vectorizer.transform(cv['cat_level2'].values)





vectorizer = CountVectorizer()
vectorizer.fit(train['cat_level3'].values)
train_cat_level3 = vectorizer.fit_transform(train['cat_level3'].values)
test_cat_level3 = vectorizer.transform(test['cat_level3'].values)
cv_cat_level3 = vectorizer.transform(cv['cat_level3'].values)





vectorizer = CountVectorizer()
vectorizer.fit(train['name'].values)
train_name = vectorizer.fit_transform(train['name'].values)
test_name = vectorizer.transform(test['name'].values)
cv_name = vectorizer.transform(cv['name'].values)





vectorizer = TfidfVectorizer(ngram_range=(1,2),min_df = 10,max_features=100000,dtype=np.float32)
vectorizer.fit(train['item_description'].values)
train_item_des = vectorizer.transform(train['item_description'].values)
test_item_des = vectorizer.transform(test['item_description'].values)
cv_item_des = vectorizer.transform(cv['item_description'].values)


# ### Encoding



# reshape all numerical in (-1,1)



train_neg = train['negative'].values.reshape(-1,1)
train_pos = train['positive'].values.reshape(-1,1)
train_neu = train['neutral'].values.reshape(-1,1)
train_comp = train['compound'].values.reshape(-1,1)
train_des_len = train['des_len'].values.reshape(-1,1)
train_name_len = train['name_len'].values.reshape(-1,1)




test_neg = test['negative'].values.reshape(-1,1)
test_pos = test['positive'].values.reshape(-1,1)
test_neu = test['neutral'].values.reshape(-1,1)
test_comp = test['compound'].values.reshape(-1,1)
test_des_len = test['des_len'].values.reshape(-1,1)
test_name_len = test['name_len'].values.reshape(-1,1)




cv_neg = cv['negative'].values.reshape(-1,1)
cv_pos = cv['positive'].values.reshape(-1,1)
cv_neu = cv['neutral'].values.reshape(-1,1)
cv_comp = cv['compound'].values.reshape(-1,1)
cv_des_len = cv['des_len'].values.reshape(-1,1)
cv_name_len = cv['name_len'].values.reshape(-1,1)





train_item_cond = csr_matrix(pd.get_dummies(train[['item_condition_id','shipping']],
                                          sparse=True).values)
test_item_cond = csr_matrix(pd.get_dummies(test[['item_condition_id','shipping']],
                                          sparse=True).values)
cv_item_cond = csr_matrix(pd.get_dummies(cv[['item_condition_id','shipping']],
                                          sparse=True).values)


# ### Normalizer


#Normalization of all numerical feature



normalizer = Normalizer()
normalizer.fit(train_neg)
train_neg = normalizer.transform(train_neg)
test_neg = normalizer.transform(test_neg)
cv_neg = normalizer.transform(cv_neg)




normalizer = Normalizer()
normalizer.fit(train_pos)
train_pos = normalizer.transform(train_pos)
test_pos = normalizer.transform(test_pos)
cv_pos = normalizer.transform(cv_pos)



normalizer = Normalizer()
normalizer.fit(train_neu)
train_neu = normalizer.transform(train_neu)
test_neu = normalizer.transform(test_neu)
cv_neu = normalizer.transform(cv_neu)




normalizer = Normalizer()
normalizer.fit(train_comp)
train_comp = normalizer.transform(train_comp)
test_comp = normalizer.transform(test_comp)
cv_comp = normalizer.transform(cv_comp)





normalizer = Normalizer()
normalizer.fit(train_name_len)
train_name_len = normalizer.transform(train_name_len)
test_name_len = normalizer.transform(test_name_len)
cv_name_len = normalizer.transform(cv_name_len)



normalizer = Normalizer()
normalizer.fit(train_des_len)
train_des_len = normalizer.transform(train_des_len)
test_des_len = normalizer.transform(test_des_len)
cv_des_len = normalizer.transform(cv_des_len)


# ### Preparing Features for model




X_train_data = hstack((train_name,train_brand_name,train_cat_level1,train_cat_level2,train_cat_level3,
                    train_item_cond,train_item_des,train_des_len,train_name_len,train_neg, train_pos, train_neu, train_comp)).tocsr()




Y_train_data = np.log(train['price'].values+1)





Y_cv_data = np.log(cv['price'].values+1)





X_test_data = hstack((test_name,test_brand_name,test_cat_level1,test_cat_level2,test_cat_level3,
                    test_item_cond,test_item_des,test_des_len,test_name_len,test_neg, test_pos, test_neu, test_comp)).tocsr()





X_cv_data = hstack((cv_name,cv_brand_name,cv_cat_level1,cv_cat_level2,cv_cat_level3,
                    cv_item_cond,cv_item_des,cv_des_len,cv_name_len,cv_neg, cv_pos, cv_neu, cv_comp)).tocsr()



def rmsle(a,p):
    s = np.sqrt(np.sum(np.square(np.log(p+1)-np.log(a+1)))/p.shape[0])
    return s


# ### Ridge



print("training ridge regression")
#grid search for best parameter
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




rig = Ridge(alpha = 10,solver = "sag", fit_intercept=False)
rig.fit(X_train_data, Y_train_data)
pred = rig.predict(X_cv_data)
print("RMSLE: ridge",rmsle(cv.price,np.expm1(pred)))


# ### lightgbm


print("training lightgbm")
lgbm_params_1 = {'n_estimators': 5000,
                 'max_depth': 40,
                 'num-leaves' : 100,
                 'n_jobs': -1}




import lightgbm as lgb
model = lgb.LGBMRegressor(**lgbm_params_1)
model.fit(X_train_data, Y_train_data)
pred = model.predict(X_cv_data)




print(rmsle(np.expm1(Y_cv_data),np.expm1(pred)))









# Y_pred =  rig.predict(X_test_data)
# sub = {"id":test.id.values,"price":np.expm1(Y_pred)}
# submission = pd.DataFrame(sub)
# submission.to_csv("submission.csv",index=False,header=True)







