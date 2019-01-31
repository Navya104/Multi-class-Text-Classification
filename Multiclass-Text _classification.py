#importing required packages
import pandas as pd
import numpy as np
from sklearn.feature_extraction import text
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn import decomposition, ensemble
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

#importing train_data.csv and train_label.csv as dataframes
df1 = pd.read_csv("train_data.csv")
df2 = pd.read_csv("train_label.csv")
#Denoising dataframes
df2.sort_values("id", inplace = True)
df2.drop_duplicates(subset ="id",keep = False, inplace = True) 
df1.sort_values("id", inplace = True)
df1.drop_duplicates(subset ="id",keep = False, inplace = True)
#Merging datasets
merged = df1.merge(df2, on="id", how="outer").fillna("")
merged.to_csv("merged.csv", index=False)        
data=pd.read_csv("merged.csv")
data=data.dropna()
#Remove stopwords in merged text data
stop = text.ENGLISH_STOP_WORDS
data['text_without_stopwords'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
#Creating a dataframe with text,id,label without any noise
data=pd.concat([data['text_without_stopwords'],data['id'],data['label']],axis=1, keys=['text', 'id','label'])
print(data)
#splitting data
x=data.iloc[:,0:1]
y=data.iloc[:,2:3]
#converting text column into list
x_list=data["text"].tolist()
#Converting label into list and Encoding 
y_list=data["label"].tolist()
S = np.array(y_list)
le = LabelEncoder()
S= le.fit_transform(S)
# split the dataset into training and validation datasets 
from sklearn.model_selection import train_test_split
train_x, valid_x, train_y, valid_y = train_test_split(x_list, S,test_size = 0.2,shuffle=False)
#Feature extraction from text 
#Word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(x_list)
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)
#N-gram level tf-idf
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(x_list)
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)
# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(x_list)
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x) 
#Training the model
def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:

        predictions = predictions.argmax(axis=-1)
    
    return metrics.accuracy_score(predictions, valid_y)
#NaivesBayes Algorithm
#Naive Bayes on Word-level TF-IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf,train_y,xvalid_tfidf)
print("NB, WordLevel TF-IDF: ", accuracy) #0.80
# Naive Bayes on Ngram Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("NB, N-Gram Vectors: ", accuracy) #0.67
# Naive Bayes on Character Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print("NB, CharLevel Vectors: ", accuracy) #0.71


#Logistic Regression
# Linear Classifier on Word Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf)
print("LR, WordLevel TF-IDF: ", accuracy)#0.87
# Linear Classifier on Ngram Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("LR, N-Gram Vectors: ", accuracy)#0.74
# Linear Classifier on Character Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print("LR, CharLevel Vectors: ", accuracy)#0.83

#DecisionTree Classifier
#DecisionTreeClassifier on Word Level TF IDF Vectors
accuracy = train_model(DecisionTreeClassifier(), xtrain_tfidf, train_y, xvalid_tfidf)
print("DT, WordLevel TF-IDF: ", accuracy)#0.69
#DecisionTreeClassifier on Ngram Level TF IDF Vectors
accuracy = train_model(DecisionTreeClassifier(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("DT, N-Gram Vectors: ", accuracy)#0.61
#decisionTreeClassifier on Character Level TF IDF Vectors
accuracy = train_model(DecisionTreeClassifier(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print("DT, CharLevel Vectors: ", accuracy)#0.61

#LogisticRegression with WordLevel TF IDF vectors got more accuracy,We'll forward to see Performance Metrics

from sklearn.metrics import classification_report, confusion_matrix 
def performance_metrics(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
        
    #return confusion_matrix(predictions, valid_y)
    return classification_report(predictions, valid_y)  
#Logistic
classification_report = performance_metrics(LogisticRegression(), xtrain_tfidf,train_y,xvalid_tfidf)
print(classification_report)
#precision-0.89
#Recall-0.88
#F1-score-0.88
#support-4051

#Ensemble Methods #Bagging
# RF on Word Level TF IDF Vectors
seed=7
num_trees = 100
max_features = 3
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = ensemble.RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = model_selection.cross_val_score(model, xtrain_tfidf,train_y, cv=kfold)
print(results.mean()) #0.88

#Accuracy is more for Logistic_Regression and performance also good so we can use Logistic_Regression Model to get the probabilities for test_data.csv
#Test data
test=pd.read_csv("test_data.csv")
test.sort_values("id", inplace = True)
test.drop_duplicates(subset ="id",keep = False, inplace = True) #denoising
x1_list=test["text"].tolist()
#Word level tf-idf for test data
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(x1_list)
x1_tfidf =  tfidf_vect.transform(x1_list)
#Logistic Regression
from sklearn.linear_model import LogisticRegression
clf1=LogisticRegression()
clf1.fit(xtrain_tfidf,train_y)
pred1= clf1.predict_proba(x1_tfidf)

#Exporting pred values into csv as pred.csv
pred=pd.DataFrame(pred)
pred.to_csv("pred.csv",)
pred.columns = ['Indoor/Outdoor', 'Commercial / Residential',
       'ENERGY STAR Certified', 'Hardware Included', 'Package Quantity',
       'Flooring Product Type', 'Color', 'Tools Product Type', 'Included',
       'Voltage (volts)', 'Assembly Required', 'Features', 'Wattage (watts)',
       'Finish', 'Shape']
df=pd.read_csv("sample_submission.csv")
df[['Indoor/Outdoor', 'Commercial / Residential',
       'ENERGY STAR Certified', 'Hardware Included', 'Package Quantity',
       'Flooring Product Type', 'Color', 'Tools Product Type', 'Included',
       'Voltage (volts)', 'Assembly Required', 'Features', 'Wattage (watts)',
       'Finish', 'Shape']] = pred[['Indoor/Outdoor', 'Commercial / Residential',
       'ENERGY STAR Certified', 'Hardware Included', 'Package Quantity',
       'Flooring Product Type', 'Color', 'Tools Product Type', 'Included',
       'Voltage (volts)', 'Assembly Required', 'Features', 'Wattage (watts)',
       'Finish', 'Shape']].values
df.to_csv("output_sample_submission.csv")
