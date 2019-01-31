# Multi-class-Text-Classification
#python 
-noise removal 
- data visualization 
- model fitting 
- performance metric evaluation 
- analyzing mistakes and improving the model

Here,I explained these terms for the Multi-Class-Text-Classification Task.

Noise Removal:

	Noise Removal is defined as cleaning the data.It includes
	1.Removing redundant values
		I removed the duplicate values from the columns of text,id,label from the given csv files.
	2.Removing stopwords
		In NLP,useless data is referred to as StopWords.A stop word is a commonly used word (such as “the”, “a”, “an”, “in”) that a search engine has been programmed to ignore, both when 
		indexing entries for searching and when retrieving them as the result of a search query.
		We would not want these words taking up space in our database, or taking up valuable processing time. For this, we can remove them easily, by storing a list of words that you consider to be stop words.
	3.Removing Null values
		Removed the null values from the coulmns.
	After finishing these 3 tasks, our dataframe size has reduced.Now the dataframe is denoised.

Data Visualization:
	Data consists of 3 columns(text,id and label).It is a Multiclass Text Classification data.But the computer can't understand the text.So,I convert the text column and
	Label columns into vector.Done the Label encoder for label vector.It has 15 labels(0-14).
	Extract the features from Text column vector by using Word-level,N-gram level,Character level tf-idf vectors.

Model Fitting:
	Model Fitting is picking the best model that best describes the data.
	By seeing the data,I came to know that we have to use linear models to map a set of values to a set of Normal distributions.
	This task is based on Multi-class text classification.There are many classification algorithms in Machine Learning like Logistic Regression,Naive Bayes,SVM,Decision Tree.Here,
	I performed classification using all algorithms.Logistic Regression Model gives more accuracy(0.87).So,For the test data to predict probabilites of the labels for the 
	text I used Logistic Regression.
           Logistic Regression:
		It isn't a regression algorithm but a probabilistic classification model.Logistic Regression has a Sigmoid curve(F(X)=1/1+e^-X).It gives the output
		ranges between 0 and 1. 

Performance Metric Evaluation:
	Here,I evaluate the performance of the model using Precision,Recall,F1 score and Support.
	Precision: Refers to Positive Predictive Value.Exactness, what % of tuples that the classifier labeled as positive are actually positive.
	Recall:Completeness – what % of positive tuples did the classifier label as positive.

Analyzing mistakes and improving the Model:
	We can improve the model using Ensemble Methods like Bagging and Boosting.
	Bagging(Bootstrap Aggregation):Majority vote or averaging the prediction over a collection of classifiers.I used RandomForest for improving the model.
				   It gives the accuracy 0f 0.88.
	Boosting:Weighted vote or averaging with a collection of classifiers.
