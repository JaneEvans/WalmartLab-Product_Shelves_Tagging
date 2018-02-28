import pandas as pd
import numpy as np
import csv

train_df = pd.read_csv('./train.tsv', sep = '\t',header=0)
test_df = pd.read_csv('./test.tsv', sep = '\t',header=0)
# print(train_df.info())
# print(test_df.info())

## ======================================================================
## Data Preprocessing 
## ==================

## Select predictors -----
train_df = train_df[['item_id', 'Item Class ID', 'Aspect Ratio', 'MPAA Rating','Recommended Use','Product Name', 'Product Long Description' , 'Product Short Description', 'Short Description', 'Actual Color', 'Color', 'actual_color', 'Genre ID', 'tag']]
train_df['Product Long Description'].fillna('NoProductLongDescription', inplace=True)
train_df['Product Name'].fillna('NoProductName', inplace=True)
train_df['New Product Name'] = 'Item' + train_df['Item Class ID'].astype(str) + ' ' + 'GenreID' + train_df['Genre ID'].astype(str) + \
							' ' + train_df['Actual Color'].astype(str) + ' ' + train_df['Color'].astype(str) + ' ' + train_df['actual_color'].astype(str) + \
							' ' + 'AspectRatio' + train_df['Aspect Ratio'].astype(str) + ' ' + 'MPAARating' + train_df['MPAA Rating'].astype(str) + \
							' ' + train_df['Recommended Use'].astype(str) + \
							' ' + train_df['Product Name'].astype(str) 
train_df['Product Short Description'].fillna('NoProductShortDescription', inplace=True)
train_df['Short Description'].fillna('NoShortDescription', inplace=True)
train_df['Combined Short Description'] = 'Item' + train_df['Item Class ID'].astype(str) +' '+ train_df['Product Short Description'].astype(str) + ' ' + train_df['Short Description'].astype(str) + \
										' ' + 'GenreID' + train_df['Genre ID'].astype(str) 
train_df = train_df.drop(['Item Class ID', 'Aspect Ratio', 'MPAA Rating','Recommended Use', 'Product Short Description', 'Short Description', 'Actual Color', 'Color', 'actual_color', 'Genre ID'], axis = 1)
# print(train_df.head())
# print(train_df.info())

test_df = test_df[['item_id', 'Item Class ID', 'Aspect Ratio', 'MPAA Rating','Recommended Use','Product Name', 'Product Long Description' , 'Product Short Description', 'Short Description', 'Actual Color', 'Color', 'actual_color', 'Genre ID']]
test_df['Product Long Description'].fillna('NoProductLongDescription', inplace=True)
test_df['Product Name'].fillna('NoProductName', inplace=True)
test_df['New Product Name'] = 'Item' + test_df['Item Class ID'].astype(str) + ' ' + 'GenreID' + test_df['Genre ID'].astype(str) + \
							' ' + test_df['Actual Color'].astype(str) + ' ' + test_df['Color'].astype(str) + ' ' + test_df['actual_color'].astype(str) + \
							' ' + 'AspectRatio' + test_df['Aspect Ratio'].astype(str) + ' ' + 'MPAARating' + test_df['MPAA Rating'].astype(str) + \
							' ' + test_df['Recommended Use'].astype(str) + \
							' ' + test_df['Product Name'].astype(str) 							
test_df['Product Short Description'].fillna('NoProductShortDescription', inplace=True)
test_df['Short Description'].fillna('NoShortDescription', inplace=True)
test_df['Combined Short Description'] = 'Item' + test_df['Item Class ID'].astype(str) +' '+ test_df['Product Short Description'].astype(str) + ' ' + test_df['Short Description'].astype(str) + \
										' ' + 'GenreID' + test_df['Genre ID'].astype(str) 
test_df = test_df.drop(['Item Class ID', 'Aspect Ratio', 'MPAA Rating','Recommended Use', 'Product Short Description', 'Short Description', 'Actual Color', 'Color', 'actual_color', 'Genre ID'], axis = 1)
# print(test_df.head())
# print(test_df.info())


## Convert "tag" to 32 binary variables -----
def convertTags(tagRow):
	return tagRow.replace('[', '').replace(']','').replace(' ','').split(',')
train_df['tagList'] = train_df['tag'].map(lambda x: convertTags(x))

taglist = []
for tags in train_df['tagList']:
	for tag in tags:
		taglist.append(tag)

from collections import Counter
taglist = (Counter(taglist))

def binTag(tagr,tagRow):
	if tagr in tagRow:
		return 1
	else: 
		return 0

for _tag in taglist:
	train_df[_tag] = train_df['tagList'].map(lambda x: binTag(_tag, x))

## Prepare to extract features from texts -----
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()


## ==================================================================================================
## Modeling: Multi-Layer Perceptron Classifier
## ===========================================
from sklearn.neural_network import MLPClassifier

## ------------------------------------------------------------------------
## First run: Use "New Product Name" to predict Tag for all products ------
print('## First Run Results: -----------------------------------------')

## Extracting features from "Product Name" -----
NewProductName_train_counts = count_vect.fit_transform(train_df['New Product Name'])
NewProductName_test_counts = count_vect.transform(test_df['New Product Name'])
NewProductName_train_tfidf = tfidf_transformer.fit_transform(NewProductName_train_counts)
NewProductName_test_tfidf = tfidf_transformer.transform(NewProductName_test_counts)
print('Train NewPN-tfidf shape:', NewProductName_train_tfidf.shape)
print('Test NewPN-tfidf shape:', NewProductName_test_tfidf.shape)
df_train_NewPN = NewProductName_train_tfidf
df_test_NewPN = NewProductName_test_tfidf

def predictTags_byNewPN(tagID):

	##-- Test model performance by cross-varlidations
	# from sklearn import cross_validation
	# df_train, df_test, label_train, label_test = cross_validation.train_test_split(df_train_NewPN, train_df[tagID], test_size=0.10, random_state = 0)
	##-----------------------------------------------

	label_train = train_df[tagID]
	df_train = df_train_NewPN
	clf = MLPClassifier(activation = 'relu', solver = 'lbfgs', alpha = 0.1, hidden_layer_sizes = (50, ), random_state = 1)
	clf.fit(df_train, label_train)

	# return clf.score(df_test, label_test)
	return clf.predict(df_test_NewPN)

test_predict = test_df.drop(['New Product Name'], axis = 1)

## Predict 32 Tags one by one ---
for tag_id in taglist:
	test_predict[tag_id] = predictTags_byNewPN(tag_id) * int(tag_id)


test_predict = test_predict.replace({0:np.nan})
print(test_predict.info())
test_predict['tag'] = test_predict[test_predict.columns[4:]].apply(lambda x: ', '.join(x.dropna().astype(int).astype(str)),axis=1)
test_predict = test_predict.drop(taglist.keys(), axis = 1)
test_predict['tag'] = test_predict['tag'].map(lambda x: '['+str(x)+']')
print(test_predict[test_predict['tag'] == '[]'].count())

test_df2 = test_predict[test_predict['tag'] == '[]'].drop(['tag'], axis = 1)
test_predict1 = test_predict[test_predict['tag'] != '[]'].drop(['Product Name','Product Long Description', 'Combined Short Description'], axis = 1)
print(test_df2.info())


## --------------------------------------------------------------------------------------
## Second run: Use "Product Long Description" to predict tag for untagged products ------
print('## Second Run Results: -----------------------------------------')

## Extracting features from "Product Long Description" -----
ProductLongDescription_train_counts = count_vect.fit_transform(train_df['Product Long Description'])
ProductLongDescription_test_counts = count_vect.transform(test_df2['Product Long Description'])
ProductLongDescription_train_tfidf = tfidf_transformer.fit_transform(ProductLongDescription_train_counts)
ProductLongDescription_test_tfidf = tfidf_transformer.transform(ProductLongDescription_test_counts)
print('Train PLD-tfidf shape:', ProductLongDescription_train_tfidf.shape)
print('Test PLD-tfidf shape:', ProductLongDescription_test_tfidf.shape)
df_train_PLD = ProductLongDescription_train_tfidf
df_test_PLD = ProductLongDescription_test_tfidf

def predictTags_byPLD(tagID):

	##-- Test model performance by cross-varlidations
	# from sklearn import cross_validation
	# df_train, df_test, label_train, label_test = cross_validation.train_test_split(df_train_PLD, train_df[tagID], test_size=0.10, random_state = 0)
	## ----------------------------------------------

	label_train = train_df[tagID]
	df_train = df_train_PLD
	clf = MLPClassifier(activation = 'relu', solver = 'lbfgs', alpha = 0.1, hidden_layer_sizes = (50, ), random_state = 1)
	clf.fit(df_train, label_train)

	return clf.predict(df_test_PLD)
	# return clf.score(df_test, label_test)

test_predict = test_df2.drop(['Product Long Description'], axis = 1)

## Predict 32 Tags one by one ---
for tag_id in taglist:
	test_predict[tag_id] = predictTags_byPLD(tag_id) * int(tag_id)

test_predict = test_predict.replace({0:np.nan})
print(test_predict.info())
test_predict['tag'] = test_predict[test_predict.columns[3:]].apply(lambda x: ', '.join(x.dropna().astype(int).astype(str)),axis=1)
test_predict = test_predict.drop(taglist.keys(), axis = 1)
test_predict['tag'] = test_predict['tag'].map(lambda x: '['+str(x)+']')
print(test_predict[test_predict['tag'] == '[]'].count())

test_df3 = test_predict[test_predict['tag'] == '[]'].drop(['tag'], axis = 1)
test_predict2 = test_predict[test_predict['tag'] != '[]'].drop(['Product Name', 'Combined Short Description'], axis = 1)
print(test_df3.info())

## --------------------------------------------------------------------------------------
## Third run: Use "Combined Short Description" to predict tag for untagged products -----
print('## Third Run Results: -----------------------------------------')

## Extracting features from "Combined Short Description" --
ProductShortDescription_train_counts = count_vect.fit_transform(train_df['Combined Short Description'])
ProductShortDescription_test_counts = count_vect.transform(test_df3['Combined Short Description'])
ProductShortDescription_train_tfidf = tfidf_transformer.fit_transform(ProductShortDescription_train_counts)
ProductShortDescription_test_tfidf = tfidf_transformer.transform(ProductShortDescription_test_counts)
print('Train PSD-tfidf shape:',ProductShortDescription_train_tfidf.shape)
print('Test PSD-tfidf shape:', ProductShortDescription_test_tfidf.shape)
df_train_PSD = ProductShortDescription_train_tfidf
df_test_PSD = ProductShortDescription_test_tfidf

def predictTags_byPSD(tagID):

	##-- Test model performance by cross-varlidations
	# from sklearn import cross_validation
	# df_train, df_test, label_train, label_test = cross_validation.train_test_split(df_train_PLD, train_df[tagID], test_size=0.10, random_state = 0)
	## ----------------------------------------------

	label_train = train_df[tagID]
	df_train = df_train_PSD
	clf = MLPClassifier(activation = 'relu', solver = 'lbfgs', alpha = 0.1, hidden_layer_sizes = (50, ), random_state = 1)
	clf.fit(df_train, label_train)

	return clf.predict(df_test_PSD)
	# return clf.score(df_test, label_test)

test_predict = test_df3.drop(['Combined Short Description'], axis = 1)

## Predict 32 Tags one by one ---
for tag_id in taglist:
	test_predict[tag_id] = predictTags_byPSD(tag_id) * int(tag_id)

test_predict = test_predict.replace({0:np.nan})
print(test_predict.info())
test_predict['tag'] = test_predict[test_predict.columns[2:]].apply(lambda x: ', '.join(x.dropna().astype(int).astype(str)),axis=1)
test_predict = test_predict.drop(taglist.keys(), axis = 1)
test_predict['tag'] = test_predict['tag'].map(lambda x: '['+str(x)+']')
print(test_predict[test_predict['tag'] == '[]'].count())
#test_predict3 = test_predict

test_df4 = test_predict[test_predict['tag'] == '[]'].drop(['tag'], axis = 1)
test_predict3 = test_predict[test_predict['tag'] != '[]'].drop(['Product Name'], axis = 1)
print(test_df4.info())


## --------------------------------------------------------------------
## Forth run: Use Only "Product Name" to predict Tag for untagged products ------
print('## Forth Run Results: -----------------------------------------')

## Extracting features from "Product Name" -----
ProductName_train_counts = count_vect.fit_transform(train_df['Product Name'])
ProductName_test_counts = count_vect.transform(test_df4['Product Name'])
ProductName_train_tfidf = tfidf_transformer.fit_transform(ProductName_train_counts)
ProductName_test_tfidf = tfidf_transformer.transform(ProductName_test_counts)
print('Train PN-tfidf shape:', ProductName_train_tfidf.shape)
print('Test PN-tfidf shape:', ProductName_test_tfidf.shape)
df_train_PN = ProductName_train_tfidf
df_test_PN = ProductName_test_tfidf

def predictTags_byPN(tagID):

	##-- Test model performance by cross-varlidations
	# from sklearn import cross_validation
	# df_train, df_test, label_train, label_test = cross_validation.train_test_split(df_train_NewPN, train_df[tagID], test_size=0.10, random_state = 0)
	##-----------------------------------------------

	label_train = train_df[tagID]
	df_train = df_train_PN
	clf = MLPClassifier(activation = 'relu', solver = 'lbfgs', alpha = 0.1, hidden_layer_sizes = (50, ), random_state = 1)
	clf.fit(df_train, label_train)

	# return clf.score(df_test, label_test)
	return clf.predict(df_test_PN)

test_predict = test_df4.drop(['Product Name'], axis = 1)

## Predict 32 Tags one by one ---
for tag_id in taglist:
	test_predict[tag_id] = predictTags_byPN(tag_id) * int(tag_id)


test_predict = test_predict.replace({0:np.nan})
print(test_predict.info())
test_predict['tag'] = test_predict[test_predict.columns[1:]].apply(lambda x: ', '.join(x.dropna().astype(int).astype(str)),axis=1)
test_predict = test_predict.drop(taglist.keys(), axis = 1)
test_predict['tag'] = test_predict['tag'].map(lambda x: '['+str(x)+']')
print(test_predict[test_predict['tag'] == '[]'].count())
test_predict4 = test_predict

# test_df5 = test_predict[test_predict['tag'] == '[]'].drop(['tag'], axis = 1)
# test_predict4 = test_predict[test_predict['tag'] != '[]']


# test_df5['tag'] = '[4537]'
# print(test_df5.info())

print('## number of untagged products ----------------------------------------------------------------------')
test_predict = pd.concat([test_predict1, test_predict2, test_predict3, test_predict4])
test_predict = test_predict.sort(['item_id'])

print(test_predict[test_predict['tag'] == '[]'].count())

## write predicted results to a new datafile ---------------------------------
test_predict = test_predict.values
prediction_file = open('tags.tsv','wb')
prediction_file_object = csv.writer(prediction_file, delimiter = '\t')
prediction_file_object.writerow(['item_id','tag'])
for row in test_predict:
	prediction_file_object.writerow(row)
prediction_file.close()



