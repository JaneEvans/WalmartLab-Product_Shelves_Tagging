# WalmartLab-Product_Shelves_Tagging
In this problem WalmartLab asks for a machine learning solution to the task of assigning products to shelves.


"Predict_shelves_tags.py" is the code you should use to generate the "tags.tsv".

-----------
### Requires: 
	1. Python 2.7 or higher version (Cannot support Python 3. because sklearn.neural_network.MLPClassifier only supports Python 2. at this time)
	2. You need pandas and numpy packages. The easiest way to install both is to install Anaconda:
	(https://docs.continuum.io/anaconda/install)
	3. Install SciKit-Learn 0.18 version (sklearn.neural_network.MLPClassifier is new in version 0.18): 
	(http://scikit-learn.org/stable/install.html#installation-instructions)
	4. The"train.tsv" and "test.tsv" files should be in the same folder with "Predict_shelves_tags.py" (I can't include the two datasets in the .zip file because they are two big, but you can have them from your end I believe.)

-----------
### Links to external packages:
1. pandas: 
	http://pandas.pydata.org/pandas-docs/stable/overview.html
2. Numpy:
	https://docs.scipy.org/doc/numpy-dev/user/quickstart.html

3. sklearn.feature_extraction.text.CountVectorizer: 
	http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer
4. sklearn.feature_extraction.text.TfidfTransformer:
	http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html#sklearn.feature_extraction.text.TfidfTransformer
5. sklearn.neural_network.MLPClassifier:
	http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier



	
-------------
11/05/2016 By Jinjin Ge
