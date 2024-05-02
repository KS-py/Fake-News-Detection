# FAKE NEWS DETECTOR with scikit-learn Classifiers, Simplilearn tutorial

Using the training data from the two CSV files, we perform some formatting and also add a boolean field to the data to distinguish fake from true

We add this to the existing data for the training

We add a word parser to parse out all symbols and special characters from the training data so it'll be only text.

We apply this word parser to the data and then split the data into train and test sets

Using the TfidfVectorizer from sklearn.feature_extraction.text, we vectorize the x training and x test data

We employ four distinct Classifiers:
Logistic Regression
Decision Tree
Gradient Boosting 
Random Forest

In the manual_testing function, we pass a string to the functino which creates a DataFrame, apply the word parser, do the vectorization, then predict using the four classifiers.

The full dataset for the training of the classifiers can be found here:
https://drive.google.com/drive/folders/1ByadNwMrPyds53cA6SDCHLelTAvIdoF_  

Too large to upload here
