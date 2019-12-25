# News-Category-Prediction
This model predicts news category. We will use the concepts of Machine Learning. We compared the performance of two algorithms in correctly classifying the category of the news. We used Multinomial Naive Bayes and Logistic Regression. For Multinomial NB, we got accuracy of 93.86%. For Logistic Regression, we got accuracy of 94.01%. We split the dataset into 70% train dataset and 30% test dataset.

Tools Used:
1. Python
2. Numpy
3. ScikitLearn libraries

Method:
1. Import the dataset into the python notebook.
2. Import all the necessary python libraries.
3. Store the 'text' column in X and 'category' column in Y.
4. Split the X and Y in 70% train and 30% test dataset using train_test_split() method.
5. Vectorize X using tfidfVectorizer() and Y using LabelEncoder().
6. Create and test MultinomialNB and Logistic Regression models using X and Y.
7. Now test the created models on test datasets created in step 4.
8. Print the accuracy for both the models.
