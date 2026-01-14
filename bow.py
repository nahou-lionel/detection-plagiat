import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer

#Load the data

data = pd.read_csv("~/Téléchargements/data-plagiarism/orig_taska.txt", sep='\r', header=None)
# data = pd.read_csv("~/Téléchargements/sms+spam+collection/SMSSpamCollection")
# data.head()

#Create a basic sparse matrix

# vectorizer = CountVectorizer(stop_words='english')
# The code sample removes any word from the sparse matrix that appears less than 20% and over 80% of the time in each text.
# vectorizer = CountVectorizer(max_df=0.80, min_df=0.20)

# limit to the most commonly used x_number of words
# vectorizer = CountVectorizer(max_features = 50)

vectorizer = CountVectorizer(stop_words='english')


matrix = vectorizer.fit_transform(data[0])

# #Visualize as a dataframe
print(f"Vocabulary : {vectorizer.get_feature_names_out()}")
df = pd.DataFrame(data= matrix.toarray(), columns = vectorizer.get_feature_names_out())
print(f" BoW :\n{df}")


# print(f"Indexes of each feature name : {vectorizer.vocabulary_}")



