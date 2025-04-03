import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import pickle
from sklearn.svm import LinearSVC
import re

training_path = '/Users/abauer/Documents/train.tsv'
test_path = '/Users/abauer/Documents/test.tsv'

train_df = pd.read_csv(training_path,sep='\t')
test_df = pd.read_csv(test_path,sep='\t')
df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

df['combined'] = df['title'] + df['text']

x_train, x_test, y_train, y_test = train_test_split(df['combined'], 
                                                        df['label'], 
                                                        test_size=0.2)

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.85, min_df=3,
                             token_pattern=r'\b[a-zA-Z]+\b')
xvtrain = vectorizer.fit_transform(x_train)
xvtest = vectorizer.transform(x_test)
logistic_model = LogisticRegression(class_weight='balanced', max_iter=1000)
logistic_model.fit(xvtrain,y_train)

print("Training Accuracy:" , accuracy_score(y_train, logistic_model.predict(xvtrain)))
print("Testing Accuracy: ", accuracy_score(y_test, logistic_model.predict(xvtest)))

svc_model = LinearSVC()
svc_model = LinearSVC(class_weight='balanced', max_iter=5000)

print(svc_model.fit(xvtrain, y_train))
print("SVC Score: " , svc_model.score(xvtest,y_test))

def pr(txt):
    vectorized_text = vectorizer.transform([txt])
    probabilities = logistic_model.predict_proba(vectorized_text)[0]
    false_prob = probabilities[0]
    true_prob = probabilities[1]
    prediction_prob = true_prob if true_prob > false_prob else false_prob
    logistic_prediction = True if prediction_prob == true_prob else False
    svc_prediction = svc_model.predict(vectorized_text)
    consensus = svc_prediction[0] == logistic_prediction 
    confidence = round(100*prediction_prob)
    prediction = "Real News" if logistic_prediction == 1 else "Fake News"

    if (consensus and confidence > 70) or confidence > 80  :
        print(f"{confidence}% sure that this is {prediction} ")
    else :
            print(f"{confidence}% sure that this is {prediction}... You should fact check it though!")
    return svc_prediction
    

value = input('Enter something: ')
while True and value != 'q': 
    pr(value)
    value = input('Enter something: ')



