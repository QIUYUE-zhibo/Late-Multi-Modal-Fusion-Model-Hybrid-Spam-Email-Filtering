
import os
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

def make_Dictionary(train_dir):
    emails = [os.path.join(train_dir,f) for f in os.listdir(train_dir)]    
    all_words = []       
    for mail in emails:    
        with open(mail) as m:
            for i,line in enumerate(m):#enumerate
                if i == 2:
                    words = line.split()
                    all_words += words
    
    dictionary = Counter(all_words)
    #Remove non-words
    list_to_remove = dictionary.keys()
    for item in list(list_to_remove):
        if item.isalpha() == False: 
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    dictionary = dictionary.most_common(3000)
    return dictionary
    
def extract_features(mail_dir): 
    files = [os.path.join(mail_dir,fi) for fi in os.listdir(mail_dir)]
    features_matrix = np.zeros((len(files),3000))
    docID = 0;
    for fil in files:
      with open(fil) as fi:
        for i,line in enumerate(fi):
          if i == 2:
            words = line.split()
            for word in words:
              wordID = 0
              for i,d in enumerate(dictionary):
                if d[0] == word:
                  wordID = i
                  features_matrix[docID,wordID] = words.count(word)
        docID = docID + 1     
    return features_matrix
    
# Create a dictionary of words with its frequency

train_dir = 'lingspam_public\\lemm_stop\\train-mails'
dictionary = make_Dictionary(train_dir)

# Prepare feature vectors per training mail and its labels

train_labels = np.zeros(702)
train_labels[351:701] = 1
train_matrix = extract_features(train_dir)

# Training SVM and Naive bayes classifier and its variants

model1 = LinearSVC()
model2 = MultinomialNB()
model3 = RandomForestClassifier()
model4 = ExtraTreesClassifier()

model1.fit(train_matrix,train_labels)
model2.fit(train_matrix,train_labels)
model3.fit(train_matrix,train_labels)
model4.fit(train_matrix,train_labels)
# Test the unseen mails for Spam

test_dir = 'lingspam_public\\lemm_stop\\test-mails'
test_matrix = extract_features(test_dir)
test_labels = np.zeros(260)
test_labels[130:260] = 1

actual_result_1 = model1._predict_proba_lr(test_matrix)
actual_result_2 = model2.predict_proba(test_matrix)
actual_result_3 = model3.predict_proba(test_matrix)
actual_result_4 = model4.predict_proba(test_matrix)
result1 = model1.predict(test_matrix)
result2 = model2.predict(test_matrix)
result3 = model3.predict(test_matrix)
result4 = model4.predict(test_matrix)

actual_result_3=list(actual_result_3)
actual_result_4=list(actual_result_4)

#linear fusion model
aerfa = 0.25
fusion_result_prob=[]
fusion_result=[]
for i,j,k,l in actual_result_1,actual_result_2,actual_result_3,actual_result_4:
    fusion_result_prob.append(aerfa*i + aerfaj + aerfa*k + aerfa*l)
for i in fusion_result_prob:
    del (i[0])
    if i[0] > 0.5:
        fusion_result.append(1)
    else:
        fusion_result.append(0)

ac_score_1 = metrics.accuracy_score(test_labels, result1)
ac_score_2 = metrics.accuracy_score(test_labels, result2)
ac_score_3 = metrics.accuracy_score(test_labels, result3)
ac_score_4 = metrics.accuracy_score(test_labels, result4)

recall_score_1 = metrics.recall_score(test_labels, result1)
recall_score_2 = metrics.recall_score(test_labels, result2)
recall_score_3 = metrics.recall_score(test_labels, result3)
recall_score_4 = metrics.recall_score(test_labels, result4)

precision_score_1 = metrics.precision_score(test_labels, result1)
precision_score_2 = metrics.precision_score(test_labels, result2)
precision_score_3 = metrics.precision_score(test_labels, result3)
precision_score_4 = metrics.precision_score(test_labels, result4)

F1_1 = metrics.f1_score(test_labels, result1)
F1_2 = metrics.f1_score(test_labels, result2)
F1_3 = metrics.f1_score(test_labels, result3)
F1_4 = metrics.f1_score(test_labels, result4)


print(actual_result_3)
print(actual_result_4)
print(ac_score_1,ac_score_2,ac_score_3,ac_score_4)
print(recall_score_1,recall_score_2,recall_score_3,recall_score_4)
print(precision_score_1,precision_score_2,precision_score_3,precision_score_4)
print(F1_1,F1_2,F1_3,F1_4)

print(confusion_matrix(test_labels,result1))
print(confusion_matrix(test_labels,result2))
print(confusion_matrix(test_labels, result3))
print(confusion_matrix(test_labels, result4))


