import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from joblib.numpy_pickle_utils import xrange
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
def review_to_words(raw_view):
    review_text=BeautifulSoup(raw_view).get_text()
    letters_only=re.sub("[^a-zA-Z]]","",review_text)
    words = letters_only.lower()
    stops=set(stopwords.words("english"))
    meaningful_words=[w for w in words if not w in stops]
    return("".join(meaningful_words))

#“header=0”表示文件的第一行包含列名，“delimiter=\t”表示字段用制表符分隔，quoting=3告诉Python忽略双引号，否则可能会遇到错误试图读取文件
train=pd.read_csv("./labeledTrainData.tsv",header=0,delimiter='\t',quoting=3)
print(train.shape,train.columns.values)

#数据清洗和文本预处理
# example1=BeautifulSoup(train["review"][0])
# print(train["review"][0])
# print(example1.get_text())
# letters_only=re.sub("[^a-zA-Z]]","",example1.get_text())
# print(letters_only)
# lower_case=letters_only.lower()
# words=lower_case.split()
# #nltk.download()
# print(stopwords.words("english"))
# words=[w for w in words if not w in stopwords.words("english")]
# print(words)

clean_review=review_to_words(train["review"][0])
print(clean_review)
num_reviews=train["review"].size
clean_train_reviews=[]
for i in xrange(0,num_reviews):
    if ((i + 1) % 1000 == 0):
        print("Review %d of %d\n" % (i + 1, num_reviews))
    clean_train_reviews.append(review_to_words(train["review"][i]))

print("Creating the bag of words...\n")
vectorizer=CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)
train_data_features=vectorizer.fit_transform(clean_train_reviews)
train_data_features=train_data_features.toarray()
print(train_data_features.shape)
vocab=vectorizer.get_feature_names()
print(vocab)

#统计单词计数
# dist=np.sum(train_data_features,axis=0)
# for tag,count in zip(vocab,dist):
#     print(count,dist)

#随机森林
print("Training the random forest...\n")
forest=RandomForestClassifier(n_estimators=100)
forest=forest.fit(train_data_features,train["sentiment"])

test=pd.read_csv("testData.tsv",header=0,delimiter="\t",quoting=3)
print(test.shape)
test_num=len(test["review"])
clean_test_reviews=[]
print("cleaning and parsing the test set movies reviews...\n")
for i in xrange(0,test_num):
    if((i+1)%1000==0):
        print("Review %d of %d\n" % (i+1, num_reviews))
    clean_review=review_to_words(test["review"][i])
    clean_test_reviews.append(clean_review)
test_data_features=vectorizer.transform(clean_test_reviews)
test_data_features=test_data_features.toarray()
result=forest.predict(test_data_features)
output=pd.DataFrame(data={"id":test["id"],"sentiment":result})
output.to_csv("Bage_of_woeds_model.csv",index=False,quoting=3)


