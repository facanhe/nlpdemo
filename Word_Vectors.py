from gensim.models import Word2Vec
import pandas as pd
model=Word2Vec.load("300features_40minwords_10context")
print(model.wv.vectors.shape)

train=pd.read_csv("labeledTrainData.tsv",header=0,delimiter="\t", quoting=3 )
test=pd.read_csv("testData.tsv", header=0,delimiter="\t",quoting=3 )
unlabeled_train=pd.read_csv("unlabeledTrainData.tsv",header=0, delimiter="\t", quoting=3 )

import  numpy as np
def makeFeatureVec(words,model,num_features):
    featureVec=np.zeros((num_features),dtype="float32")
    nwords=0.
    index2word_set=set(model.wv.index_to_key)
    for word in words:
        if word in index2word_set:
            nwords=nwords+1.
            featureVec=np.add(featureVec,model.wv[word])
    featureVec=np.divide(featureVec,nwords)
    return featureVec
def getAvgFeatureVecs(reviews,model,num_features):
    counter=0
    reviewFeatureVecs=np.zeros((len(reviews),num_features),dtype="float32")
    for review in reviews:
        if counter%1000==0:
            print("Review %d of %d" % (counter, len(reviews)))
        reviewFeatureVecs[counter]=makeFeatureVec(review,model,num_features)
        counter=counter+1
    return reviewFeatureVecs
from Word2Vec import review_to_wordlist,num_features
clean_train_reviews=[]
for review in train["review"]:
    clean_train_reviews.append(review_to_wordlist(review,remove_stopwords=True))
trainDataVecs=getAvgFeatureVecs(clean_train_reviews,model,num_features)
print("Creating average feature vecs for test reviews")
clean_test_reviews=[]
for review in train["review"]:
    clean_test_reviews.append(review_to_wordlist(review,remove_stopwords=True))
testDataVecs=getAvgFeatureVecs(clean_test_reviews,model,num_features)

#使用平均段落向量来训练随机森林
from sklearn.ensemble import RandomForestClassifier
forest=RandomForestClassifier(n_estimators=100)
print("Fitting a random forest to labeled training data...\n")
forest=forest.fit(trainDataVecs,train["sentiment"])
result=forest.predict(testDataVecs)
output=pd.DataFrame(data={"id":test["id"]})
output.to_csv("Word2Vec_AverageVectors.csv",index=False,quoting=3)

#聚类
from sklearn.cluster import KMeans
import time
start=time.time()
word_vectors=model.wv.vectors
num_clusters=int(word_vectors.shape[0]/5)
kmeans_clustering=KMeans(n_clusters=num_clusters)
idx=kmeans_clustering.fit_predict(word_vectors)
end=time.time()
elapsed=end-start
print("Time taken for Kmeans clustering:",elapsed,"seconds.")

word_centroid_map=dict(zip(model.wv.index_to_key,idx))
for cluster in range(0,10):
    print("\nCluster %d" %cluster)
    words=[]
    for i in range(0,len(word_centroid_map.values())):
        if(list(word_centroid_map.values())[i]==cluster):
            words.append(list(word_centroid_map.keys())[i])
    print(words)

def create_bag_of_centroids(wordlist,word_centroid_map):
    num_centroids=max(word_centroid_map.values())+1
    bag_of_centroids=np.zeros(num_centroids,dtype="float32")
    for word in wordlist:
        if word in word_centroid_map:
            index=word_centroid_map[word]
            bag_of_centroids[index]+=1
    return bag_of_centroids

train_centroids=np.zeros((train["review"].size,num_clusters),dtype="float32")
counter=0
for review in clean_train_reviews:
    train_centroids[counter]=create_bag_of_centroids(review,word_centroid_map)
    counter+=1

test_centroids=np.zeros((test["review"].size,num_clusters),dtype="float32")
counter=0
for review in clean_test_reviews:
    test_centroids[counter]=create_bag_of_centroids(review,word_centroid_map)
    counter+=1

forest=RandomForestClassifier(n_estimators=100)
print("Fitting a random forest to labeled training data...")
forest = forest.fit(train_centroids,train["sentiment"])
result = forest.predict(test_centroids)

output = pd.DataFrame(data={"id":test["id"], "sentiment":result})
output.to_csv( "BagOfCentroids.csv", index=False, quoting=3 )