from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import *
from nltk.tokenize import RegexpTokenizer
import nltk
from nltk import word_tokenize          
from sklearn.feature_extraction.text import *
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import string
import numpy as np

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")#, ignore_stopwords=True) #PorterStemmer()
#from nltk.stem.lancaster import *
#stemmer =  LancasterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed
def tokenize(text):
    text = "".join([string.lower(ch) for ch in text if ch not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

'''
import nltk.stem.wordnet as wordnet
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.wordnet.ADV
    else:
        return 'n'

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(lemmatizer.lemmatize(item[0], get_wordnet_pos(item[1])))
    return stemmed


def tokenize(text):
    text = "".join([string.lower(ch) for ch in text if ch not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    tokens = nltk.pos_tag(tokens) #to add tags
    stems = stem_tokens(tokens, stemmer)
    return stems
'''

#print tokenize(unicode("we are testing some new techniques & methods"))
#import sys
#sys.exit()

'''
from wordmatch import *
d = [["we",'have','won'],['winner','is', 'amine'],['I','had','win','competition']]
print d
wordmatch(d)
print d
'''             

def show_top10(classifier, vectorizer, categories):
    feature_names = np.asarray(vectorizer.get_feature_names())
    for i, category in enumerate(categories):
        top10 = np.argsort(classifier.coef_[i])[-10:]
        print("%s: %s" % (category, " ".join(feature_names[top10])))


#def main():
cats = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
#cats = ['alt.atheism', 'comp.graphics', 'soc.religion.christian']
cats = None
twenty_train = fetch_20newsgroups(subset='train', categories=cats)#, remove=('headers', 'footers', 'quotes'))
data = twenty_train.data
#count_vect = CountVectorizer(tokenizer=tokenize, stop_words="english")
#X_train_counts = count_vect.fit_transform(data)
#tfidf_transformer = TfidfTransformer()
#X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#wordmatch(data)
vectorizer = TfidfVectorizer( stop_words="english", tokenizer=tokenize) #min_df=1,
X_train_tfidf = vectorizer.fit_transform(data)

''' NAIVE BAYES '''

#clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

from sklearn.linear_model import SGDClassifier

parameters = {
    'alpha': 1e-05,
    'penalty': 'elasticnet',
    'n_iter': 80,
}

clf= SGDClassifier().set_params(parameters)

clf.fit(X_train_tfidf, twenty_train.target)
twenty_test = fetch_20newsgroups(subset='test',categories=cats)#, remove=('headers', 'footers', 'quotes'))
docs_new = twenty_test.data

#X_new_counts = count_vect.transform(docs_new)
#X_new_tfidf = tfidf_transformer.transform(X_new_counts)
X_new_tfidf = vectorizer.transform(docs_new)

'''
with open("my_CF-IDF_Matrix.txt","a") as f:
    for i in range(X_new_tfidf.shape[0]):
        for j in range(X_new_tfidf.shape[1]):
            f.write(str(X_new_tfidf[i,j])+", ")
        f.write("\n")

'''
predicted = clf.predict(X_new_tfidf)


'''
import csv
with open('matrice.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=' ',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)

    spamwriter.writerow(ar)
'''

#from sklearn.externals import joblib
#joblib.dump(clf, 'filename.pkl')
#joblib.dump(X_train_tfidf, 'filename2.pkl')

#import pickle
#with open('test_sparse_array.dat', 'wb') as outfile:
#    pickle.dump(X_train_tfidf, outfile, pickle.HIGHEST_PROTOCOL)

#import scipy
#scipy.io.mmwrite("matrix.mtx", X_new_tfidf, comment='', field=None, precision=None)

f1 = metrics.f1_score(predicted, twenty_test.target)
print "NaiveB f1 score = "+str(f1)

#show_top10(clf, vectorizer , twenty_train.target_names)


'''
del f1, predicted, clf

######KNN######
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train_tfidf, twenty_train.target) 
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_neighbors=5, p=2, weights='uniform')
predicted = knn.predict(X_new_tfidf)
f1 = metrics.f1_score(predicted, twenty_test.target)
print "KNN f1 score = "+str(f1)
del f1, predicted, clf

#######VM##########
from sklearn import svm
clf = svm.SVC()
clf.fit(X_train_tfidf, twenty_train.target)  
svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
gamma=0.0, kernel='rbf', max_iter=-1, probability=False, random_state=None,
shrinking=True, tol=0.001, verbose=False)
predicted = clf.predict(X_new_tfidf)
f1 = metrics.f1_score(predicted, twenty_test.target)
print "SVM f1 score = "+str(f1)
del f1, predicted, clf
'''

'''
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore") #to hide warns about deprecated functions
    main()
'''
