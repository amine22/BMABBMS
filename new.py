from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import *
from nltk.tokenize import RegexpTokenizer
import nltk
from sklearn.feature_extraction.text import *
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

def vectorize(doc):
    cv = CountVectorizer()
    tokenizer = RegexpTokenizer(r'\w+') #supprimer ponctuation
    doc = tokenizer.tokenize(doc)
    doc = [l.lower() for l in doc] #mettre en miniscule
    doc = nltk.pos_tag(doc)
    stemmer = nltk.PorterStemmer()
    stemmed = []
    for word in doc:
        new = stemmer.stem(word[0])
        stemmed.append(new+word[1]) # liste de motTAG1, motTAG2 ...
    stemmed = " ".join(stemmed) #recreer le doc sous format text
    return stemmed


def main():
    #categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics']
    #categories = ['comp.graphics','soc.religion.christian']
    twenty_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42, remove=('headers', 'footers', 'quotes'))
    data = []
    
    for doc in twenty_train.data:
        #stopwords
        cv = CountVectorizer(tokenizer=True, stop_words="english")
        tokenizer = RegexpTokenizer(r'\w+') #supprimer ponctuation
        doc = tokenizer.tokenize(doc)
        doc = [l.lower() for l in doc] #mettre en miniscule
        doc = nltk.pos_tag(doc) #assigner les tags
        stemmer = nltk.PorterStemmer() #Porter stemmer
        stemmed = []
        for word in doc:
            stem = stemmer.stem(word[0])
            #stemmed.append((new,word[1]))
            stemmed.append(stem+word[1]) # liste de motTAG1, motTAG2 ...
        stemmed = " ".join(stemmed) #recreer le doc sous format text
        data.append(stemmed)        
        

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(data)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)
    

    #new = ['God is love', 'OpenGL on the GPU is fast']
    twenty_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=12, remove=('headers', 'footers', 'quotes'))
    new = twenty_test.data
    docs_new = []
    for d in new:
        docs_new.append(vectorize(d))
        
    X_new_counts = count_vect.transform(docs_new)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)

    predicted = clf.predict(X_new_tfidf)

    #for doc, category in zip(docs_new, predicted):
    #    print('%r => %s' % (doc, twenty_train.target_names[category]))

    f1 = metrics.f1_score(predicted, twenty_test.target)
    print "f1 score = "+str(f1)

    #return f1


import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore") #to hide warns about deprecated functions
    main()
