def main():

    from sklearn.datasets import fetch_20newsgroups
    #ngt = fetch_20newsgroups(subset='train') #News Group Training
    ngt = fetch_20newsgroups()

    from pprint import pprint
    #pprint(list(ngt.target_names))

    from sklearn.feature_extraction.text import TfidfVectorizer

    categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
    newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
    vectorizer = TfidfVectorizer()
    print newsgroups_train.data
    vectors = vectorizer.fit_transform(newsgroups_train.data)
    vectors.shape #(2034, 34118)

    newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'),categories=categories)
    vectors_test = vectorizer.transform(newsgroups_test.data)

    from sklearn.naive_bayes import MultinomialNB
    from sklearn import metrics

    clf = MultinomialNB(alpha=.01)
    clf.fit(vectors, newsgroups_train.target)

    pred = clf.predict(vectors_test)
    f1 = metrics.f1_score(pred, newsgroups_test.target)
    print "f1 score = "+str(f1)

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore") #to hide warns about deprecated functions
    main()
