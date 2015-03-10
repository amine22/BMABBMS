import string
import nltk
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")#, ignore_stopwords=True)
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


#categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
from sklearn.datasets import fetch_20newsgroups
ng = fetch_20newsgroups()
categories = ng.target_names

import pickle
clf = pickle.load( open( "NaiveBayes.pickle", "rb" ) )
vectorizer = pickle.load( open( "NaiveBayesVectorizer.pickle", "rb" ) )


###################################################################

import requests
#from pattern import web
from lxml import html as HTML
from BeautifulSoup import BeautifulSoup

import re
def remove_tags(html):
    TAG_RE = re.compile(r'<[^>]+>') #pour supprimer tout ce qui est entre < >
    html = html.replace('&lt;','<') #html est un htmlescapped, on le unscape !
    html = html.replace('&gt;','>')
    return TAG_RE.sub('', html)

def textToHTML(text):
    text = HTML.fromstring(text).text
    html = BeautifulSoup(text)
    return html
    
'''
def getYahoo():
    r = requests.get("http://news.yahoo.com/rss")
    print 'News from: '+r.url 
    print '@'*50
    print
    l = []
    cnt=0
    bs = BeautifulSoup(r.text)
    for item in bs.findAll('item'):
        cnt+=1
        title = item.find('title').contents[0]
        try :
            link = item.find('media:text').contents[0]
            link = textToHTML(link)
            link = link.find("a").get('href')
        except:
            link =""
            
        try :
            html = remove_tags(item.find('media:text').contents[0])
            html = HTML.fromstring(html).text
        except:
            html = ''

        l.append((title,link,html))

        data = [title+html]
        vecteur = vectorizer.transform(data)
        
        print str(cnt) +' - '+ title
        print 'url: '+ link
        print '-------------------------------------'
        print html
        print
        print 'classifier as:' + categories[clf.predict(vecteur)]
        print
        print '####################################################'
        print
    return l
    
    
def getGoogle():
    r = requests.get("http://news.google.com/news?cf=all&ned=us&hl=en&output=rss")
    print 'News from: '+r.url 
    print '@'*50
    print
    l = []
    cnt=0
    bs = BeautifulSoup(r.text)
    for item in bs.findAll('item'):
        cnt+=1
        title = item.find('title').contents[0]
        try :
            link = item.find('description').contents[0]
            link = textToHTML(link)
            link = link.find("a").get('href')
        except:
            link =""
            
        try :
            html = remove_tags(item.find('description').contents[0])
            html = HTML.fromstring(html).text
        except:
            html = ''

        l.append((title,link,html))
        
        print str(cnt) +' - '+ title
        print 'url: '+ link
        print '-------------------------------------'
        print html
        print
        print '####################################################'
        print
    return l
'''    
#l = getGoogle()
#l = getYahoo()

'''
def getNews():
    source = [("http://news.yahoo.com/rss","media:text"),("http://news.google.com/news?cf=all&ned=us&hl=en&output=rss","description")]
    #source , liste de sources avec lien et champ contenant le description
    l = []
    cnt=0
    for s in source:

        r = requests.get(s[0])
        print 'News from: '+r.url 
        print '@'*50
        print
        
        bs = BeautifulSoup(r.text)
        for item in bs.findAll('item'):
            cnt+=1
            title = item.find('title').contents[0]
            try :
                link = item.find(s[1]).contents[0]
                link = textToHTML(link)
                link = link.find("a").get('href')
            except:
                link = ""
                
            try :
                html = remove_tags(item.find(s[1]).contents[0])
                html = HTML.fromstring(html).text
            except:
                html = ''

            l.append((title,link,html))

            data = [title+html]
            vecteur = vectorizer.transform(data)
            
            print str(cnt) +' - '+ title
            print 'url: '+ link
            print '-------------------------------------'
            print html
            print
            print 'classifier as:' + categories[clf.predict(vecteur)]
            print
            print '####################################################'
            print

    return l

l = getNews()

'''


def getReuters():
    f = open("reutersFeeds.txt",'r')
    l = []
    #cnt=0
    for s in f:

        r = requests.get(s)
        print 'Got news from: '+r.url 
        print '-'*75
        
        
        bs = BeautifulSoup(r.text)
        for item in bs.findAll('item'):
            #cnt+=1
            title = item.find('title').contents[0]
            try :
                link = item.find("feedburner:origlink").contents[0]
            except:
                link = ""
                
            try :
                html = remove_tags(item.find('description').contents[0])
                html = HTML.fromstring(html).text
            except:
                html = ''

            if (title,link,html) not in l:
                l.append((title,link,html))
            '''
            data = [title+html]
            vecteur = vectorizer.transform(data)
            
            print str(cnt) +' - '+ title
            print 'url: '+ link
            print '-------------------------------------'
            print html
            print
            print 'classified as:' + categories[clf.predict(vecteur)]
            print
            print '####################################################'
            print
            '''

    f.close()
    return l

l = getReuters()
def printNews(news):
    cnt=0
    for tupl in news:
        cnt += 1
        data = [tupl[0]+' '+tupl[2]]
        vecteur = vectorizer.transform(data)
        print str(cnt) +' - '+ tupl[0]
        print 'url: '+ tupl[1]
        print '-------------------------------------'
        print tupl[2]
        print
        print 'classified as:' + categories[clf.predict(vecteur)]
        print '####################################################'
        print

printNews(l)
