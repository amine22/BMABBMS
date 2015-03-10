import requests
from pattern import web
#from BeautifulSoup import BeautifulSoup
from lxml import html as HTML

import re
def remove_tags(html):
    TAG_RE = re.compile(r'<[^>]+>') #pour supprimer tout ce qui est entre < >
    html = html.replace('&lt;','<') #html est un htmlescapped, on le unscape !
    html = html.replace('&gt;','>')
    return TAG_RE.sub('', html)

def textToHTML(text):
    text = HTML.fromstring(text).text
    html = web.Element(text)
    return html
    

def getYahoo():
    r = requests.get("http://news.yahoo.com/rss")
    print 'News from: '+r.url 
    print '@'*50
    print

    dom = web.Element(r.text)
    cnt=0
    for item in dom.by_tag('item'):
        cnt+=1
        title = item.by_tag('title')[0].content
        try :
            link = item.by_tag('media:text')[0].content
            link = textToHTML(link)
            link = link.by_tag("a")[0].attr['href'].encode('utf-8')
        except:
            link =""
            
        try :
            html = remove_tags(item.by_tag('media:text')[0].content)
            html = HTML.fromstring(html).text
        except:
            html = ''
        

        print str(cnt) +' - '+title
        print 'url: '+ link
        print '-------------------------------------'
        print html
        print
        print '####################################################'
        print
    
    
def getGoogle():
    r = requests.get("http://news.google.com/news?cf=all&ned=us&hl=en&output=rss")
    print 'News from: '+r.url 
    print '@'*50
    print
    l = []
    dom = web.Element(r.text)
    cnt=0
    for item in dom.by_tag('item'):
        cnt+=1
        title = item.by_tag('title')[0].content
        try :
            link = item.by_tag('description')[0].content
            link = textToHTML(link)
            link = link.by_tag("a")[0].attr['href'].encode('utf-8')
        except:
            link =""
        

        try :
            html = remove_tags(item.by_tag('description')[0].content)
            #html= BeautifulSoup(html, convertEntities=BeautifulSoup.HTML_ENTITIES)
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
    #return l
    
l = getGoogle()
