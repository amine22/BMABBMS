# -*- coding: cp1252 -*-
def wordmatch(data,limit=0.55):
    from difflib import SequenceMatcher
    s = SequenceMatcher(None)
    d1,d2,w1,w2=0,0,0,0
    doc, doc2=[],[]
    for d1 in range(len(data)):
        doc = data[d1].split(" ")
        for w1 in range(len(doc)):
           s.set_seq1(doc[w1])
           for d2 in range(len(data)):
                doc2 = data[d2].split(" ")
                for  w2 in range(len(doc2)):
                    s.set_seq2(doc2[w2])
                    ratio = s.ratio()
                    mblocks = len(s.get_matching_blocks())
                    b = ratio>=limit and mblocks>=2
                    #print data[d1][w1]+', '+data[d2][w2]+' : '+str(ratio) +" - "+ str(mblocks)
                    if (b):
                        doc2[w2] = doc[w1]
                        x = " ".join(doc)
                        y = " ".join(doc2)
                        data[d1] , data[d2] = x , y
    #return data

d = [unicode('we have won compet'),'winner is amine','I had win competition']
print d
wordmatch(d)
print d

#ça marche ya si salim mais la complexité O(n^4) :(((((((((((((
#si je le lance pour le vrai dataset ça bugguera c sur, nsayi f lil.
