import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import math
import numpy as np

#PREPROCESSING
latihinformasi = []
latihkeluhan = []
latihsaran = []
with open ('tesinformasi.txt') as datalatih:
    for element in datalatih.readlines():
        latih1 = element.replace('\n','')
        latihinformasi.append(latih1)

with open ('teskeluhan.txt') as datalatih2:
    for element in datalatih2.readlines():
        latih2 = element.replace('\n','')
        latihkeluhan.append(latih2)

with open ('tessaran.txt') as datalatih3:
    for element in datalatih3.readlines():
        latih3 = element.replace('\n','')
        latihsaran.append(latih3)

def Stemming(a):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stemming = [stemmer.stem(str(a)) for a in a]
    return stemming
def Casefold(a):
    casefold = lambda kata: kata.lower()
    hasilcasefold = [casefold(str(kata)) for kata in a]
    return hasilcasefold
def Tokenisasi(a):
    tokenize = lambda kata: kata.split(" ")
    hasiltoken = [tokenize(str(kata)) for kata in a]
    return hasiltoken
def Stopword(a):
    with open("stopwords.txt") as stopword:
        content = stopword.readlines()
    stopwords = [x.strip() for x in content]
    filter = lambda doc: [w for w in doc if w not in stopwords]
    hasilstopword = [filter(doc) for doc in a]
    return hasilstopword
def preprocessing(a):
    stem = Stemming(a)
    casefold = Casefold(stem)
    token = Tokenisasi(casefold)
    stopword = Stopword(token)
    return stopword


#PEMBOBOTAN
#a = idf
#b = tf
tfidf = []
def hitungtfidf(a,b):
    i = 0
    while i < len(a):
        tfidf.append([])
        j = 0
        while j < len(b[i]):
            tfidf[i].append(a[i] * b[i][j])
            j += 1
        i += 1
    return tfidf
#a = yang mau ditranspose
#b = panjang (0)
def transposematriks(a,b):
    transpose = [[a[j][i] for j in range(len(a))] for i in range(len(a[b]))]
    return transpose


#Training level 1

#a = dok
#b = transposetfidf
kernelpolynomial= []
def kernelpoly(a,b):
    for i in range(len(a)):
        for j in range(len(b)):
            hasilkernel = (sum(a[j] * a[i]) + 1) ** 2
            kernelpolynomial.append(hasilkernel)
    return kernelpolynomial
#a = polyhasil
#b = kalihes
def matrikshessian(a,kali):
    matrikshessianlvl1 = []
    for isi in range(len(a)):
        hessiann = []
        for i in range(len(a[isi])):
            hasilhes = (a[i][isi] + 9) * kali[isi]
            hessiann.append(hasilhes)
        matrikshessianlvl1.append(hessiann)
    return matrikshessianlvl1
def sequentiallevel1(iterasi,hessian,a,g,transpos,c):
    for j in range(iterasi):
        for i in range(len(hessian)):
            e = [transpos[i][k] * a[k] for k in range(len(transpos[i]))]
            delta = [min(max(g * (1 - e[k]), -1 * a[k]), c - a[k]) for k in range(len(e))]
        a = [a[i] + delta[i] for i in range(len(e))]
    return a
def supportvector(ai):
    sv = max(ai)
    return sv
def mencarinilaix (a,sv):
    x = [isi/sv for isi in a]
    return x
def nilaiw(a,x,kali):
    w = []
    for i in range(len(a)):
        hasilw = a[i] * x[i] * kali[i]
        w.append(hasilw)
    return w

def traininglvl1(a,kali):
    kali = []
    kali2 = []
    while len(kali) < 9:
        if len(kali) < 3:
            kali.append(1)
        else:
            kali.append(-1)
    while len(kali2) < 6:
        if len(kali2) < 3:
            kali2.append(1)
        else:
            kali2.append(-1)
    size = 9
    tfidf = transposematriks(a, 0)
    arraytfidf = np.array(tfidf)
    kernelpolynomiallvl1 = kernelpoly(arraytfidf,tfidf)
    kernelpolynomiallevel1 = [kernelpolynomiallvl1[i:i + size] for i in range(0, len(kernelpolynomiallvl1), size)]
    matrikshessianlevel1 = matrikshessian(kernelpolynomiallevel1, kali)
    maksiterasi = 2
    ai = []
    ai2 = []
    gamma = 0.01
    c = 1
    transposehessianlevel1 = transposematriks(matrikshessianlevel1, 8)
    sequentialtraininglevel1 = sequentiallevel1(maksiterasi, matrikshessianlevel1, ai, gamma,transposehessianlevel1, c)
    supportvectorpositiflevel1 = supportvector(sequentialtraininglevel1[0:3])
    supportvectornegatiflevel1 = supportvector(sequentialtraininglevel1[3:9])
    xpositiflevel1 = mencarinilaix(sequentialtraininglevel1, supportvectorpositiflevel1)
    xnegatiflevel1 = mencarinilaix(sequentialtraininglevel1, supportvectornegatiflevel1)
    wpositiflevel1 = nilaiw(sequentialtraininglevel1, xpositiflevel1, kali)
    wnegatiflevel1 = nilaiw(sequentialtraininglevel1, xnegatiflevel1, kali)
    sumwpositiflevel1 = sum(wpositiflevel1)
    sumwnegatiflevel1 = sum(wnegatiflevel1)
    biaslevel1 = -0.5 * (sumwpositiflevel1 + sumwnegatiflevel1)
    return biaslevel1


matrikshessianlvl2 = []
def matrikshessian2(a,kali):
    for isi in range(len(a)):
        hessian = []
        for i in range(len(a[isi])):
            hasilhes2 = (a[i][isi] + 9) * kali[i]
            hessian.append(hasilhes2)
        matrikshessianlvl2.append(hessian)
    return matrikshessianlvl2
tfidfuji = []

#TESTING
def hitungtfidfuji(a,b):
    i = 0
    while i < len(a):
        tfidfuji.append([])
        j = 0
        while j < len(b[i]):
            tfidfuji[i].append(a[i] * b[i][j])
            j += 1
        i += 1
    return tfidfuji
#lena = transposetfidfuji
#lenb = tfidf
#a = arraytfidfuji
#b = arraytfidf
kernelpolydatauji = []
def kernelpolyuji(lena,lenb,a,b):
    for i in range(len(lena)):
        for j in range(len(lenb)):
            hasilkernel = (sum(a[i] * b[j]) + 1) ** 2
            kernelpolydatauji.append(hasilkernel)
    return kernelpolydatauji
#a = arraykerneluji
#b = array sequential
#c = kali
aiyixlvl1 = []
def aiyixlevel1(kernelpolyuji,arraykerneluji,arrayseq,kali):
    for i in range(len(kernelpolyuji)):
        for j in arrayseq:
            hasil = arraykerneluji[i] * j * kali
        aiyixlvl1.append(hasil)
    return aiyixlvl1
klasifikasi = []
def hasilklasifikasilevel1(aiyixlevel1,biaslevel1):
    for i in range(len(aiyixlevel1)):
        bbb = sum(aiyixlevel1[i]) + biaslevel1
        klasifikasi.append(bbb)
    return klasifikasi
aiyixlvl2 = []
def aiyixlevel2(kernelpolyuji,arraykerneluji,arrayseq,kali):
    for i in range(len(kernelpolyuji)):
        for j in arrayseq:
            hasil = arraykerneluji[i] * j * kali
        aiyixlvl2.append(hasil)
    return aiyixlvl2
klasifikasi2 = []
def hasilklasifikasilevel2(aiyixlevel1,biaslevel1):
    for i in range(len(aiyixlevel1)):
        bbb = sum(aiyixlevel1[i]) + biaslevel1
        klasifikasi2.append(bbb)
    return klasifikasi2

