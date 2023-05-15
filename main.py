from typing import List
import function as func
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import math
import numpy as np

latihinformasi = []
latihkeluhan = []
latihsaran = []
with open ('datalatihinformasipengujian.txt') as datalatih:
    for element in datalatih.readlines():
        latih1 = element.replace('\n','')
        latihinformasi.append(latih1)

with open ('datalatihkeluhanpengujian.txt') as datalatih2:
    for element in datalatih2.readlines():
        latih2 = element.replace('\n','')
        latihkeluhan.append(latih2)

with open ('datalatihsaranpengujian.txt') as datalatih3:
    for element in datalatih3.readlines():
        latih3 = element.replace('\n','')
        latihsaran.append(latih3)

latihinformasi = func.preprocessing(latihinformasi)
latihkeluhan = func.preprocessing(latihkeluhan)
latihsaran = func.preprocessing(latihsaran)

#pembobotan kata
#terms
gabungan = latihinformasi, latihkeluhan, latihsaran

terms = []
for dok in gabungan:
    for docs in dok:
        for kata in docs:
            if kata not in terms:
                terms.append(kata)

#tf
tf_info = []
for term in terms:
    panjang = []
    for doc in latihinformasi:
        panjang.append(float(doc.count(term)))
    tf_info.append(panjang)

tf_kel = []
for term in terms:
    panjang = []
    for doc in latihkeluhan:
        panjang.append(float(doc.count(term)))
    tf_kel.append(panjang)

tf_saran = []
for term in terms:
    panjang = []
    for doc in latihsaran:
        panjang.append(float(doc.count(term)))
    tf_saran.append(panjang)

tf = []
for isi in range(len(tf_info)):
    tf.append(tf_info[isi] + tf_kel[isi] + tf_saran[isi])


#df
dfinfo = []
dfkel = []
dfsaran = []
for i in tf_info:
    dfinfo.append(sum(i))
pandadfinfo = pd.DataFrame(dfinfo, index= terms)
for i in tf_kel:
    dfkel.append(sum(i))
pandadfkel = pd.DataFrame(dfkel, index = terms)
for i in tf_saran:
    dfsaran.append(sum(i))
pandadfsaran = pd.DataFrame(dfsaran, index = terms)
dfgabungan = []
for isi in dfinfo,dfkel,dfsaran:
    dfgabungan.append(isi)
dftotal = [sum(i) for i in zip(*dfgabungan)]

#idf
idf = []
for isi in dftotal:
    idf.append(float(math.log10(108/isi)))

#tfidf
tfidf = func.hitungtfidf(idf,tf)

#SVM TRAINING
#LEVEL 1
kali = []
kali2 = []
while len(kali) < 108:
    if len(kali) < 36:
        kali.append(1)
    else:
        kali.append(-1)

while len(kali2) < 72:
    if len(kali2) < 36:
        kali2.append(1)
    else:
        kali2.append(-1)
size = 108
tfidf = func.transposematriks(tfidf,0)
arraytfidf = np.array(tfidf)
kernelpolynomiallevel1 = func.kernelpoly(arraytfidf,tfidf)
kernelpolynomiallevel1 = [kernelpolynomiallevel1[i:i+size] for i in range(0, len(kernelpolynomiallevel1), size)]
matrikshessianlevel1 = func.matrikshessian(kernelpolynomiallevel1,kali)
maksiterasi = 1000
ai = []
ai2 = []
while len(ai) < 108:
    ai.append(0)
while len(ai2) < 72:
    ai2.append(0)
gamma = 0.000001
c = 0.001
transposehessianlevel1 = func.transposematriks(matrikshessianlevel1,0)
sequentialtraininglevel1 = func.sequentiallevel1(maksiterasi,matrikshessianlevel1,ai,gamma,transposehessianlevel1,c)
supportvectorpositiflevel1 = func.supportvector(sequentialtraininglevel1[0:36])
supportvectornegatiflevel1 = func.supportvector(sequentialtraininglevel1[36:108])
xpositiflevel1 = func.mencarinilaix(sequentialtraininglevel1,supportvectorpositiflevel1)
xnegatiflevel1 = func.mencarinilaix(sequentialtraininglevel1,supportvectornegatiflevel1)
wpositiflevel1 = func.nilaiw(sequentialtraininglevel1,xpositiflevel1,kali)
wnegatiflevel1 = func.nilaiw(sequentialtraininglevel1,xnegatiflevel1,kali)
sumwpositiflevel1 = sum(wpositiflevel1)
sumwnegatiflevel1 = sum(wnegatiflevel1)
biaslevel1 = -0.5*(sumwpositiflevel1+sumwnegatiflevel1)
print(biaslevel1)

#LEVEL 2
kernelpolynomiallevel2 = []
del kernelpolynomiallevel1[0:36]
for i in kernelpolynomiallevel1:
    del i[0:36]
    kernelpolynomiallevel2.append(i)
matrikshessianlevel2 = func.matrikshessian(kernelpolynomiallevel2,kali2)
transposehessianlevel2 = func.transposematriks(matrikshessianlevel2,0)
sequentialtraininglevel2 = func.sequentiallevel1(maksiterasi,matrikshessianlevel2,ai2,gamma,transposehessianlevel2,c)
supportvectorpositiflevel2 = func.supportvector(sequentialtraininglevel2[0:36])
supportvectornegatiflevel2 = func.supportvector(sequentialtraininglevel2[36:72])
xpositiflevel2 = func.mencarinilaix(sequentialtraininglevel2,supportvectorpositiflevel2)
xnegatiflevel2 = func.mencarinilaix(sequentialtraininglevel2,supportvectornegatiflevel2)
wpositiflevel2 = func.nilaiw(sequentialtraininglevel2,xpositiflevel2,kali2)
wnegatiflevel2 = func.nilaiw(sequentialtraininglevel2,xnegatiflevel2,kali2)
sumwpositiflevel2 = sum(wpositiflevel2)
sumwnegatiflevel2 = sum(wnegatiflevel2)
biaslevel2 = -0.5*(sumwpositiflevel2+sumwnegatiflevel2)
print(biaslevel2)



#SVM TESTING
#preprocessing testing
uji = []
with open ('data uji 9 banding 1.txt') as datauji:
    for element in datauji.readlines():
        datauji = element.replace('\n','')
        uji.append(datauji)
uji = func.preprocessing(uji)

#pembobotanuji
tf_uji = []
for term in terms:
    panjang = []
    for doc in uji:
        panjang.append(float(doc.count(term)))
    tf_uji.append(panjang)
tfidfuji = func.hitungtfidfuji(idf,tf_uji)
transposetfidfuji = func.transposematriks(tfidfuji,0)
arraytfidfuji = np.array(transposetfidfuji)
kernelpolyuji = func.kernelpolyuji(transposetfidfuji,tfidf,arraytfidfuji,arraytfidf)
kernelpolyujitranspose = [kernelpolyuji[i:i+size] for i in range(0, len(kernelpolyuji), size)]
arrayseq = np.array(sequentialtraininglevel1)
arraykerneluji = np.array(kernelpolyujitranspose)
arrayseq2 = np.array(sequentialtraininglevel2)
kelashasil=[]

#LEVEL 1
aiyixlvl1 = func.aiyixlevel1(kernelpolyujitranspose,arraykerneluji,arrayseq,kali)
hasilklasifikasilevel1 = func.hasilklasifikasilevel1(aiyixlvl1,biaslevel1)
print(hasilklasifikasilevel1)
for ele in hasilklasifikasilevel1:
    if ele >0:
        kelashasil.append('informasi')


#LEVEL 2
kernelpolyuji2 = []
del kernelpolyujitranspose[0:4]#16 + 4
for i in kernelpolyujitranspose:
    del i[0:36]#24 - 4
for i in kernelpolyujitranspose:
    kernelpolyuji2.append(i)
#kernelpolyuji2 = [kernelpolyuji2[i:i+size] for i in range(0, len(kernelpolyuji2), size)]

arraykerneluji2 = np.array(kernelpolyuji2)
aiyixlvl2 = func.aiyixlevel2(kernelpolyuji2,arraykerneluji2,arrayseq2,kali2)
hasilklasifikasilevel2 = func.hasilklasifikasilevel2(aiyixlvl2,biaslevel2)
print(hasilklasifikasilevel2)
for ele in hasilklasifikasilevel2:
    if ele > 0:
        kelashasil.append('keluhan')
    else:
        kelashasil.append('saran')

with open('data uji 9 banding 1.txt', 'r') as f:
    print(list(zip(kelashasil, f.readlines())))

print(len(kelashasil))
hasilinfo = []
hasilkeluhan = []
hasilsaran = []
for i in kelashasil[0:4]:
    if i == "informasi":
        hasilinfo.append(i)
for i in kelashasil[4:8]:
    if i == "keluhan":
        hasilkeluhan.append(i)
for i in kelashasil[8:12]:
    if i == "saran":
        hasilsaran.append(i)

akurasiinfo = (len(hasilinfo)/4)*100
print(akurasiinfo)
akurasikeluhan = (len(hasilkeluhan)/4)*100
print(akurasikeluhan)
akurasisaran = (len(hasilsaran)/4)*100
print(akurasisaran)
akurasi = ((akurasiinfo+akurasikeluhan+akurasisaran)/3)
print(akurasi,"%")
asd = ((len(hasilinfo)+len(hasilkeluhan)+len(hasilsaran))/60)*100
print(asd)