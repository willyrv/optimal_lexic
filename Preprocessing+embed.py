
#imports
import spacy
import base64
import time
import collections
import re
import itertools
import nltk
import gensim
import kaleido
import plotly
import pandas as pd
import numpy as np
from spacy.lang.en import English
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from kaleido.scopes.plotly import PlotlyScope
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from scipy import sparse
from soyclustering import SphericalKMeans
from soyclustering import proportion_keywords
from sklearn.cluster import KMeans
from sklearn import metrics


ndocs = 1000

#datasets and log 
asrs = pd.read_csv('./ASRS_data.csv', sep="|",nrows = ndocs)
abr1 = pd.read_csv('./ASRS_Abr.csv')
abr2 = pd.read_csv('./ASRS_Abbreviations.csv')
f= open("logs.html","w")
a = '<h1>ASRS Data</h1>'
f.write(a)

#Removing some abreviations

#Preparation
abr1 = abr1[['Abr','word']]
abr1['Abr'] = abr1['Abr'].apply(lambda x : x.lower())
abr1['word'] = abr1['word'].apply(lambda x : x.lower())
trans_abr = abr1.set_index("Abr").T
abrd = trans_abr.to_dict('list')
asrs['Narrative'] = asrs['Narrative'].apply(lambda x : x.lower())
asrs["Narrative"] = asrs['Narrative'].str.replace('[^\w\s\d]','')
asrs["Narrative"] = asrs['Narrative'].str.replace(r'\d+','')


#function
def replace_abr(narr):
    # find all states that exist in the string
    abr_found = filter(lambda abr: abr in narr, abrd.keys())

    # replace each state with its abbreviation
    for abr in abr_found:
        narr = narr.replace(' '+abr+' ', ' '+abrd[abr][0]+' ')
    # return the modified string (or original if no states were found)
    return narr

#Replacement
asrs['Narrative'] = asrs['Narrative'].apply(replace_abr)


#Stopwords, lemmatization, tokenization
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
clean = lambda new : " ".join([lemmatizer.lemmatize(i) for i in re.sub("[^a-zA-Z]", " " ,
	new.lower()).split() if i not in stop_words]).split()
start_time = time.time()
asrs['cleaned']=asrs['Narrative'].apply(clean)
f.write('Time to preprocess : %s s <br>' % (time.time() - start_time))

#1st Vocab size
a = asrs['cleaned'].tolist()
b = list(set(list(itertools.chain.from_iterable(a))))
f.write('Initial vocabulary size : %s s <br>' % len(b))

freq_min = 10

#freq discrimination
#apply(low) takes a lot of time
biglist = list(itertools.chain.from_iterable(asrs['cleaned']))
count = collections.Counter(biglist).most_common()
def lowfreq(count):
    i = 0
    for word, num in reversed(count):
        i+=1
        if num > freq_min:
            return i          
ind = lowfreq(count)
to_remove = list(dict(count[-ind:]).keys())
low = lambda x : list(filter(lambda m : m not in to_remove,x))
start_time = time.time()
asrs['cleaned'] = asrs['cleaned'].apply(low)
f.write('Low frequency discrimination time : %s <br>' % (time.time() - start_time))


#2nd Vocab size
a = asrs['cleaned'].tolist()
b = list(set(list(itertools.chain.from_iterable(a))))
f.write('Initial vocabulary size : %s <br>' % len(b))


idfmin = 5.4

#IDF discrimination
#apply(low) also takes a lot of time
vectorizer = TfidfVectorizer(analyzer = lambda x : [w for w in x if w not in stop_words])
vectorized = vectorizer.fit(asrs["cleaned"])
idf = vectorized.idf_
idex = np.argwhere(idf>idfmin)
idex = [idx[0] for idx in idex]
vocab = vectorized.get_feature_names()
to_remove2 = {vocab[i] for i in idex}
low = lambda x : list(filter(lambda m : m not in to_remove2,x))
start_time = time.time()
asrs['cleaned'] = asrs['cleaned'].apply(low)
f.write('idf discrimination time : %s s <br>' % (time.time() - start_time))

#3rd Vocab size
a = asrs['cleaned'].tolist()
b = list(set(list(itertools.chain.from_iterable(a))))
f.write('Initial vocabulary size : %s <br>' % len(b))


#Word2vec and normalization
wlist = asrs['cleaned'].tolist()
model = Word2Vec(wlist, min_count=1,size= 50,workers=3, window =3, sg = 1)
model.init_sims(replace=True)
key_vec = model.wv
word_vec = key_vec.vectors
dic = key_vec.index2word

n_neigh = 15

#TSNE Plots
def plot_TSNE_local(keyed_vec, word, n_words):
    arr = []
    labels = []
    
    close_words = key_vec.similar_by_word(word,topn = n_words)
    
    arr.append(np.array(keyed_vec[word]))
    labels.append(word)
    
    for wrd in close_words:
        arr.append(np.array(keyed_vec[wrd[0]]))
        labels.append(wrd[0])
        
    tsne = TSNE(n_components = 2)
    Y = tsne.fit_transform(arr)
    
    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
                    x = x_coords, 
                    y = y_coords,
                    mode = 'markers + text',
                    text = labels,
                    textposition="bottom center"
                            ))
    
    title = str(n_words) + ' Closest words to the word ' + word + ' :'
    fig.update_layout(
            title = title,
            autosize=False,
            width=700,
            height=700)
    
    #fig.show()
    #fig.write_image("fig1.png")
    return fig
scope = PlotlyScope()

with open("twr.png", "wb") as file:
    file.write(scope.transform(plot_TSNE_local(key_vec,'twr',n_neigh), format="png"))

data_uri = base64.b64encode(open('twr.png', 'rb').read()).decode('utf-8')
img_tag = '<img src="data:image/png;base64,{0}">'.format(data_uri)
f.write(img_tag)


with open("smoke.png", "wb") as file:
    file.write(scope.transform(plot_TSNE_local(key_vec,'smoke',n_neigh), format="png"))

data_uri = base64.b64encode(open('smoke.png', 'rb').read()).decode('utf-8')
img_tag = '<img src="data:image/png;base64,{0}">'.format(data_uri)
f.write(img_tag)

with open("flight.png", "wb") as file:
    file.write(scope.transform(plot_TSNE_local(key_vec,'flight',n_neigh), format="png"))

data_uri = base64.b64encode(open('flight.png', 'rb').read()).decode('utf-8')
img_tag = '<img src="data:image/png;base64,{0}">'.format(data_uri)
f.write(img_tag)

with open("terrain.png", "wb") as file:
    file.write(scope.transform(plot_TSNE_local(key_vec,'terrain',n_neigh), format="png"))

data_uri = base64.b64encode(open('terrain.png', 'rb').read()).decode('utf-8')
img_tag = '<img src="data:image/png;base64,{0}">'.format(data_uri)
f.write(img_tag)

with open("rain.png", "wb") as file:
    file.write(scope.transform(plot_TSNE_local(key_vec,'rain',n_neigh), format="png"))

data_uri = base64.b64encode(open('rain.png', 'rb').read()).decode('utf-8')
img_tag = '<img src="data:image/png;base64,{0}">'.format(data_uri)
f.write(img_tag)

with open("vfr.png", "wb") as file:
    file.write(scope.transform(plot_TSNE_local(key_vec,'vfr',n_neigh), format="png"))

data_uri = base64.b64encode(open('vfr.png', 'rb').read()).decode('utf-8')
img_tag = '<img src="data:image/png;base64,{0}">'.format(data_uri)
f.write(img_tag)

with open("passenger.png", "wb") as file:
    file.write(scope.transform(plot_TSNE_local(key_vec,'seat',n_neigh), format="png"))

data_uri = base64.b64encode(open('passenger.png', 'rb').read()).decode('utf-8')
img_tag = '<img src="data:image/png;base64,{0}">'.format(data_uri)
f.write(img_tag)

Y = word_vec

n_clusters = 15

#faut normaliser

#spherical K means clustering
km = KMeans(n_clusters=n_clusters)
km.fit_transform(word_vec)
print("Top terms per cluster:")
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = dic
for i in range(n_clusters):
    print("Cluster %d:" % i, end='')
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind], end='')
    print()


#faire kmeans sur les tsne, kmeans na pas de sens du tout..
#voir pk kmeans fait de la merde

#feed into LDA?