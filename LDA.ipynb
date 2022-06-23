#!/usr/bin/env python
# coding: utf-8

# # Import libraries

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
import pandas as pd
import numpy as np
import re
# import nltk
import spacy
# import gensim
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from matplotlib.collections import LineCollection
from nltk.tokenize import ToktokTokenizer
from nltk.stem import wordnet
from nltk.corpus import stopwords
from string import punctuation

# import pyLDAvis.sklearn

import pickle
import itertools
warnings.filterwarnings('ignore')

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import jaccard_score, make_scorer, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score, KFold, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.svm import LinearSVC

import glob
import json
import os
import string as st

# nltk.download("stopwords")
# nltk.download("wordnet")
sns.set()


# In[2]:


# ! python -m pip index versions gensim


# # Import data

# In[3]:


# BASE_PATH = "./" #os.path.dirname(os.path.abspath('__file__'))
# DATA_PATH = os.path.join(BASE_PATH, 'ch')

# import html pages previously parsed as json files
# files = glob.glob(DATA_PATH+'/**/*.json', recursive=True)
# print("Found",len(files),"html pages as json files")

# df_json = pd.DataFrame(columns=["ROW_ID", "SOURCE", "Body"])
# for i, file in enumerate(files):
    # with open(file,"r") as json_file:
        # jf = json.load(json_file)
        # df_json = df_json.append({
                                         # "ROW_ID": i,
                                         # "Body":  jf["page_content"],
                                         # "SOURCE": jf["url"]
                                          # }, ignore_index=True)

# df_pdf = pd.read_csv("C:/Users/luca.perrozzi/Downloads/ontology-docs/part1/parsed/test.csv")
# print("Found",len(df_pdf),"pdf parsed as text files")
df = pd.read_pickle("CONTACT_REPORT2_REPORT_DETAILS_hk_plus_sg_combined.pickle")
df = df.rename(columns={"CONTACT_REPORT2_CONTACT_REPORT_ID": "SOURCE", "ALL": "Body"})
df = df.fillna("")


# In[4]:


print(len(df))
display(df.head())


# In[5]:


# df_tags = pd.concat([df_json, df_pdf])
df_tags = df.sample(frac=1, random_state=1)
print("Found",len(df_tags),"items in total. Items contatenated and randomized.")


# In[6]:


# df_tags = df_tags.drop('Unnamed: 0', axis=1)
# df_tags = df_tags.drop('ROW_ID', axis=1)
# df_tags = df_tags[["SOURCE", "Body"]]
with pd.option_context('display.max_colwidth', -1, 'display.max_columns', 10):
    display(df_tags)


# # Text pre-processing
# 
# 
# *   Removing html format using Beautiful Soup library
# *   Transforming abbreviations
# *   Lowering text
# *   Removing stop words
# *   Lemmatizing words - grouping together the inflected forms of a word
# *   Removing verbs and adjectives, since they don't give a valuable information about the post

# In[7]:


def clean_text(text):
    ''' Lowering text and removing undesirable marks
    Parameter:
    text: document to be cleaned    
    '''
    text = " ".join([word.lower() if word != word.upper() else word for word in text.split()])
    
    # # text = re.sub('\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b', " ", text) #remove emails
    # # text = re.sub(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$", " ", text) #remove emails
    # text = re.sub(r'[A-Za-z0-9_.+-]*@[A-Za-z]*\.?[A-Za-z0-9]*', " ", text) # remove emails
    # text = re.sub(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', " ", text) # remove emails
    text = re.sub(r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)', " ", text) # remove emails
    # text = re.sub(r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)", " ", text) # remove emails
    
    text = re.sub(r"\'\n", " ", text)
    text = re.sub(r"\'\xa0", " ", text)
    text = re.sub('\s+', ' ', text) # matches all whitespace characters
    text = text.strip(' ')
    return text


def strip_list_noempty(mylist):
    newlist = (item.strip() if hasattr(item, 'strip') else item for item in mylist)
    return [item for item in newlist if item != '']


punct = punctuation
regex = re.compile('[%s]' % re.escape(punct))
token = ToktokTokenizer()
def clean_punct(text): 
    ''' Remove all the punctuation from text, unless it's part of an important
    tag (ex: c++, c#, etc)
    Parameter:
    text: text to remove punctuation from it
    '''
    text = text.replace("&", " ")
    words = token.tokenize(text) # text.split() # 
    punctuation_filtered = []
    remove_punctuation = str.maketrans(' ', ' ', punct)
    
    for w in words:
        w = re.sub('^[0-9]*', " ", w)
        punctuation_filtered.append(regex.sub(' ', w))
  
    filtered_list = strip_list_noempty(punctuation_filtered)
        
    return ' '.join(map(str, filtered_list))


punctuation_chars = st.punctuation+'–'+"•"
def remove_punct(text):
    return ("".join([ch for ch in text if ch not in punctuation_chars]))
# print(punctuation_chars)


# function to remove numbers
def remove_numbers(text):
    # define the pattern to keep
    pattern = r'[^a-zA-z.,!?/:;\"\'\s]' # Better would be to keep the numbers within a word, like post1q, and remove the numbers like 1000
    return re.sub(pattern, ' ', text)

# stop_words = set(stopwords.words("english"))

stop_words = [ "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", 
"they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", 
"do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", 
"before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", 
"all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "d", 
    # SPECIFIC STOPWORDS
    "gmtpriority", "id", "amp", "also",
             ] 

def stopWordsRemove(text):
    ''' Removing all the english stop words from a corpus
    Parameter:
    text: document to remove stop words from it
    '''
    
    # stop_words = set(stopwords.words("english"))
    words = token.tokenize(text)
    # words = text.split()
    filtered = [w for w in words if not w in stop_words and len(w)>1]
    
    return ' '.join(map(str, filtered))


nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
# lemma = wordnet.WordNetLemmatizer()
def lemmatization(texts):
    ''' It keeps the lemma of the words (lemma is the uninflected form of a word),
    and deletes the undesidered POS tags
    Parameters:
    texts (list): text to lemmatize
    allowed_postags (list): list of allowed postags, like NOUN, ADL, VERB, ADV
    '''
    doc = nlp(texts) 
    texts_out = []
    
    for token in doc:
        texts_out.append(token.lemma_)
     
    texts_out = ' '.join(texts_out)

    return texts_out


# In[8]:


_id = "CRSG754190" #"CRSG807467" #"CRSG462899"
example_text =  df_tags[df_tags["SOURCE"] == _id]["SOURCE"].values[0]
print("SOURCE:",example_text)
example_text =  df_tags[df_tags["SOURCE"] == _id]["Body"].values[0]
# example_text =  "dad@d calling all.de@juliusbaer.comlucasr ciao & gmtprioritY"
# example_text =  df_tags[df_tags["Body"].str.contains(" amp ")]["Body"].values[0]

print("ORIGINAL:",example_text)
example_text = clean_text(example_text)
print("\nclean_text:",example_text)
example_text = clean_punct(example_text)
print("\nclean_punct:",example_text)
example_text = remove_punct(example_text)
print("\nremove_punct:",example_text)
example_text = remove_numbers(example_text)
print("\nremove_numbers:",example_text)
example_text = stopWordsRemove(example_text)
print("\nstopWordsRemove:",example_text)
example_text = lemmatization(example_text)
print("\nlemmatization:",example_text)


# example_test
# with pd.option_context('display.max_colwidth', -1, 'display.max_columns', 10):
    # display(df_tags)


# In[9]:


# print("Remove HTML tags")
# df_tags['Body'] = df_tags['Body'].apply(lambda x: BeautifulSoup(x).get_text())
print("Apply generic text cleaning")
df_tags['Body'] = df_tags['Body'].apply(lambda x: clean_text(x))
print("Clean punctuation")
df_tags['Body'] = df_tags['Body'].apply(lambda x: clean_punct(x)) 
print("Remove punctuation")
df_tags['Body'] = df_tags['Body'].apply(lambda x: remove_punct(x)) 
print("Remove numbers")
df_tags['Body'] = df_tags['Body'].apply(lambda x: remove_numbers(x)) 
print("Remove stop words")
df_tags['Body'] = df_tags['Body'].apply(lambda x: stopWordsRemove(x)) 
print("Apply lemmatization")
df_tags['Body'] = df_tags['Body'].apply(lambda x: lemmatization(x))


# In[10]:


df_tags


# In[11]:


with pd.option_context('display.max_rows', 100,'display.max_colwidth', -1): 
    # display(df_tags[df_tags["Body"].str.contains(" we ")].head(3))
    display(len(df_tags[df_tags["CONTACT_REPORT2_REPORT_DETAILS"].str.contains("This e-mail is")].head()))
    display((df_tags[df_tags["CONTACT_REPORT2_REPORT_DETAILS"].str.contains("This e-mail is")].head(1)))
    


# In[12]:


print("Advisory Session", len(df_tags[df_tags["CONTACT_REPORT2_REPORT_DETAILS"].str.contains("This e-mail is")]))
df_tags =  df_tags[~df_tags["CONTACT_REPORT2_REPORT_DETAILS"].str.contains("This e-mail is")]
print("Left entries",len(df_tags))


# In[13]:


# df_tags.to_pickle("CONTACT_REPORT2_REPORT_DETAILS_hk_plus_sg_combined_cleaned2.pickle")
df_tags = pd.read_pickle("CONTACT_REPORT2_REPORT_DETAILS_hk_plus_sg_combined_cleaned2.pickle")


# # Word frequency

# In[14]:


words = dict()

for sentence in df_tags['Body']:
    for word in sentence.split(): #token.tokenize(sentence):
        if word not in words.keys(): # we can use nltk.FraqDist instead
            words[word] = 1
        elif word in words.keys():
            words[word] += 1
            
occurences = []
    
for word, occurence in words.items():
    occurences.append([word, occurence])
            
occurences.sort(key = lambda x:x[1], reverse = True)
words_occurences = occurences[0:51]
print(words_occurences)


# In[15]:


# Graph showing the words most used in the items
plt.figure(figsize=(20, 10))
ax = plt.axes()
y_axis = [i[1]  for i in words_occurences]
x_axis = [k for k,i in enumerate(words_occurences)]
label_x = [i[0] for i in words_occurences]
plt.xticks(rotation=90, fontsize=15)
ax = ax.set(xlabel="Word", ylabel="Number of occurences")

plt.bar(label_x, y_axis, color='purple')
plt.xticks(rotation=85)
plt.title("Words most used in the posts",fontsize=20)
plt.show()


# # Test using only TF.IDF

# In[16]:


import csv
from sklearn.feature_extraction.text import TfidfTransformer #to compute the IDF


#helper functions
def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
 
def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]
 
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
 
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results

#Using the TF.IDF technique to extract the keywords from the corpus
#create a vocabulary of words
#ignore words that appear in more than 85% of the document
def TfIdf(ngram):
    cv = CountVectorizer(max_df = 0.95, 
                         # max_features = 10000, 
                         ngram_range=(ngram,ngram))    
    word_count_vector = cv.fit_transform(corpus)

    #to see the words in the vocabulary use: list(cv.vocabulary_.keys())[:10]
    print("words length:",ngram," - vocabulary length:",len(cv.vocabulary_.keys()))

    #calculate the IDF
    #WARNING: ALWAYS USE IDF ON A LARGE CORPUS
    tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(word_count_vector)


    # do this once, this is a mapping of index 
    feature_names = cv.get_feature_names()
    
    return cv, tfidf_transformer, feature_names

def extract_features_TfIdf(doc, cv, tfidf_transformer, feature_names):
    #generate tf-idf for the given document
    tf_idf_vector = tfidf_transformer.transform(cv.transform([doc]))
    return [feature_names, tf_idf_vector, doc]

def extract_tags_TfIdf(doc, cv, tfidf_transformer, feature_names, 
                            cv2, tfidf_transformer2, feature_names2):
    
    #perform TF.IDF on BOW
    tfidf = extract_features_TfIdf(doc, cv, tfidf_transformer, feature_names)
    #sort the tf-idf vectors by descending order of scores
    sorted_items=sort_coo(tfidf[1].tocoo())
    # print(sorted_items[:100])
    #extract only the top n; n here is 5
    keywords = extract_topn_from_vector(tfidf[0],sorted_items,10)
    top_tags = ""
    for y in keywords.keys():
        top_tags = top_tags + (y + "|")                                
    
    #perform TF.IDF on bi-grams
    tfidf2 = extract_features_TfIdf(doc, cv2, tfidf_transformer2, feature_names2)
    #sort the tf-idf vectors by descending order of scores
    sorted_items=sort_coo(tfidf2[1].tocoo())
    #extract only the top n; n here is 5
    keywords=extract_topn_from_vector(tfidf2[0],sorted_items,10)
    top_tags2 = ""
    for y in keywords.keys():
        top_tags2 = top_tags2 + (y + "|")                                

    return top_tags, top_tags2


# In[17]:


df = df_tags.copy(deep=True)
#creating the corpus for all the articles
corpus = df_tags["Body"].tolist()
df['tags'] = ""
df['tags2'] = ""

df = df.reset_index()
df = df.drop('index', axis=1)

cv, tfidf_transformer, feature_names = TfIdf(1)
cv2, tfidf_transformer2, feature_names2 = TfIdf(2)


# In[ ]:


for x in range(len(corpus)):
    
    doc = corpus[x]
    
    top_tags, top_tags2 = extract_tags_TfIdf(doc, cv, tfidf_transformer, feature_names,
                                                  cv2, tfidf_transformer2, feature_names2)
    df.loc[x,'tags'] = top_tags    
    df.loc[x,'tags2'] = top_tags2
        
    if x < 100:
        print("document",x)
        print("unigrams:",top_tags)
        print("bigrams: ",top_tags2)

        # break


# In[ ]:


# df.to_excel("../../test_tags.xlsx", engine="xlsxwriter")


# In[ ]:


# Program of the recommendation system using tf.idf
# doc = input('Insert text to tag: ')
# top_tags, top_tags2 = extract_tags_TfIdf(doc, cv, tfidf_transformer, feature_names,
                                                  # cv2, tfidf_transformer2, feature_names2)
# print('Recommended tags are:\nUnigrams:',top_tags,"\nBigrams:",top_tags2)


# In[ ]:


# tags_from_taxonomy = pd.read_excel("C:/Users/luca.perrozzi/Downloads/taxonomy_tags.xlsx")
# tags_from_taxonomy = tags_from_taxonomy.drop('Unnamed: 0', axis=1)
# tags_from_taxonomy.head()


# In[ ]:


df


# In[ ]:


df.to_pickle("CONTACT_REPORT2_REPORT_DETAILS_hk_plus_sg_combined_cleaned2_tfidf_tags.pickle")


# In[ ]:


raise


# # Topic classification

# ### TF-IDF : Term Frequency - Inverse Document Frequency
# The term frequency is the number of times a term occurs in a document. Inverse document frequency is an inverse function of the number of documents in which that a given word occurs.<p>
# The product of these two terms gives tf-idf weight for a word in the corpus. The higher the frequency of occurrence of a word, lower is it's weight and vice-versa. This gives more weightage to rare terms in the corpus and penalizes more commonly occuring terms.<p>
# Other widely used vectorizer is Count vectorizer which only considers the frequency of occurrence of a word across the corpus.
# 

# In[ ]:


df = df_tags


# In[ ]:


# Sampling dataset
vectorizer_X = TfidfVectorizer(analyzer='word', min_df=2, max_df = 0.95, 
                                   strip_accents = None, encoding = 'utf-8', 
                                   preprocessor=None, 
                                   token_pattern=r"(?u)\S\S+", # Need to repeat token pattern
                                   # max_features=10000
                                  )

# 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    df['Body'], df['SOURCE'], test_size=0.2,train_size=0.8, random_state=0)

# TF-IDF matrices
X_tfidf_train = vectorizer_X.fit_transform(X_train)
X_tfidf_test = vectorizer_X.transform(X_test)


# In[ ]:


# Sampling dataset for 2gram model

vectorizer_X_2gram = TfidfVectorizer(analyzer='word', min_df=2, max_df = 0.95, 
                                   strip_accents = None, encoding = 'utf-8', 
                                   preprocessor=None, ngram_range=(2,2),
                                   token_pattern=r"(?u)\S\S+", # Need to repeat token pattern
                                   # max_features=10000
                                  )

# 80/20 split
X_2gram_train, X_2gram_test, y_2gram_train, y_2gram_test = train_test_split(
    df['Body'], df['SOURCE'], test_size=0.2,train_size=0.8, random_state=0)

# TF-IDF matrices
X_2gram_tfidf_train = vectorizer_X_2gram.fit_transform(X_2gram_train)
X_2gram_tfidf_test = vectorizer_X_2gram.transform(X_2gram_test)


# # Unsupervised models using LDA

# In[ ]:


ntopic_to_use = 20


# In[ ]:


def print_top_words(model, feature_names, n_top_words, data):
    ''' It shows the top words from the different clusters of a model
    
    Parameters:
    
    model: model 
    feature_names: different words to show 
    n_top_words (int): number of words to print for each feature 
    data: data for the model
    '''

    list_topics = []
    list_occurences = []
    n_topics = model.n_components

    for i in model.transform(data):
        list_topics.append(i.argmax())
    
    for topic in range(n_topics):
        list_occurences.append(list_topics.count(topic))

    top_topics = sorted(range(len(list_occurences)), 
                        key=lambda k: list_occurences[k], reverse=True)
    
    for topic_idx, topic_id in zip(range(1, n_topics + 1), top_topics):
        message = "Tag #%d: " % topic_idx
        message += " / ".join([feature_names[i]
                             for i in model.components_[topic_id].argsort()[:-n_top_words - 1:-1]])
        print(message)
    
    print()


# In[ ]:


def lda(vectorizer, data_train, data_test):

    ''' Showing the perplexity score for several LDA models with different values
    for n_components parameter, and printing the top words for the best LDA model
    (the one with the lowest perplexity)

    Parameters:

    vectorizer: TF-IDF convertizer                                              
    data_train: data to fit the model with
    data_test: data to test
    '''

    # number of topics 
    n_top_words = 10
    best_perplexity = np.inf
    best_lda = 0
    start_ntopics = 5
    best_ntopics = 5
    perplexity_list = []
    n_topics_list = []
    print("Extracting term frequency features for LDA...")

    for n_topics in np.linspace(start_ntopics, 25, 5, dtype='int'):
        lda_model = LatentDirichletAllocation(n_components=n_topics, max_iter=5,
                                        learning_method='online',
                                        learning_offset=50.,
                                        random_state=0).fit(data_train)
        n_topics_list.append(n_topics)
        perplexity = lda_model.perplexity(data_test)
        perplexity_list.append(perplexity)

        print("Number of topics =",str(n_topics),"---> perplexity =",perplexity)
        
        # Perplexity is defined as exp(-1. * log-likelihood per word)
        # Perplexity: The smaller the better
        if perplexity <= best_perplexity:
            best_perplexity = perplexity
            best_lda = lda_model
            best_ntopics = n_topics
                                
    plt.title("Evolution of perplexity score depending on number of topics")
    plt.xlabel("Number of topics")
    plt.ylabel("Perplexity (the lower the better)")
    plt.plot(n_topics_list, perplexity_list)
    plt.show()

    print("\n The tags in the LDA model :")
    tf_feature_names = vectorizer.get_feature_names()
    print_top_words(best_lda, tf_feature_names, n_top_words, data_test)
    
    return best_ntopics


# In[ ]:


# LDA model (BOW)
best_ntopics = lda(vectorizer_X, X_tfidf_train, X_tfidf_test)
print("Best LDA model for BoW reached for number of topics:",best_ntopics)


# In[ ]:


best_lda = LatentDirichletAllocation(n_components=ntopic_to_use, max_iter=10,
                                        learning_method='online',
                                        learning_offset=50.,
                                        random_state=0).fit(X_tfidf_train)

# pyLDAvis.enable_notebook()
# panel = pyLDAvis.sklearn.prepare(best_lda, X_tfidf_test, vectorizer_X, mds='tsne')
# panel


# In[ ]:


# 2gram model without code lines

best_ntopics = lda(vectorizer_X_2gram, X_2gram_tfidf_train, X_2gram_tfidf_test)
print("Best LDA model for bi-gram reached for number of topics:",best_ntopics)


# In[ ]:


best_lda = LatentDirichletAllocation(n_components=ntopic_to_use, max_iter=10,
                                        learning_method='online',
                                        learning_offset=50.,
                                        random_state=0).fit(X_2gram_tfidf_train)

# pyLDAvis.enable_notebook()
# panel = pyLDAvis.sklearn.prepare(best_lda, X_2gram_tfidf_test, vectorizer_X_2gram, mds='tsne')
# panel


# ## Compute Topic based on single words

# In[ ]:


def Recommend_tags_lda_test(X_tfidf_test, X_train):
    
    ''' Recomendation system for items based on a lda model, it returns up to 5 tags.
    Parameters:
    X_tfidf_test: the posts after TF-IDF transformation
    X_train: data to fit the model with
    '''
    df_tags_test_lda = pd.DataFrame(index=[i for i in range(X_tfidf_test.shape[0])], 
             columns=['0.010', '0.011', '0.012', '0.013'])
    corpus = X_tfidf_test
    n_topics = ntopic_to_use

    X_tfidf_train = vectorizer_X.fit_transform(X_train)

    lda_model = LatentDirichletAllocation(n_components=n_topics, max_iter=10,
                                        learning_method='online',
                                        learning_offset=50.,
                                        random_state=0).fit(X_tfidf_train)
    corpus_projection = lda_model.transform(corpus)
    
    feature_names = vectorizer_X.get_feature_names()
    lda_components = lda_model.components_ / lda_model.components_.sum(axis=1)[:, np.newaxis] # normalization

    for column, threshold in zip(range(4), [0.010, 0.011, 0.012, 0.013]): #  threshold to exceed to be considered as a relevant tag

        for text in range(corpus.shape[0]):
            list_scores = []
            list_words = []

            for topic in range(n_topics):
                topic_score = corpus_projection[text][topic]

                for (word_idx, word_score) in zip(lda_components[topic].argsort()[:-5:-1], sorted(lda_components[topic])[:-5:-1]):
                    score = topic_score*word_score

                    if score >= threshold:
                        list_scores.append(score)
                        list_words.append(feature_names[word_idx])

            results = [tag for (y,tag) in sorted(zip(list_scores,list_words), 
                                                 key=lambda pair: pair[0], reverse=True)]
            df_tags_test_lda.iloc[text, column] = results[:5] #maximum five tags
        
        break

    return df_tags_test_lda


# In[ ]:


df_tags_test_lda = Recommend_tags_lda_test(X_tfidf_test, X_train)


# In[ ]:


df_tags_test_lda.head(10)


# ## Compute Topic based on bi-grams

# In[ ]:


def Recommend_tags_lda_test_bigram(X_2gram_tfidf_test, X_2gram_train):
    
    ''' Recomendation system for items based on a lda model, it returns up to 5 tags.
    Parameters:
    X_2gram_tfidf_test: the posts after TF-IDF transformation
    X_2gram_train: data to fit the model with
    '''
    df_tags_test_lda = pd.DataFrame(index=[i for i in range(X_2gram_tfidf_test.shape[0])], 
             columns=['0.010', '0.011', '0.012', '0.013'])
    corpus = X_2gram_tfidf_test
    n_topics = ntopic_to_use

    X_tfidf_train = vectorizer_X_2gram.fit_transform(X_2gram_train)

    lda_model = LatentDirichletAllocation(n_components=n_topics, max_iter=10,
                                        learning_method='online',
                                        learning_offset=50.,
                                        random_state=0).fit(X_tfidf_train)
    corpus_projection = lda_model.transform(corpus)
    
    feature_names = vectorizer_X_2gram.get_feature_names()
    lda_components = lda_model.components_ / lda_model.components_.sum(axis=1)[:, np.newaxis] # normalization

    for column, threshold in zip(range(4), [0.010, 0.011, 0.012, 0.013]): #  threshold to exceed to be considered as a relevant tag

        for text in range(corpus.shape[0]):
            list_scores = []
            list_words = []

            for topic in range(n_topics):
                topic_score = corpus_projection[text][topic]

                for (word_idx, word_score) in zip(lda_components[topic].argsort()[:-5:-1], sorted(lda_components[topic])[:-5:-1]):
                    score = topic_score*word_score

                    if score >= threshold:
                        list_scores.append(score)
                        list_words.append(feature_names[word_idx])

            results = [tag for (y,tag) in sorted(zip(list_scores,list_words), 
                                                 key=lambda pair: pair[0], reverse=True)]
            df_tags_test_lda.iloc[text, column] = results[:5] #maximum five tags
        break
        
    return df_tags_test_lda


# In[ ]:


df_tags_test_lda_bigram = Recommend_tags_lda_test_bigram(X_2gram_tfidf_test, X_2gram_train)


# In[ ]:


df_tags_test_lda_bigram.head(10)


# # Compute topic on external text

# In[ ]:


def Recommend_tags_lda(text, X_train):
    
    ''' Recomendation system for stackoverflow posts based on a lda model, 
    it returns up to 5 tags.

    Parameters:

    text: the stackoverflow post of user
    X_train: data to fit the model with
    '''

    text = clean_text(text)
    text = clean_punct(text)
    text = remove_punct(text)
    text = remove_numbers(text)
    text = stopWordsRemove(text)
    text = clean_text(text)
    text = lemmatization(text)
        
    n_topics = 10
    threshold = 0.011
    list_scores = []
    list_words = []
    used = set()

    vectorizer_X.fit(X_train)
    text_tfidf = vectorizer_X.transform([text])

    lda_model = LatentDirichletAllocation(n_components=n_topics, max_iter=5,
                                        learning_method='online',
                                        learning_offset=50.,
                                        random_state=0).fit(X_tfidf_train)
    text_projection = lda_model.transform(text_tfidf)
    feature_names = vectorizer_X.get_feature_names()
    lda_components = lda_model.components_ / lda_model.components_.sum(axis=1)[:, np.newaxis] # normalization

    for topic in range(n_topics):
        topic_score = text_projection[0][topic]

        for (word_idx, word_score) in zip(lda_components[topic].argsort()[:-5:-1], sorted(lda_components[topic])[:-5:-1]):
            score = topic_score*word_score

            if score >= threshold:
                list_scores.append(score)
                list_words.append(feature_names[word_idx])
                used.add(feature_names[word_idx])

    results = [tag for (y,tag) in sorted(zip(list_scores,list_words), key=lambda pair: pair[0], reverse=True)]
    unique_results = [x for x in results if x not in used] # get only unique tags
    tags = " ".join(results[:5])

    return tags


# In[ ]:


# Program of the recommendation system using lda

# text = input('Ask a question: ')
# tags = Recommend_tags_lda(text, X_train)
# print('Recommended tags are:', tags)


# In[ ]:


# tags_from_taxonomy_list = tags_from_taxonomy["tags"].tolist()
# tags_from_taxonomy_list = [tag.replace("UBS/","").lower() for tag in tags_from_taxonomy_list]
# tags_from_taxonomy_list[500:600]


# In[ ]:


# # from nltk import word_tokenize 
# # from nltk.util import ngrams

# def match_tags_to_taxonomy(tags, tags2):
#     unigram_tags_split = list(filter(None, tags.split("|")))
#     # print("unigram_tags_split",unigram_tags_split)
#     bigram_tags_split = list(filter(None, tags2.split("|")))
#     # print("bigram_tags_split",bigram_tags_split)
#     matched_tags_final = ""
#     for taxonomy_tag in tags_from_taxonomy_list:
        
#         # print(taxonomy_tag)
#         taxonomy_tag_split = taxonomy_tag.split("/")
#         # # taxonomy_tag_split = [tag.split() for tag in taxonomy_tag_split]
#         # # taxonomy_tag_split = [item for sublist in taxonomy_tag_split for item in sublist]
#         # # taxonomy_tag_split = list(set(taxonomy_tag_split))        
#         taxonomy_tokens = nltk.word_tokenize(" ".join(taxonomy_tag_split))
#         bigrams = list(ngrams(taxonomy_tokens, 2)) 
#         bigrams = [" ".join(bigram) for bigram in bigrams]
#         # print(taxonomy_tag_split)
#         # print(tokens)
#         # print(bigrams)
        
#         # for tag in unigram_tags_split:
#             # if tag in taxonomy_tokens:
#                 # print("matched taxonomy tag:",taxonomy_tag,"from word:",tag)
#                 # matched_tags_final = matched_tags_final+"|"+taxonomy_tag
        
#         for tag2 in bigram_tags_split:
#             if tag2 in bigrams:
#                 # print("matched taxonomy tag:",taxonomy_tag,"from bigram:",tag2)
#                 matched_tags_final = matched_tags_final+"|"+taxonomy_tag

#     return matched_tags_final


# In[ ]:


# tags = "investment|may|tech|financial|information|ag|stimulus|us|report|"
# tags2 = "structure investment|standalone measure|four tech|dynamic favor|big four|house report|investment may|recover covid|trump tweet|"
# # matched = match_tags_to_taxonomy(tags, tags2)
# # print(matched)


# In[ ]:


# df['Taxonomy_tags'] = df.apply(lambda x: match_tags_to_taxonomy(x["tags"], x["tags2"]), axis=1)
# df['Taxonomy_tags']


# In[ ]:


# df.head()


# In[ ]:


# df.to_excel("C:/Users/luca.perrozzi/Downloads/docs_matched_to_taxonomy_tags.xlsx", index=False)


# In[ ]:




