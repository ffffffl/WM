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
