\#number of topics we will cluster for: 10
num_topics = 10;



# NMF topic clustering technique

#For NMF, we need to obtain a design matrix. To improve results, I am going to apply TfIdf transformation to the counts.



#We only need the Headlines_text column from the data
data_text = data[['Body']];
data_text = data_text.astype('str');



for idx in range(len(data_text)):
    
    #go through each word in each data_text row, remove stopwords, and set them on the index.
    data_text.iloc[idx]['Body'] = [word for word in data_text.iloc[idx]['Body'].split(' ')]; # if word not in stopwords.words()];
    
    #print logs to monitor output
    if idx % 1000 == 0:
        sys.stdout.write('\rc = ' + str(idx) + ' / ' + str(len(data_text)));
        
        
#get the words as an array for lda input
train_headlines = [value[0] for value in data_text.iloc[0:].values];


#the count vectorizer needs string inputs, not array, so I join them with a space.
train_headlines_sentences = [' '.join(text) for text in train_headlines]


vectorizer = CountVectorizer(analyzer='word', 
                            # max_features=5000,
                            max_df = 0.5, min_df = 5, 
                             max_features = None,
                            ngram_range=(1,1),
                            );
x_counts = vectorizer.fit_transform(train_headlines_sentences);


#Next, we set a TfIdf Transformer, and transform the counts with the model.
transformer = TfidfTransformer(smooth_idf=False);
x_tfidf = transformer.fit_transform(x_counts);

# And now we normalize the TfIdf values to unit length for each row.
xtfidf_norm = normalize(x_tfidf, norm='l1', axis=1)


#And finally, obtain a NMF model, and fit it with the sentences.

#obtain a NMF model.
model = NMF(n_components=num_topics, init='nndsvd');


#fit the model
model.fit(xtfidf_norm)

def get_nmf_topics(model, n_top_words):
    
    #the word ids obtained need to be reverse-mapped to the words so we can print the topic names.
    feat_names = vectorizer.get_feature_names()
    
    word_dict = {};
    for i in range(num_topics):
        
        #for each topic, obtain the largest values, and add the words they map to into the dictionary.
        words_ids = model.components_[i].argsort()[:-20 - 1:-1]
        words = [feat_names[key] for key in words_ids][:n_top_words]
        word_dict['Topic # ' + '{:02d}'.format(i+1)] = words;
    
    df = pd.DataFrame(word_dict)
    df.index.rename("Top Words \ Topics", inplace=True)
    return df



get_nmf_topics(model, 10)
