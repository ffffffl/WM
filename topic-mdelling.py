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
