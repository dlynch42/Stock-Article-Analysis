import os
import pickle
import re
import warnings
import nltk
import numpy as np
import pandas as pd
import pyprind
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer, HashingVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

# 1) Preparing IMDb movie review data for test processing
# Preprocessing Movie Dataset into more convenient format
df = pd.read_csv('movie_data.csv')

# 2) Introducing the Bag-of-Words Model
# Transforming Words into Feature Vectors, CountVectorizer takes array of text and constructs bag-of-words model
# Transform following three sentences into sparse feature vectors
count = CountVectorizer()
docs = np.array([
    'The sun is shining',
    'The weather is sweet',
    'The sun is shining, the weather is sweet',
    'and one and one is two'])
bag = count.fit_transform(docs)
# Print contents of vocabulary
print(count.vocabulary_)
# Print Feature Vectors
print(bag.toarray())

# Assessing word relevancy via term frequency-inverse document frequency (tf-idf)
# Scikit-Learn, transform raw term frequencies from CountVectorizer class as input and transforms them into tf-idfs
tfidf = TfidfTransformer(use_idf=True,
                         norm='l2',
                         smooth_idf=True)
np.set_printoptions(precision=2)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())

# Cleaning Text Data
# Illustrate importance of cleaning data
print(df.loc[0, 'review'][-50:])
# Remove all punctuation marks except emoticon characters (useful for sentiment analysis)
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)      # remove HTML markup from reviews
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text)            # regex to find emoticons
    text = (re.sub('[\W]+', ' ', text.lower() ) +
            ' '.join(emoticons).replace('-', ''))      # remove all non-word characters and convert text into lowercase letters
    return text
# Confirm preprocessor works correctly
print(preprocessor(df.loc[0, 'review'][-50:]))
print(preprocessor("</a>This :) is :( a test :-)!"))        # should remove punctuation and '-'
# Apply preprocessor fxn to all reviews in df
df['review'] = df['review'].apply(preprocessor)

# Processing docs into tokens
# Tokenize docs by splitting into individual words, split cleaned docs at whitespace characters
def tokenizer(text):
    return text.split()
print(tokenizer('runners like running and thus they run'))

# Porter Stemming, reduce words to root form
porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]
print(tokenizer_porter('runners like running and thus they run'))

# Stop-Word removal
# Use set of 127 stop-words
nltk.download('stopwords')

# Apply English Stop-word set
stop = stopwords.words('english')
print([w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:]
       if w not in stop])



# 3) Training LR model for document classification, classify reviews into positive and negative reivews
# Divide docs into 25000 training and 25000 testing
X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

# Find optimal set of parameters for LR model using 5-fold stratified cross-validation
tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)
param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'vect__use_idf':[False],
               'vect__norm':[None],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]}]
lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf',
                      LogisticRegression(random_state=0))])
# GridSearch is on MLBookScratch.txt



# 4) Working with Big Data - Online Algorithms and Out of Core Learning
nltk.download('stopwords')

# Define tokenizer fxn that cleans unprocessed text data
stop = stopwords.words('english')
def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) \
           + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

# Define generator fxn stream_docs, reads in and returns one doc at a time
def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv)   # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label

# Verify stream_docs works correctly
print(next(stream_docs(path='movie_data.csv')))

# Define fxn get_minibatch, take doc stream from doc_stream and return particular number of docs specified by size parameter
def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y

# Initialize HashingVectorizer (like CountVectorizer but can use in OoC Learning), reinitialize LR Classifier
vect = HashingVectorizer(decode_error='ignore',
                         n_features=2**21,
                         preprocessor=None,
                         tokenizer=tokenizer)
clf = SGDClassifier(loss='log', random_state=1, max_iter=1)
doc_stream = stream_docs(path='movie_data.csv')

# Start OoC learning
pbar = pyprind.ProgBar(45)
classes = np.array([0, 1])
for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream, size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
    pbar.update()

# Use last 5000 docs to evaluate performance of model
X_test, y_test = get_minibatch(doc_stream, size=5000)
X_test = vect.transform(X_test)
print('Accuracy: %.3f' % clf.score(X_test, y_test))

# Use last 5000 docs to update model
clf = clf.partial_fit(X_test, y_test)



# 5) Topic Modeling with Latent Dirichlet Allocation (LDA, not same as chapter 5)
# LDA with Scikit-Learn
# Load dataset into a pandas DF
df = pd.read_csv('movie_data.csv', encoding='utf-8')

# Use CountVectorizer to create bag-of-words matrix as input to LDA, use English Stop word library
count = CountVectorizer(stop_words='english',
                        max_df=.1,
                        max_features=5000)
X = count.fit_transform(df['review'].values)

# Fit LDA Estimator to Bag-Of-Words Matrix and infer 10 different topics from the docs
lda = LatentDirichletAllocation(n_components=10,
                                random_state=123,
                                learning_method='batch')
X_topics = lda.fit_transform(X)
print(lda.components_.shape)

# Print top 5 words, need to sort topic array in reverse order
n_top_words = 5
feature_names = count.get_feature_names()
for topic_idx, topic in enumerate(lda.components_):
    print("Topic %d:" % (topic_idx + 1))
    print(" ".join([feature_names[i]
                    for i in topic.argsort()\
                    [:-n_top_words - 1:-1]]))
# Confirm categories make sense based off reviews, plot three movies from horror movies category (category 6 at index 5)
horror = X_topics[:, 5].argsort()[::-1]     # remove 'horror' from this line and 'movie' from line above since this is about stocks
for iter_idx, stock_idx in enumerate(horror[:3]):
    print('\nHorror Movie #%d:' % (iter_idx + 1))
    print(df['review'][stock_idx][:300], '...')



# Create pkl object
# Serializing Fitted Scikit-Learn Estimators
# Serialize and deserialize python object structures to compact bytecode, create pkl_objects subdirectory to
# save serialized Python objects to local drive
dest = os.path.join('stockclassifier', 'pkl_objects')
if not os.path.exists(dest):
    os.makedirs(dest)

pickle.dump(stop,
            open(os.path.join(dest, 'stopwords.pkl'), 'wb'),
            protocol=4)
pickle.dump(clf,
            open(os.path.join(dest, 'classifier.pkl'), 'wb'),
            protocol=4)
