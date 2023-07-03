"""
--------------------------------------------------------------------------- 
Script to (1) pre-process the patent text, (2) train a Doc2Vec model for patents and (3) get the vectors from the trained model.
Suppose having all the data stored (input and output) in the path './data'.

"Measuring the Position and Differentiation of Firms in Technology Space" -- Sam Arts, Bruno Cassiman, and Jianan Hou
---------------------------------------------------------------------------
"""

import pandas as pd
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.stem import PorterStemmer

########## Preprocess the patent text ##########

# Define the path for the raw patent text and keywords
## 'patent txt raw' contains patents sorted by patent id with four columns with the patent id, title, abstract and claims
## 'keywords' contains patents sorted by patent id with two columns with the patent id and the number of unique stemmed keywords
path_patents = './data/patent txt raw.csv'
path_keywords = './data/keywords.csv'

# Define the text pre-processed data path
processed_data_path = './data/patents_processed.csv'

# Create or overwrite the existing file with only headers
with open(processed_data_path,'w') as writer:
    writer.write('patent,text\n')

# Create a Porter Stemmer instance
stemmer = PorterStemmer()

# Load the raw patent data and the keywords data in chunks to handle large datasets
ds_raw = pd.read_csv(path_patents, on_bad_lines = 'skip', chunksize = 1000, converters = {'patent':int})
ds_keywords = pd.read_csv(path_keywords, on_bad_lines = 'skip', chunksize = 1000, converters = {'patent':int})

# Process each chunk of data. Given that the patents are ordered by patent id, same chunks contains the same patents
# Process by chunk in order to do not use large memory
for df_raw, df_keywords in zip(ds_raw, ds_keywords):
    
    # Merge the raw and keywords dataframes on the 'patent' column
    df = pd.merge(df_raw, df_keywords, on=['patent']).fillna('')

    # Combine 'title', 'abstract' and 'claims' to form the 'text' column
    df['text'] = df['title'] + df['abstract'] + df['claims']

    # Apply stemming to the 'text' column
    df['text'] = df['text'].apply(lambda x: ' '.join([stemmer.stem(t) for t in x.split(' ')]))
    
    # Keep only the words in 'text' that are also in 'keywords'
    df['text'] = df.apply(lambda x: ' '.join([t for t in x['text'].split(' ') if t in x['keywords'].split(' ')]), axis=1)

    # Write the processed chunk to the csv file
    df[['patent', 'text']].to_csv(processed_data_path, header=False, index=False, mode='a')


########## Train the Doc2vec ##########

# Read the data from the csv file with all patent texts
df = pd.read_csv('./data/all_patent_text.csv.csv')

# Create a list of TaggedDocument objects.
documents = [TaggedDocument(doc, [i]) for i, doc in zip(df['patent'], df['text'])]

# Initialize a Doc2Vec model with a vector size of 700 and window size of 5.
model = Doc2Vec(vector_size=700, window=5)

# Build a vocabulary from the documents
model.build_vocab(documents)

# Train the model on the documents for 10 epochs. "total_examples" is optional
model.train(documents, total_examples=model.corpus_count, epochs=10)

# Save the trained model for later use
model.save("./data/doc2vec.model")


########## Get the patent vectors ##########


# Create a new DataFrame to store the patent vectors
df_vectors = pd.DataFrame()

# Iterate over all patents
for patentId in df['patent']:
    # Retrieve the vector for the patent from the trained model
    vector = model.dv[patentId]

    # Add the vector to the new DataFrame
    df_vectors = df_vectors.append(pd.Series(vector, name=patentId))

# Transpose the DataFrame so that each patentId is a row and each feature is a column
df_vectors = df_vectors.transpose()

# Save the DataFrame to a CSV file
df_vectors.to_csv('patents_vectors.csv')
