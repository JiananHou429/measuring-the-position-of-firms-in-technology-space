"""
--------------------------------------------------------------------------- 
Script to train a Doc2Vec model for patents and get the vectors from the trained model.
"Measuring the Position and Differentiation of Firms in
 Technology Space" -- Sam Arts, Bruno Cassiman, and Jianan Hou
---------------------------------------------------------------------------
"""

# Import necessary libraries
import pandas as pd
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
from sklearn.metrics.pairwise import cosine_similarity

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
model.save("patent_doc2vec.model")

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
