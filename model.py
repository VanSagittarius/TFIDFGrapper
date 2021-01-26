
#TF–IDF Grapper

from pathlib import Path

#append each text file name to the list
all_txt_files =[]
for file in Path('...txt').rglob("*.txt"):
    all_txt_files.append(file.parent / file.name)

#counts the length of the list
n_files = len(all_txt_files)
print(n_files)

#sorting file in ascending numerical order
all_txt_files.sort()


#print first to see if it works
all_txt_files[0]

#converting each .txt file to string
all_docs= []
for txt_file in all_txt_files:
    with open(txt_file) as f:
        txt_file_as_string = f.read()
    all_docs.append(txt_file_as_string)


# `all_doc` is now a string containing all text from .txt file

#import the TfidfVectorizer from Scikit-Learn
from sklearn.feature_extraction.text import TfidfVectorizer

#Tokenization and removing punctuation will happened automatically when using TfidVectorizer
#when converting strings to tf-idf scores
#TfidfVectorizer is a class
vectorizer = TfidfVectorizer(max_df=.65, min_df=1, stop_words=[], use_idf=True, norm=None )

#fit_trasform() method is then run on list of strings
transformed_documents = vectorizer.fit_transform(all_docs)

print(vectorizer.stop_words_)

#`fit_transform()` method converts the list of strings to tf-idf **sparse matrix** (matrix with few zeros) then use `toarray()` method to convert sprase matrix to numpy array.

transformed_documents_as_array = transformed_documents.toarray()

#verifing if numpy array represents the same amount of documents
len(transformed_documents_as_array)

import pandas as pd
pd.DataFrame(transformed_documents_as_array)


# Numpy Array in `transformed_documents_as_array` is converted to a format where every td-idf score
# for every term in every document is represented. Sparse matrices, in contrast, exclude zero-value term scores.
# Every term must be represented so that each document has the same number of values, one for each word in corpus.


# make the output folder if it doesn't already exist
Path(".../tf_idf_output").mkdir(parents=True, exist_ok=True)

# construct a list of output file paths using the previous list of text files the relative path for tf_idf_output
output_filenames = [str(txt_file).replace(".txt", ".csv").replace('.../data', "tf_idf_output/") for txt_file in all_txt_files]

# loop each item in transformed_documents_as_array, using enumerate to keep track of the current position
for counter, doc in enumerate(transformed_documents_as_array):
    # construct a dataframe
    tf_idf_tuples = list(zip(vectorizer.get_feature_names(), doc))
    one_doc_as_df = pd.DataFrame.from_records(tf_idf_tuples, columns=['term', 'score']).sort_values(by='score', ascending=False).reset_index(drop=True)

    # output to a csv using the enumerated value for the filename
    one_doc_as_df.to_csv(output_filenames[counter])