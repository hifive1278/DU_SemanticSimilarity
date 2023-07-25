# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('all-MiniLM-L6-v2')

# #Our sentences we like to encode
# sentences = ['This framework generates embeddings for each input sentence',
#     'Sentences are passed as a list of string.',
#     'The quick brown fox jumps over the lazy dog.']

# #Sentences are encoded by calling model.encode()
# embeddings = model.encode(sentences)

# #Print the embeddings
# for sentence, embedding in zip(sentences, embeddings):
#     print("Sentence:", sentence)
#     print("Embedding:", embedding)
#     print("")

# ----------------------------------------------------------------

from sentence_transformers import SentenceTransformer, util
from torch import max, mean
model = SentenceTransformer('all-MiniLM-L6-v2')

# Two lists of sentences
# sentences1 = ['Hello Python!',
# 'When to use Python?',
# 'The Python Interface',
# 'Any comments?',
# 'Python as a calculator',
# 'Variables and Types',
# 'Variable Assignment',
# 'Calculations with variables',
# 'Other variable types',
# 'Guess the type',
# 'Operations with other types',
# 'Type conversion',
# 'Can Python handle everything?']    #0.0922 mean

sentences1 = ['Introduction to feature engineering and data preparation',   # 0.247 mean
              'Dealing with outliers',
              'Evaluation of missing data',
              'Filling or dropping data based on rows',
              'Fixing data based on columns',
              'Encoding options']   # Feature Engineering and Data Preparation udemy, mvp section 2

sentences2 = ['Addressing missing data',
'Dealing with missing data',
'Strategies for remaining missing data',
'Imputing missing plane prices',
'Converting and analyzing categorical data',
'Finding the number of unique values',
'Flight duration categories',
'Adding duration categories',
'Working with numeric data',
'Flight duration',
'Adding descriptive statistics',
'Handling outliers',
'What to do with outliers',
'Identifying outliers',
'Removing outliers']  # Exploratory Data Analysis in Python datacamp, mvp section 1

#Compute embedding for both lists
embeddings1 = model.encode(sentences1, convert_to_tensor=True)
embeddings2 = model.encode(sentences2, convert_to_tensor=True)

#Compute cosine-similarities
cosine_scores = util.cos_sim(embeddings1, embeddings2)

#Output the pairs with their score
for j in range (len(sentences1)):
    for i in range(len(sentences2)):
        print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[j], sentences2[i], cosine_scores[j][i]))

mean_val = mean(cosine_scores)

# max_value = max(cosine_scores[0, :].clone())  #copies only the first row (first word in sentences1) of cosine_scores

print(mean_val.item())  #.item() converts tensor to a number