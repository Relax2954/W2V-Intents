import numpy as np

# Category -> words
data = {
  'Greeting': ['hello','whats up','yo','sup', 'hi', 'how is it going', 'good evening', 'good morning'],
  'Colors': ['yellow', 'red','green'],
  'Places': ['tokyo','bejing','washington','mumbai'],
}
# Words -> category
categories = {word: key for key, words in data.items() for word in words}

print(categories)

# Load the whole embedding matrix
embeddings_index = {}
with open('cc.en.300.vec.txt') as f:
  for line in f:
    values = line.split()
   # print(values)
    word = values[0]
    #print(word)
    embed = np.array(values[1:], dtype=np.float32)
    #print(embed)
    embeddings_index[word] = embed
print('Loaded %s word vectors.' % len(embeddings_index))
# Embeddings for available words
data_embeddings = {key: value for key, value in embeddings_index.items() if key in categories.keys()}

# Processing the query
def process(query):
  query_embed = embeddings_index[query]
  scores = {}
  for word, embed in data_embeddings.items():
    category = categories[word]
    dist = query_embed.dot(embed)
    dist /= len(data[category])
    scores[category] = scores.get(category, 0) + dist
  return scores

# Testing
print(process('hey'))
print(process('sasha'))
print(process('moscow'))