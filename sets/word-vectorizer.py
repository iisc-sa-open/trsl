import sys
import gensim
import json

out = open(sys.argv[2],"w")
f = open(sys.argv[1],"r").read().split()
model = sys.argv[3]

w = gensim.models.word2vec.Word2Vec
model = w.load_word2vec_format(model, binary=True)

for word in f:
  try:
    out.write(json.dumps( [word, model[word].tolist()] ) + "\n")
  except KeyError:
    continue