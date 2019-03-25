import fasttext
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.metrics import silhouette_score
import numpy as np
from tqdm import tqdm

global_word_embedding_path = '/Users/codeur/Desktop/fuxi_nlp/public_sentiment_analyse/nlp/nishuihan_word_embedding.bin'
fasttext_word_embedding_model = fasttext.load_model(global_word_embedding_path)

with open('/Users/codeur/Desktop/fuxi_nlp/ABSA-PyTorch/keyword.txt', 'r', encoding='utf-8') as f:
	keywords = f.readlines()
keywords = [each.replace('\n', '') for each in keywords]

keywords = list(set(keywords))

keyword_embedding = np.zeros((len(keywords), fasttext_word_embedding_model.dim))

game_keyword_embedding = []
for keyword in keywords:
	game_keyword_embedding.append(scale(fasttext_word_embedding_model[keyword]))

clusters = 50

best_sc_score = -float('inf')
for cluster in tqdm(range(5, clusters)):
	kmenas_model = KMeans(n_clusters=cluster).fit(game_keyword_embedding)
	sc_score = silhouette_score(game_keyword_embedding, kmenas_model.labels_, metric='euclidean')
	if sc_score > best_sc_score:
		best_model = kmenas_model
		best_sc_score = sc_score

keyword_label = best_model.labels_

label_keyword = {}
for idx, label in enumerate(keyword_label):
	if label not in label_keyword:
		label_keyword[label] = [keywords[idx]]
	else:
		label_keyword[label].append(keywords[idx])

label_keyword = sorted(label_keyword.items(), key=lambda x: len(x[1]), reverse=True)
with open('best_cluster_result.txt', 'w', encoding='utf-8') as f:
	for label, kws in label_keyword:
		f.write('cluster size:'+str(len(kws))+'\t'+' '.join(kws)+'\n')

