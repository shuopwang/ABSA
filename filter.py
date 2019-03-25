import os
import pandas as pd
from tqdm import tqdm
import random
import re

with open('keyword.txt','r', encoding='utf-8') as f:
	keywords = f.readlines()
with open('nishuihan_keyword.txt', 'r', encoding='utf-8') as f:
	nishuihan_keyword = f.readlines()

paren_path = os.path.abspath(os.path.dirname(os.getcwd()))
corpus_path = os.path.join(paren_path, 'chinese_corpus/taptap_comments.csv')
#corpus_path = os.path.join(paren_path, 'sentiment_analyse/downsample_labeled_data.txt')

content = []
aspect = []
polarity = []

keywords = keywords + nishuihan_keyword
keywords = list(set(keywords))
keywords = [each.replace('\n','') for each in keywords]
keywords = sorted(keywords, key=lambda x:len(x), reverse=True)

# taptap_comments = pd.read_csv('taptap_comments.csv')
# total_content = taptap_comments['contents'].values
# total_score = taptap_comments['score'].values

taptap_comments = pd.read_csv('old_koubei.csv')
total_content = taptap_comments['content'].values
total_score = taptap_comments['trend'].values
# with open('tieba100w.txt', 'r', encoding='utf-8') as f:
# 	tieba_data = f.readlines()

def replace(line):
	line = line.lower()
	line = re.sub('id:[A-Za-z0-9\u4e00-\u9fa5]+|id：[A-Za-z0-9\u4e00-\u9fa5]+', '', line)
	line = re.sub('日期:[A-Za-z0-9\u4e00-\u9fa5]+|日期：[A-Za-z0-9\u4e00-\u9fa5]+', '', line)
	line = re.sub('心情:|心情：', '', line)
	line = re.sub('回复 .*：|回复 .*:', '', line)
	line = re.sub('秒拍视频', '', line)
	line = re.sub('snap_', '', line)
	line = re.sub('微信图片_', '', line)
	line = re.sub('YY图片', '', line)
	line = re.sub('QQ图片', '', line)
	line = re.sub('#[a-zA-Z_0-9]+|(#[^>]*#)|\([^>]*\)|\*[^>]*\*|\[[^>]*\]|<[^>]*>|([0-9]+(小时前))|([a-zA-Z0-9]+\.(jpg|gif|bmp|bnp|png))', '', line)
	line = re.sub("(@[A-Za-z0-9\u4e00-\u9fa5]+)|([^0-9A-Za-z\u4e00-\u9fa5 \t])|(\w+:\/\/\S+)", " ", line)
	line = line.strip()
	return line

count = 0
for index, tmp_content in enumerate(tqdm(tieba_data)):
	used_key = []
	#tmp_score = (total_score[index])
	tmp_content = str(tmp_content)
	tmp_content = tmp_content.replace('\n', ' ')
	tmp_content = replace(tmp_content)
	sentiment = 0
	# tmp_score = int(tmp_score)
	# if tmp_score > 3:
	# 	sentiment = 1
	# if tmp_score == 3:
	# 	sentiment = 0
	# if tmp_score < 3:
	# 	sentiment = 2
	if tmp_score == '正面':
		sentiment = 1
	if tmp_score == '中立':
		sentiment = 0
	if tmp_score == '负面':
		sentiment = 2
	for key in keywords:
		if key in used_key:
			continue
		if key in tmp_content:
			after_replace = tmp_content.replace(key, '$T$')
			after_replace = after_replace.replace('\n', ' ')
			key = key.replace('\n', ' ')
			content.append(after_replace)
			aspect.append(key)
			polarity.append(sentiment)
			used_key.append(key)
			n=1
			used_key += ([key[i:i+n] for i in range(len(key)-n+1)])
			n=2
			used_key += ([key[i:i+n] for i in range(len(key)-n+1)])
			count += 1
# with open(corpus_path, 'r', encoding='utf-8') as f:
# 	for line in tqdm(f):
# 		used_key = []
# 		try:
# 			each, sentiment = line.split('\t')
# 			for key in keywords:
# 				if key in used_key:
# 					continue
# 				try:
# 					if key in str(each):
# 						after_replace = each.replace(key, '$T$')
# 						content.append(after_replace)
# 						aspect.append(key)
# 						polarity.append(sentiment)
# 						used_key.append(key)
# 						n=1
# 						used_key += ([key[i:i+n] for i in range(len(key)-n+1)])
# 						n=2
# 						used_key += ([key[i:i+n] for i in range(len(key)-n+1)])
# 						count += 1
# 				except Exception as e:
# 					continue
# 				else:
# 					pass
# 				finally:
# 					pass
# 		except Exception as e:
# 			continue
	#print('for the data set has keyword number:{}\n'.format(count))

#print(keywords)

# if os.path.isdir(corpus_path):
# 	for filename in os.listdir(corpus_path):
# 		if filename.endswith('.csv'):
# 			real_filename = os.path.join(corpus_path, filename)
# 			if 'weibo' in real_filename:
# 				continue
# 			print('start process the file: {}'.format(filename))
# 			tmp_df = pd.read_csv(real_filename, error_bad_lines=False)
# 			tmp_content = tmp_df['content'].values
# 			count = 0
# 			for each in tqdm(tmp_content):
# 				used_key = []
# 				for key in keywords:
# 					if key in used_key:
# 						continue
# 					try:
# 						if key in str(each):
# 							each = each.replace(key, '$T$')
# 							content.append(each)
# 							aspect.append(key)
# 							used_key.append(key)
# 							n=1
# 							used_key += ([key[i:i+n] for i in range(len(key)-n+1)])
# 							n=2
# 							used_key += ([key[i:i+n] for i in range(len(key)-n+1)])
# 							count += 1
# 							each = each.replace('$T$', key)
# 					except:
# 						continue
# 			print('for the data set {} has keyword number:{}\n'.format(filename, count))
print('get aspect data:{}/{}'.format(len(content), len(tieba_data)))
sample_size = 20000
total_size = len(content)
random_pick = random.sample(list(range(total_size)), sample_size)
with open('sample_tieba.txt','w',encoding='utf-8') as f:
	for idx in random_pick:
		f.write(content[idx]+'\n')
		f.write(aspect[idx]+'\n')
		f.write(str(polarity[idx])+'\n')