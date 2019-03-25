import random
import argparse


if __name__ == '__main__':
	# Hyper Parameters
	parser = argparse.ArgumentParser()
	parser.add_argument('--text_file', type=str)
	parser.add_argument('--size', default=30000, type=int)
	opt = parser.parse_args()
	with open(opt.text_file, 'r', encoding='utf-8') as f:
		data = f.read()
	data = data.split('\n')
	print(len(data))
	print(data[0])
	total_size = int(len(data)/3)

	random_pick = random.sample(list(range(total_size)), opt.size)

	content = []
	aspect = []
	porilty = []
	for idx in random_pick:
		real_index = idx * 3
		tmp_content = data[real_index]
		tmp_aspect = data[real_index + 1]
		tmp_porilty = data[real_index + 2]

		content.append(tmp_content)
		aspect.append(tmp_aspect)
		porilty.append(tmp_porilty)

	with open('sampled_' + opt.text_file, 'w', encoding='utf-8') as f:
		for idx, tmp_content in enumerate(content):
			f.write(tmp_content+'\n')
			f.write(aspect[idx]+'\n')
			f.write(porilty[idx]+'\n')

