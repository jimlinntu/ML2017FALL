def main():
	parser = argparse.ArgumentParser(description='q1.py')
	parser.add_argument('DATA_FILE_PATH', type=str,
                    help='batch size for training and testing (default: 100)')
	args = parser.parse_args()
	count_and_print(args.DATA_FILE_PATH)
def count(DATA_FILE_PATH):
	index_to_word = {}
	word_to_index = {}
	countlist = []
	with open(DATA_FILE_PATH) as f:
		for line in f:
			word_list = line.split()
			for i, word in enumerate(word_list):
				if word not in index_to_word.values():
					# Add new index
					word_to_index[word] = len(countlist)
					index_to_word[len(countlist)] = word
					countlist.append(0)
				countlist[word_to_index[word]] += 1
	with open("./Q1.txt", "w") as f:
		for i, count in enumerate(countlist):
			if i == 0:
				print("{} {} {}".format(index_to_word[i], i, count), end='', file=f)
			else:
				print("\n{} {} {}".format(index_to_word[i], i, count), end='', file=f)

				

if __name__ == '__main__':
	main()