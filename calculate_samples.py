


with open('cal_log.txt') as f:
	data = f.readlines()

# calculate sentences
#max_num = 0
#min_num = 100
#sents = 0
#count = 0
#for line in data:
#	l = str.split(line, ',')
#	d = [str.split(item, ':') for item in l]
#	print(d)
#	sent_num = int(d[1][1])
#	max_num = max(max_num, sent_num)
#	min_num = min(min_num, sent_num)
#	sents += sent_num
#	count += 1

#sent_num_avg = sents / count
#print('------')
#print('max_sent_num: %d' % max_num)
#print('min_sent_num: %d' % min_num)
#print('sents_num: %d' % sents)
#print('counts: %d' % count)
#print('avg_sent_num: %d' % sent_num_avg)
#print('------')

# calculate words
max_num = 0
min_num = 100
words = 0
count = 0
for line in data:
	word_list = str.split(str.split(line, ':')[-1], ',')
	word_list = [int(item) for item in word_list[:-1]]
	print(word_list)
	for item in word_list:
		max_num = max(max_num, item)
		min_num = min(min_num, item)
		words += item
		count += 1

avg_num = words/count
print('------')
print('max: %d' % max_num)
print('min: %d' % min_num)
print('words: %d' % words)
print('count: %d' % count)
print('avg: %d' % avg_num)
print('------')