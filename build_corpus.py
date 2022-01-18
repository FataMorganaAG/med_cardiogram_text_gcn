import re
# build corpus


dataset = 'cardo'

f = open('data/' + dataset + '.txt', 'r')
lines = f.readlines()
docs = []
for line in lines:
    temp = line.split("\t")
    doc_file = open(temp[0], 'r')
    doc_content = doc_file.read()
    doc_file.close()
    print(temp[0], doc_content)
    doc_content = doc_content.replace('\n', ' ')
    docs.append(doc_content)


corpus_str = '\n'.join(docs)
f.close()

f = open('data/corpus/' + dataset + '.txt', 'w')
f.write(corpus_str)
f.close()
