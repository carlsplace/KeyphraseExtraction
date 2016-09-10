with open('./data/WWW_filelist', 'r') as f:
    kdd = f.read()

kdd = kdd.split(',')
print(len(kdd))

count = 0
for file in kdd:
    with open('./data/WWW/gold/'+str(file), 'r') as f:
        gold = f.read()
    count += len(gold.split('\n')) - 1

print(count)