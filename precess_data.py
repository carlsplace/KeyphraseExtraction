from os import listdir
from os.path import isfile, join

path = './data/KDD/omega_phi/'
files = [f for f in listdir(path) if isfile(join(path, f))]


to_file = ''
for file in files:
    with open(path+file, 'r') as f:
        to_file = to_file + f.readline()

with open(path+'result.csv', 'w') as f:
    f.write(to_file)

# with open('./data/KDD_filelist', 'r') as f:
#     file = f.read()
# filelist = file.split(',')
# print(filelist[33])