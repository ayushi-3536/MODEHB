import json
import os
import glob

def load_json_file(path,spath,idx):
    # Opening JSON file
    f = open(path)
    fs = open(spath+'\\'+'_'+str(idx)+'.txt','w')
    data = (f.read().split("\n"))
    print(len(data))
    for i in data:
         #print(i)
         line = json.loads(i)
         print(line)
         fs.write(str(line['num_params'])+' '+str(line['acc'])+'\n')

    # Closing file
    f.close()
    fs.close()


path = 'C:\\Users\\ayush\\PycharmProjects\\MODEHB\\MODEHB\\modehb_10_runs\\dehb_run'
extension = 'json'
os.chdir(path)
result = glob.glob('*.{}'.format(extension))
print(result)
for idx,f in enumerate(result):
    if(idx<9):
        continue
    load_json_file(path + '\\' + f,path,idx)