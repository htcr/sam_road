import numpy as np
import os
import argparse
import json
parser = argparse.ArgumentParser()
parser.add_argument('-savedir', type=str)
args = parser.parse_args()

topo = []
precision = []
recall = []
for file_name in os.listdir(f'../{args.savedir}/results/topo'):
    if '.txt' not in file_name:
        continue
    with open(f'../{args.savedir}/results/topo/{file_name}') as f:
        lines = f.readlines()
    p = float(lines[-1].split(' ')[0].split('=')[-1])
    r = float(lines[-1].split(' ')[-1].split('=')[-1])
    topo.append(2*p*r/(p+r))
    precision.append(p)
    recall.append(r)

    print(file_name,topo[-1])
print('TOPO',np.mean(topo),'Precision',np.mean(precision),'Recall',np.mean(recall))

save_path =  f'../{args.savedir}/score/topo.json'
save_dir = os.path.dirname(save_path)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
with open(f'../{args.savedir}/score/topo.json','w') as jf:
    json.dump({'mean topo':[np.mean(topo),np.mean(precision),np.mean(recall)],'prec':precision,'recall':recall,'f1':topo},jf)

