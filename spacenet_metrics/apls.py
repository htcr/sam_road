import numpy as np
import os
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str)

args = parser.parse_args()

apls = []
output_apls = []
name_list = os.listdir(f'../{args.dir}/results/apls')
name_list.sort()
for file_name in name_list :
    with open(f'../{args.dir}/results/apls/{file_name}') as f:
        lines = f.readlines()
    # print(file_name,lines[0].split(' ')[-1])
    # print(lines[0].split(' '))
    if 'NaN' in lines[0]:
        pass
        # apls.append(0)
        # output_apls.append([file_name,0])
    else:
        apls.append(float(lines[0].split(' ')[-1]))
        output_apls.append([file_name,float(lines[0].split(' ')[-1])])

print('APLS',np.sum(apls)/len(apls))
with open(f'../{args.dir}/results/apls.json','w') as jf:
    json.dump({'apls':output_apls,'final_APLS':np.mean(apls)},jf)