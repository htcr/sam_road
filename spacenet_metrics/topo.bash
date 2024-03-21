# source directory
dir=$1

python ./topo/main.py -savedir $dir
python topo.py -savedir $dir