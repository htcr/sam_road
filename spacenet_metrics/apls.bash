declare -a arr=( $(jq -r '.test[]' ../spacenet/data_split.json) )

# source directory
dir=$1
data_dir='spacenet'
mkdir -p ../$dir/results/apls

echo $dir
# now loop through the above array
for i in "${arr[@]}"   
do
    # gt_graph=${i}__gt_graph_dense_spacenet.p
    gt_graph=${i}__gt_graph.p
    if test -f "../${dir}/graph/${i}.p"; then
        echo "========================$i======================"
        python ./apls/convert.py "../${data_dir}/RGB_1.0_meter/${gt_graph}" gt.json
        python ./apls/convert.py "../${dir}/graph/${i}.p" prop.json
        
        /usr/local/go/bin/go run ./apls/main.go gt.json prop.json ../$dir/results/apls/$i.txt  spacenet
    fi
done
python apls.py --dir $dir