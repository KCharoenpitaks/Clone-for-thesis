#url=$1 # pymarl repo url
#dest=$2 # destination dir

#bash config/common/ubuntu/pymarl.sh $url $dest
#bash config/common/conda.sh $dest


python3 src/main.py --config=coma_smac with running_mode=4 --env-config=sc2
python3 src/main.py --config=coma_smac with running_mode=3 --env-config=sc2
python3 src/main.py --config=coma_smac with running_mode=2 --env-config=sc2
python3 src/main.py --config=coma_smac with running_mode=1 --env-config=sc2
python3 src/main.py --config=coma_smac with running_mode=0 --env-config=sc2
python3 src/main.py --config=coma_smac with running_mode=normal --env-config=sc2


