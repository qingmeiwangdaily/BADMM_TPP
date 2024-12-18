n_head=4
n_layers=4
d_model=512
d_rnn=64
d_inner=1024
d_k=512
d_v=512
dropout=0.1
lr=1e-4
smooth=0.1
rho=1

device=2
batch=32
epoch=70
seed=766
n_it=2
alpha=0.1
lambda_=0.1
mode='badmm12'

checkpoints_path=./checkpoints
data=../data/retweet/
log=./log/retweet_${n_it}_${alpha}_${lambda_}_${seed}.txt
name=${n_it}_${alpha}_${lambda_}

echo "nohup bash runretweet.sh >> outs/retweet_${n_it}_l${lambda_}_a${alpha}_${seed}.out 2>&1 &"

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$device python main.py -data $data -batch $batch -n_head $n_head -n_layers $n_layers -d_model $d_model -d_rnn $d_rnn -d_inner $d_inner -d_k $d_k -d_v $d_v -dropout $dropout -lr $lr -smooth $smooth -epoch $epoch -n_it $n_it -rho $rho -log $log -seed $seed -wandb $wandb -name $name -alpha $alpha -lambda_ $lambda_ -checkpoints_path $checkpoints_path -mode $mode

