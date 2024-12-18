dropout=0.1
lr=5e-5
hidden=16
dmodel=16
attenheads=8
nLayers=4
trainratio=0.8
lambdal2=3e-4
devratio=0.1
samples=10
rho=1
task=retweet

cuda=0
epochs=60
batch=16
n_it=2
lambda_=1
alpha=0.1
seed=999
use_wandb=0
save_model=1
use_model=0
data=retweet
mode='badmm'

wandb=SAHP-${task}-${seed}
name=badmm_n${n_it}_l${lambda_}_a${alpha}
logdir=./logs/${task}_${n_it}_l${lambda_}_a${alpha}_${seed}.log
checkpoints_path=./checkpoints/${task}
checkpoint_model_path=./checkpoints/${task}/${task}_${n_it}_l${lambda_}_a${alpha}_${seed}_best.pth
echo "nohup bash runretweet.sh >> outs/${task}_${n_it}_l${lambda_}_a${alpha}_${seed}.out 2>&1 &"

mkdir $checkpoints_path
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$cuda python main.py -data $data --batch $batch --atten-heads $attenheads --nLayers $nLayers --d-model $dmodel --dropout $dropout --lr $lr --epochs $epochs --hidden $hidden --train-ratio $trainratio --lambda-l2 $lambdal2 --dev-ratio $devratio --samples $samples --task $task -seed $seed -n_it $n_it  -data $data --log-dir $logdir -wandb $wandb -name $name -rho $rho -use_wandb $use_wandb -checkpoints_path $checkpoints_path -checkpoint_model_path $checkpoint_model_path -use_model $use_model -lambda_ $lambda_ -alpha $alpha -save_model $save_model -mode $mode

