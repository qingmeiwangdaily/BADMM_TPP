data="../data/retweet/"
epoch=1
log='log.txt'
seed=999
model="BADMM_nuclear"
predict_time_nums=2
result='result.txt'
rho=1
alpha=0.1
lambd=0.1
checkpoint='checkpoint.pkl'
num_iteration=10
pre_model=False

python main.py -data $data -epoch $epoch -seed $seed -log $log -predict_time_nums $predict_time_nums -result $result -rho $rho -alpha $alpha -lambd $lambd -checkpoint $checkpoint -num_iteration $num_iteration -pre_model $pre_model -model $model

#nohup ./run.sh >> 1.out 2>&1 &
