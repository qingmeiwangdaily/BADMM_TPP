# BADMM_TPP
[AAAI2025] The official implementation of "A Plug-and-Play Bregman ADMM Module for Inferring Event Branches in Temporal Point Processes"

## Instructions

Here are the instructions to use the code base.

## Dependencies

This code is written in python. To use it you will need:

* PyTorch == 1.13.1
* Python ==  3.8.1

## Dataset 

The datasets are available on this [Google drive] (https://drive.google.com/drive/folders/0BwqmV0EcoUc8UklIR1BKV25YR1U?resourcekey=0-OrlU87jyc1m-dVMmY5aC4w). To run the model, you should download them to the parent directory of the source code, with the folder name `data`. For SAHP, to make the data format consistent, it is necessary to run the script [convert_realdata_syntheform.py](utils/convert_realdata_syntheform.py) first. 

The five datasets are listed below:

- Conttime
- Retweet
- StackOverflow
- Amazon
- Taobao

Also,we use the famous film named 12 Angry Men to further verify the  practicality of our module.

### Step 1. Quick Start

- Get code

```
git clone 
```

- Build environmet

```bash
conda create -n BADMM python=3.8
conda activate BADMM
```

### Step 2. Prepare datasets 

The `data` directory contains the dataset to use.

### Step 3. Train the model

We integrate our BADMM module into existing TPPs and evaluate its impact accordingly. We consider the classic Hawkes process (**HP** in (Zhou, Zha, and Song 2013b)) and two Transformer-based TPPs (**THP** in (Zuo et al. 2020) and **SAHP** in (Zhang et al. 2020)) .

For **HP**,  if you want to use the BADMM_nuclear module with the `data/retweet` data file, you can run the script like this:

```bash
cd HP
python main.py -data ../data/retweet/ -mode BADMM_nuclear -alpha alpha -lambd lambd
```

If you wish to change the module or dataset, simply alter the corresponding parameter values. For instance, to use the HP_BADMM12 module or the EM module with the `data/retweet` dataset, run:

```bash
cd HP
python main.py -data ../data/retweet/ -mode BADMM_12 -alpha alpha -lambd lambd
python main.py -data ../data/retweet/ -mode EM 
```

Additionally, if you want to use the BADMM module with the film **12 Angry Men** dataset, you can run:

```bash
cd HP
python main_12_angry_men.py -mode BADMM_nuclear
```

For **THP** and **SAHP**,   you can switch between the modules: softmax, sinkhorn, BADMM12, and BADMM_nuclear. To do so, you can run the script like this:

```bash
cd THP
python main.py -data <your data path> -mode badmm -n_it n_it -lambda_ lambda_ -alpha alpha
python main.py -data <your data path> -mode badmm12 -n_it n_it -lambda_ lambda_ -alpha alpha
python main.py -data <your data path> -mode sinkhorn  -n_it n_it
python main.py -data <your data path> -mode softmax
```

```bash
cd SAHP
python main.py -data <your data name> -mode badmm -n_it n_it -lambda_ lambda_ -alpha alpha  
python main.py -data <your data name> -mode badmm12 -n_it n_it
python main.py -data <your data name> -mode sinkhorn  -n_it n_it
python main.py -data <your data name> -mode softmax
```

## Parameters

```n_it``` Specifies the number of iterations for the Bregman ADMM or Sinkhorn-scaling algorithm. The default value is set to 2.

```alpha``` Represents the weight of the regularization term. The value should be within the range (0,1).

```lambda```  Indicates the weight of the regularization term. Acceptable values are {0.01,0.1,1,10,100}.

###  



