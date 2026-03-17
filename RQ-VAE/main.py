

from time import time
from train import  train
from infer import infer
params = {
    "data_path":"../data/course_item_embs.h5",
    "ckpt_dir":"./ckpt/course",
    "semantic_id_file":"../data/course/course_rqvae_codes.npy",
    "in_dim":768,
    "num_emb_list":[8,8,8],
    "e_dim":32,                        
    "layers":[256,128],            
    "dropout":0.1,
    "batch_normalize":False,
    "loss_type":"mse",
    "quant_loss_weight":0.1,
    "beta":0.25,
    "kmeans_init":True,
    "kmeans_iters":50,
    "lr":1e-3,
    "epochs":100,
    "warmup_epochs":5,
    "batch_size":64,
    "num_workers":4,
    "eval_step":50,
    "sk_epsilons":[0.01, 0.01, 0.01],
    "sk_iters":50,
    "learner":"Adamw",
    "lr_scheduler_type": "linear",
    "weight_decay":1e-4,
    "save_limit":5,
    "eval_step":50,
    "device":"cuda:0",
    
} 

if __name__ == '__main__':
    train(params)
    infer(params)