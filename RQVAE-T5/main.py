from train import train
from evaluate import infer
task_ID = 'task1'
params = {
        'task_id': task_ID,
        'code_path' : '../data/course/course_rqvae_codes.npy',
        'train_dataset_path' : '../data/tiger/train_dataset.h5',
        'test_dataset_path' : '../data/tiger/test_dataset.h5',
        'batch_size':256,
        'infer_size':256,
        'num_epochs':500,
        'lr':1e-3,
        'device':'cuda:2',
        'num_layers':2, # encoder 层数
        'num_decoder_layers':2, # decoder 层数
        'd_model':64, # encoder 隐藏层状态
        'd_ff':256,
        'num_heads':4,
        'd_kv':16, # d_kv = d_model / num_heads = 64/4 = 16
        'dropout_rate':0.1,
        'vocab_size':64,
        'codebook_size':8,
        'pad_token_id':0,
        'eos_token_id':31,
        'feed_forward_proj':'relu',
        'max_len':20,
        'log_path':'./logs/tiger.log',
        'seed':42,
        'save_path':f'./ckpt/tiger_{task_ID}.pth',
        'params_path':'./RQVAE-T5-results.csv',
        'early_stop':10,
        'topk_list':[2,5,10,20],
        'loss_plot_path':f'./loss_picture/{task_ID}.png',
        'beam_size':5,
    }

def main():
    train(params)
    infer(params)
if __name__ == "__main__":
    main()
