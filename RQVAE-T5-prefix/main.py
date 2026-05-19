from train import train
from evaluate import infer
task_ID = 'task32'
params = {
        'task_id': task_ID,
        'code_path' : '../data/course/course_rqvae_codes.npy',
        'train_dataset_path' : '../data/tiger/train_dataset.h5',
        'test_dataset_path' : '../data/tiger/test_dataset.h5',
        'batch_size':256,
        'infer_size':256,
        'num_epochs':300,
        'lr':1e-3,
        'device':'cuda:1',
        'num_layers':2, # encoder 层数 (t5-small)
        'num_decoder_layers':4, # decoder 层数 (t5-small)
        'd_model':128, # encoder 隐藏层状态 (t5-small)
        'd_ff':512,
        'num_heads':8,
        'd_kv':16,
        'dropout_rate':0.1,
        'vocab_size':64,
        'codebook_size':8,
        'pad_token_id':0,
        'eos_token_id':31,
        'feed_forward_proj':'relu',
        'max_len':20,
        'pretrained_path':'../pretrained/t5-small',
        'log_path':'./logs/tiger.log',
        'seed':42,
        'save_path':f'./ckpt/tiger_{task_ID}.pth',
        'early_stop':10,
        'topk_list':[2,5,10,20],
        'loss_plot_path':f'./loss_picture/{task_ID}.png',
        'params_path':'./RQVAE-T5-prefix-result.csv',
        'beam_size':5,
        'bert_dim':768,
        # 三级专业能力嵌入路径
        'prof_h5_paths':{
            'prof_lvl1': '../data/professional/prof_lvl1.h5',
            'prof_lvl2': '../data/professional/prof_lvl2.h5',
            'prof_lvl3': '../data/professional/prof_lvl3.h5',
        },
    }

def main():
    train(params)
    infer(params)
if __name__ == "__main__":
    main()
