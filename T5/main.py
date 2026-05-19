from train import train
from evaluate import infer
from item_encode import encode
task_ID = 'task1'
params = {
        'task_id': task_ID, 
        'rec_path' : '../data/user_item_interact.h5',
        'course_path': '../data/course_info.h5',
        'course_id_map_path' : '../data/course_id_map.h5',
        'item_emb_h5_path': '../data/course_item_embs.h5',
        'user_emb_h5_path':'../data/user_profile_embs.h5',
        'encode_model': 'bert-base-uncased',
        'encode_batch_size':20,
        'encode_max_len':512,
        'batch_size':256,
        'infer_size':256,
        'num_epochs':100,
        'lr':1e-3,
        'device':'cuda:1',
        'num_layers':2, # encoder 层数 (t5-small)
        'd_model':512, # encoder 隐藏层状态 (t5-small)
        'd_ff':256,
        'num_heads':4,
        'd_kv':16,
        'dropout_rate':0.3,
        'feed_forward_proj':'relu',
        'input_emb_dim':768, # Bert embedding dimension
        'target_emb_dim':768,
        'temperature':0.07, # infoNCE temperature
        'pretrained_path':'../pretrained/t5-small',
        'log_path':'./logs/tiger.log',
        'seed':42,
        'save_path':f'./ckpt/tiger_{task_ID}.pth',
        'params_path':'./T5-results.csv',
        'early_stop':10,
        'topk_list':[2,5,10,20],
        'loss_plot_path':f'./loss_picture/{task_ID}.png'
    }
def main():
    encode(params)
    train(params)
    infer(params)
if __name__ == "__main__":
    main()
