from train import train
from evaluate import infer

params = {
        'rec_path' : '../test_data/user_item_interact.h5',
        'course_path': '../test_data/course_info.h5',
        'course_id_map_path' : '../test_data/course_id_map.h5',
        'batch_size':20,
        'infer_size':16,
        'num_epochs':4,
        'lr':1e-4,
        'device':'cuda',
        'num_layers':2, # encoder 层数
        'd_model':16, # encoder 隐藏层状态
        'd_ff':32,
        'num_heads':3,
        'd_kv':8,
        'dropout_rate':0.1,
        'feed_forward_proj':'relu',
        'input_emb_dim':32, # Bert embedding dimension
        'target_emb_dim':32,
        'temperature':0.07, # infoNCE temperature
        'log_path':'./logs/tiger.log',
        'seed':42,
        'save_path':'./ckpt/tiger.pth',
        'early_stop':10,
        'topk_list':[5,10,20],
    }
def main():
    train(params)
    infer(params)
if __name__ == "__main__":
    main()
