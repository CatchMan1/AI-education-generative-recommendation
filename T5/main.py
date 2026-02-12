from train import train
from evaluate import infer

params = {
        'rec_path' : '../data/user_item_interact.h5',
        'course_path': '../data/course_info.h5',
        'course_id_map_path' : '../data/course_id_map.h5',
        'batch_size':256,
        'infer_size':256,
        'num_epochs':30,
        'lr':1e-4,
        'device':'cuda',
        'num_layers':2, # encoder 层数
        'd_model':256, # encoder 隐藏层状态
        'd_ff':512,
        'num_heads':8,
        'd_kv':64,
        'dropout_rate':0.3,
        'feed_forward_proj':'relu',
        'input_emb_dim':768, # Bert embedding dimension
        'target_emb_dim':768,
        'temperature':0.07, # infoNCE temperature
        'log_path':'./logs/tiger.log',
        'seed':42,
        'save_path':'./ckpt/tiger.pth',
        'early_stop':10,
        'topk_list':[5,10,20],
        'loss_plot_path':'./loss_picture/task1.png'
    }
def main():
    train(params)
    infer(params)
if __name__ == "__main__":
    main()
