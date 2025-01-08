import time
import argparse
import pickle
from model import *
from utils import *
from tqdm import tqdm


def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='tmall', help='cosmetics/tmall')
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--hiddenSize', type=int, default=100)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--activate', type=str, default='relu')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay.')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay.')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty ')
parser.add_argument('--num_layer', type=float, default=1, help='number of layer')
parser.add_argument('--dropout_gcn', type=float, default=0, help='Dropout rate.')
parser.add_argument('--dropout_local', type=float, default=0.5, help='Dropout rate.')
parser.add_argument('--dropout_global', type=float, default=0.5, help='Dropout rate.')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=3)
opt = parser.parse_args()


def main():
    init_seed(2021)

    if opt.dataset == 'cosmetics':
        num_item = 31895    #(31894+1)
        num_brnd = 228      #(227+1)
        num_cat = 457       #(456+1)
    elif opt.dataset == 'tmall':
        num_item = 40728    #(40727+1)
        num_brnd = 4161     #(4160+1)
        num_cat = 712       #(711+1)


    train_data = pickle.load(open('datasets/' + opt.dataset + '/train.txt', 'rb'))
    test_data = pickle.load(open('datasets/' + opt.dataset + '/test.txt', 'rb'))
    brand = pickle.load(open('datasets/' + opt.dataset + '/brand.txt', 'rb'))
    category = pickle.load(open('datasets/' + opt.dataset + '/category.txt', 'rb'))
    
    train_data = Data(train_data, brand, category)
    test_data = Data(test_data, brand, category)
    num_total = num_item + num_brnd + num_cat -2
    # Combined Side Information-driven Graph Neural Networks
    model = trans_to_cuda(CSI_GNN(opt, num_item, num_brnd, num_cat, num_total, brand, category))
    
    result_map = {}
    print(opt)
    start = time.time()
    best_result_k10 = [0, 0]
    best_result_k20 = [0, 0]
    best_result_k30 = [0, 0]
    best_result_k40 = [0, 0]
    best_result_k50 = [0, 0]
    
    
    best_epoch_k10 = [0, 0]
    best_epoch_k20 = [0, 0]
    bad_counter_k20 = bad_counter_k10 = 0

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        hit_k10, mrr_k10, hit_k20, mrr_k20, hit_k30, mrr_k30, hit_k40, mrr_k40, hit_k50, mrr_k50 = train_test(model, train_data, test_data) 
        
        flag_k10 = 0
        if hit_k10 >= best_result_k10[0]:
            best_result_k10[0] = hit_k10
            best_epoch_k10[0] = epoch
            flag_k10 = 1
        if mrr_k10 >= best_result_k10[1]:
            best_result_k10[1] = mrr_k10
            best_epoch_k10[1] = epoch
            flag_k10 = 1            
        bad_counter_k10 += 1 - flag_k10
        print("\n")            
        print('\tRecall@10, MMR@10: \t%.4f\t%.4f' % (best_result_k10[0], best_result_k10[1]))

        
        flag_k20 = 0
        if hit_k20 >= best_result_k20[0]:
            best_result_k20[0] = hit_k20
            best_epoch_k20[0] = epoch
            flag_k20 = 1
        if mrr_k20 >= best_result_k20[1]:
            best_result_k20[1] = mrr_k20
            best_epoch_k20[1] = epoch
            flag_k20 = 1
        bad_counter_k20 += 1 - flag_k20
        print('\tRecall@20, MMR@20: \t%.4f\t%.4f' % (best_result_k20[0], best_result_k20[1]))
        
        if hit_k30 >= best_result_k30[0]:
            best_result_k30[0] = hit_k30
        if mrr_k30 >= best_result_k30[1]:
            best_result_k30[1] = mrr_k30
        print('\tRecall@30, MMR@30: \t%.4f\t%.4f' % (best_result_k30[0], best_result_k30[1]))
        
        if hit_k40 >= best_result_k40[0]:
            best_result_k40[0] = hit_k40
        if mrr_k40 >= best_result_k40[1]:
            best_result_k40[1] = mrr_k40
        print('\tRecall@40, MMR@40: \t%.4f\t%.4f' % (best_result_k40[0], best_result_k40[1]))
        
        if hit_k50 >= best_result_k50[0]:
            best_result_k50[0] = hit_k50
        if mrr_k50 >= best_result_k50[1]:
            best_result_k50[1] = mrr_k50
        print('\tRecall@50, MMR@50: \t%.4f\t%.4f' % (best_result_k50[0], best_result_k50[1]))
        
        result_map.update({epoch:[best_result_k20[0], best_result_k20[1]]})
        pickle.dump(result_map, open('datasets/result-'+ str(opt.dataset) + '-layers' + str(opt.num_layer) + '.txt', 'wb'))
        
        if ((bad_counter_k20 >= opt.patience) and (bad_counter_k10 >= opt.patience)):
            break
        
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()
