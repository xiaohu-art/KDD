import argparse
import time

import numpy as np
import torch
from colorama import Back, Fore, Style, init
from model import DQN, Bigraph, ElecGraph, TraGraph
from utils import calculate_pairwise_connectivity, influenced_tl_by_elec, init_env

init()

parser = argparse.ArgumentParser(description='begraph attack')

parser.add_argument('--epoch', type=int, default=1000, help='Times to train')
parser.add_argument('--batch', type=int, default=20, help='Data number used to train')
parser.add_argument('--gamma', type=float, default=0.9, help='Related to further reward')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--epsilon', type=float, default=0.6, help='Epsilon greedy policy')
parser.add_argument('--feat', type=str, required=True, help='pre-train feat or random feat')
parser.add_argument('--label', type=str, required=True, help='train or test')

args = parser.parse_args()

FILE = './data/e10kv2tl.json'
EFILE = './data/electricity/all_dict_correct.json'
TFILE1 = './data/road/road_junc_map.json'
TFILE2 = './data/road/road_type_map.json'
TFILE3 = './data/road/tl_id_road2elec_map.json'
ept = './embedding/elec_feat.pt'
tpt = './embedding/primary.pt'
bpt = ('./embedding/bifeatures/bi_elec_feat.pt', './embedding/bifeatures/bi_tra_feat.pt') 
EMBED_DIM = 64
HID_DIM = 128
FEAT_DIM = 64
KHOP=5
EPOCH = args.epoch
LR = args.lr
BATCH_SIZE = args.batch
GAMMA = args.gamma
EPSILON = args.epsilon
MEMORY_CAPACITY = 2000
TARGET_REPLACE_ITER = 25
BASE = 100000000

device = torch.device("cuda:1" if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":

    egraph = ElecGraph(file=EFILE,
                    embed_dim=EMBED_DIM,
                    hid_dim=HID_DIM,
                    feat_dim=FEAT_DIM,
                    khop=KHOP,
                    epochs=500,
                    pt_path=ept)

    tgraph = TraGraph(file1=TFILE1, file2=TFILE2, file3=TFILE3,
                    embed_dim=EMBED_DIM,
                    hid_dim=HID_DIM,
                    feat_dim=FEAT_DIM,
                    r_type='primary',
                    khop=KHOP,
                    epochs=300,
                    pt_path=tpt)

    bigraph = Bigraph(efile=EFILE, tfile1=TFILE1, tfile2=TFILE2, tfile3=TFILE3, file=FILE,
                    embed_dim=EMBED_DIM,
                    hid_dim=HID_DIM,
                    feat_dim=FEAT_DIM,
                    r_type='primary',
                    subgraph = (egraph, tgraph),
                    khop=KHOP,
                    epochs=600,
                    pt_path=bpt)

    print(bigraph.nxgraph)
    exit()

    if args.feat == "ptr":
        elec_feat = bigraph.feat['power'].detach()
        road_feat = bigraph.feat['junc'].detach()
        features = torch.vstack((elec_feat,road_feat))
        features = features.to(device)
        MODEL_PT = './model_param/bi_ptr.pt'
    elif args.feat == "rdn":
        try:
            features = torch.load('./random/bi_rdn_emb.pt')
        except:
            features = torch.rand(bigraph.node_num, EMBED_DIM).to(device)
            torch.save(features, './random/bi_rdn_emb.pt')
        MODEL_PT = './model_param/bi_rdn.pt'
                    
    agent = DQN(in_dim=EMBED_DIM,
                hid_dim=HID_DIM,
                out_dim=EMBED_DIM,
                memory_capacity=MEMORY_CAPACITY,
                iter=TARGET_REPLACE_ITER,
                batch_size=BATCH_SIZE,
                lr=LR,
                epsilon=EPSILON,
                gamma=GAMMA,
                label=args.label,
                model_pt=MODEL_PT)
    
    elec_env = init_env()
    initial_power = elec_env.ruin([])

    print()
    print(Fore.RED,Back.YELLOW,'begin attacking ...')
    print(Style.RESET_ALL)

    if args.label == 'test':
        
        t = time.time()
        g = bigraph.nxgraph
        tgc = tgraph.nxgraph.copy()
        num = bigraph.node_num
        state = torch.sum(features, dim=0) / num

        total_reward = 0
        choosen = []
        choosen_road = []
        choosen_elec = []
        elec_env.reset()
        result = []

        origin_val = calculate_pairwise_connectivity(tgc)
        t_val = calculate_pairwise_connectivity(tgc) / origin_val
        tpower = initial_power

        done = False
        while not done:

            node = agent.attack(features, state, choosen)

            if bigraph.node_list[node]//BASE < 3:
                continue
            choosen.append(node)
            if bigraph.node_list[node]//BASE == 9:
                choosen_road.append(bigraph.node_list[node])
            else:
                choosen_elec.append(bigraph.node_list[node])
            num -= 1
            
            current_power,elec_state = elec_env.ruin(choosen_elec,flag=0)
            choosen_road += influenced_tl_by_elec(elec_state, bigraph.elec2road, tgraph.nxgraph)
            tgc.remove_nodes_from(choosen_road)
            val = calculate_pairwise_connectivity(tgc) / origin_val

            v_elec = current_power / initial_power
            v_road = val

            _state = (state * (num+1) - features[node]) / num
            state = _state
    
            result.append([len(choosen), 0.5*v_elec + 0.5*v_road])

            if len(choosen) == 50:
                done = True
        
        result = np.array(result)
        print(Fore.RED,Back.YELLOW,'saving RL attack result ...')
        print(Style.RESET_ALL)
        np.savetxt('./result/bi_result_'+args.feat+'.txt', result)

    elif args.label == 'train':

        g = bigraph.nxgraph
        result_reward = []
        for epoch in range(EPOCH):

            
            t = time.time()
            num = bigraph.node_num
            state = torch.sum(features, dim=0) / num
            total_reward = 0
            choosen = []
            choosen_road = []
            choosen_elec = []
            exist = [node for node,id in bigraph.node_list.items() if id//100000000 > 2]

            elec_env.reset()
            tgc = tgraph.nxgraph.copy()

            origin_val = calculate_pairwise_connectivity(tgc)
            t_val = calculate_pairwise_connectivity(tgc) / origin_val
            tpower = initial_power 
            done = False
            result = []

            while not done:

                h_val = t_val
                hpower = tpower
                node = agent.choose_node(features, state, choosen, exist)
                if bigraph.node_list[node]//BASE < 3:
                    continue
                choosen.append(node)
                exist.remove(node)
                if bigraph.node_list[node]//BASE == 9:
                    choosen_road.append(bigraph.node_list[node])
                else:
                    choosen_elec.append(bigraph.node_list[node])
                num -= 1
                _state = (state * (num+1) - features[node]) / num
                tpower,elec_state = elec_env.ruin(choosen_elec,flag=0)
                choosen_road += influenced_tl_by_elec(elec_state, bigraph.elec2road, tgraph.nxgraph)
                tgc.remove_nodes_from(choosen_road)
                t_val = calculate_pairwise_connectivity(tgc) / origin_val

                reward_elec = (hpower - tpower) / initial_power
                reward_road = h_val - t_val
                reward = (0.5 * reward_road + 0.5 * reward_elec) * 1000
                total_reward += reward 

                agent.store_transition(state.data.cpu().numpy(),
                                        node, reward,
                                    _state.data.cpu().numpy())
                
                if agent.memory_num > agent.mem_cap:
                    
                    agent.learn(features)

                state = _state

                if len(choosen) == 20:
                    result_reward.append((epoch+1,total_reward))
                    done = True
                    result.append([epoch, total_reward])
                    print("Epoch:", '%03d' % (epoch + 1), " total reward = ", "{:.5f} ".format(total_reward),
                            " time =", "{:.4f}".format(time.time() - t)
                            )
            
            if epoch % 50 == 0:
                torch.save(agent.enet.state_dict(), './model/'+args.feat+'/'+str(epoch)+'.pt')

        np.savetxt('./result/bi_'+args.feat+'reward.txt',np.array(result_reward))
        
