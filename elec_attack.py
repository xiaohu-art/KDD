from model import ElecGraph, TraGraph, DQN
from utils import init_env, calculate_anc

import time
import torch
import numpy as np
import argparse

from colorama import init
from colorama import Fore,Back,Style

init()

parser = argparse.ArgumentParser(description='elec attack')

parser.add_argument('--epoch', type=int, default=1000, help='Times to train')
parser.add_argument('--batch', type=int, default=20, help='Data number used to train')
parser.add_argument('--gamma', type=float, default=0.9, help='Related to further reward')
parser.add_argument('--lr', type=float, default=0.01, help='Laerning rate')
parser.add_argument('--epsilon', type=float, default=0.6, help='Epsilon greedy policy')
parser.add_argument('--feat', type=str, required=True, help='pre-train feat or random feat')
parser.add_argument('--label', type=str, required=True, help='train or test')

args = parser.parse_args()

EFILE = './data/electricity/all_dict_correct.json'
TFILE = './data/road/road_junc_map.json'
ept = './embedding/elec_feat.pt'
tpt = './embedding/tra_feat.pt'
EMBED_DIM = 64
HID_DIM = 128
FEAT_DIM = 64
KHOP=5
EPOCH = args.epoch
LR = args.lr
BATCH_SIZE = args.batch
GAMMA = args.gamma
EPSILON = args.epsilon
MEMORY_CAPACITY = 1000
TARGET_REPLACE_ITER = 25

device = torch.device("cuda:1" if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":

    egraph = ElecGraph(file=EFILE,
                    embed_dim=EMBED_DIM,
                    hid_dim=HID_DIM,
                    feat_dim=FEAT_DIM,
                    khop=KHOP,
                    epochs=500,
                    pt_path=ept)

    tgraph = TraGraph(file=TFILE,
                    embed_dim=EMBED_DIM,
                    hid_dim=HID_DIM,
                    feat_dim=FEAT_DIM,
                    khop=KHOP,
                    epochs=300,
                    pt_path=tpt)
    
    elec_env = init_env()

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
                model_pt='./model_param/elec_ptr.pt')

    if args.feat == "ptr":
        features = egraph.feat.detach()
        features = features.to(device)
    elif args.feat == "rdn":
        features = torch.rand(egraph.node_num, EMBED_DIM).to(device)

    initial_power = elec_env.ruin([])

    print()
    print(Fore.RED,Back.YELLOW,'begin attacking ...')
    print(Style.RESET_ALL)

    if args.label == 'test':
        
        t = time.time()
        num = egraph.node_num
        state = torch.sum(features, dim=0) / num
        choosen = []
        elec_env.reset()

        result = []

        done = False
        while not done:
            node = agent.attack(features, state, choosen)
            if egraph.node_list[node]// 100000000 < 3:
                continue

            choosen.append(node)
            num -= 1

            current_power = elec_env.ruin([egraph.node_list[node]])
            _state = (state * (num+1) - features[node]) / num

            result.append([len(choosen), current_power])

            if len(choosen) == 20:
                done = True
        
        result = np.array(result)
        np.savetxt('./results/elec_result_'+args.feat+'.txt', result)


    elif args.label == 'train':

        for epoch in range(EPOCH):

            t = time.time()
            num = egraph.node_num
            state = torch.sum(features, dim=0) / num
            total_reward = 0
            choosen = []
            exist = [node for node,id in egraph.node_list.items() if id//100000000 > 2]
            elec_env.reset()

            done = False
            result = []

            while not done:
                hpower = elec_env.ruin([])
                node = agent.choose_node(features, state, choosen, exist)
                if egraph.node_list[node]// 100000000 < 3:
                    continue

                choosen.append(node)
                exist.remove(node)
                num -= 1

                tpower = elec_env.ruin([egraph.node_list[node]])
                _state = (state * (num+1) - features[node]) / num

                reward = (hpower - tpower) / 1e05
                total_reward += reward

                agent.store_transition(state.data.cpu().numpy(),
                                        node, reward,
                                    _state.data.cpu().numpy())
                            
                if agent.memory_num > agent.mem_cap:
                    agent.learn(features)

                state = _state

                if len(choosen) == 100:
                    done = True
                    result.append([epoch, total_reward])
                    print(Fore.RED,Back.YELLOW)
                    print("\nEpoch:", '%03d' % (epoch + 1), " total reward = ", "{:.5f} ".format(total_reward),
                            " time =", "{:.4f}".format(time.time() - t)
                            )
                    print(Style.RESET_ALL)

        torch.save(agent.enet.state_dict(), 'elec_'+args.feat+'.pt')