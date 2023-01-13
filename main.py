from model import ElecGraph, TraGraph, ElecNoStep, DQN
from utils import init_env

import time
import torch
import argparse

from colorama import init
from colorama import Fore,Back,Style

init()

# parser = argparse.ArgumentParser(description='degree attack')

# parser.add_argument('--epoch', type=int, default=10000, help='Times to train')
# parser.add_argument('--batch', type=int, default=20, help='Data number used to train')
# parser.add_argument('--gamma', type=float, default=0.9, help='Related to further reward')
# parser.add_argument('--lr', type=float, default=0.01, help='Laerning rate')
# parser.add_argument('--epsilon', type=float, default=0.6, help='Epsilon greedy policy')
# parser.add_argument('--layer', type=int, default=3, help='GNN embedding layers')

# args = parser.parse_args()

EFILE = './data/all_dict_correct.json'
TFILE = './data/road_junc_map.json'
ept = './embedding/elec_feat.pt'
tpt = './embedding/tra_feat.pt'
EMBED_DIM = 64
HID_DIM = 128
FEAT_DIM = 64
KHOP=5
EPOCH = 500
LR = 0.01
BATCH_SIZE = 20
GAMMA = 0.9
EPSILON = 0.6
MEMORY_CAPACITY = 1000
TARGET_REPLACE_ITER = 25

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
    
    config, power_10, power_load, topology = init_env()
    elec_env = ElecNoStep(config, topology, power_10, power_load)

    agent = DQN(in_dim=EMBED_DIM,
                hid_dim=HID_DIM,
                out_dim=EMBED_DIM,
                memory_capacity=MEMORY_CAPACITY,
                iter=TARGET_REPLACE_ITER,
                batch_size=BATCH_SIZE,
                lr=LR,
                epsilon=EPSILON,
                gamma=GAMMA)

    features = egraph.feat.detach()
    initial_power = elec_env.ruin([])
    limit = initial_power * 0.2

    print(Fore.RED,Back.YELLOW,'\nbegin attacking ...')
    print(Style.RESET_ALL)
    for epoch in range(EPOCH):

        t = time.time()
        num = egraph.node_num
        state = torch.sum(features, dim=0) / num
        total_reward = 0
        choosen = []
        # exist = list(range(num))
        exist = [node for node,id in egraph.node_list.items() if id//100000000 > 2]
        elec_env.reset()

        done = False

        while not done:
            hpower = elec_env.ruin([])
            node = agent.choose_node(features, state, choosen, exist)
            choosen.append(node)
            exist.remove(node)
            num -= 1

            tpower = elec_env.ruin([egraph.node_list[node]])
            _state = (state * (num+1) - features[node]) / num

            reward = (hpower - tpower) / 1e05
            total_reward += reward

            agent.store_transition(state.data.cpu().numpy(),
                                    node, reward,
                                   _state.data.cpu().numpy(),)
                        
            if agent.memory_num > agent.mem_cap:
                agent.learn(features)

            state = _state

            # if tpower < limit:
            if len(choosen) == 10:
                done = True
                print(Fore.RED,Back.YELLOW)
                print("\nEpoch:", '%03d' % (epoch + 1), " total reward = ", "{:.5f} ".format(total_reward),
                        " node num =", "%04d" % (egraph.node_num - num)
                        )
                print(Style.RESET_ALL)
