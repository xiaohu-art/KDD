import json
import copy
import numpy as np
import yaml
import math
from pyproj import Geod
from shapely.geometry import Point, LineString
from pypower.api import ppoption, runpf
import os, sys

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

BASE = 100000000
geod = Geod(ellps="WGS84")

def str2int(json_data):
    """
    将json的“键”由string转为int
    """
    new_dict = {}
    for key, value in json_data.items():
        new_dict[int(key)] = value
    return new_dict

class ElecNoStep:
    def __init__(self, config, topology, power_10kv):
        self.basic_config = config["basic_config"]
        self.index_config = config["index_config"]
        self.topology = topology
        self.power_10kv_valid = power_10kv
        self.get_fixed_info()
        self.reset()

    def get_fixed_info(self):
        # 状态集合,i,级别，电源，500，220，110，10对应着12345 valid 正常， ruined 直接打击 cascaded 级联 stop 上游没电导致的停电
        self.facility_state_dict_valid = {
            i: {"valid": set(), "ruined": set(), "cascaded": set(),"stopped": set()} for i in range(1, 6)
        }
        for i, key in enumerate(self.topology):
            if i == 5:
                break
            self.facility_state_dict_valid[i+1]["valid"] = set(self.topology[key].keys())

        # 计算节点数量
        self.power_num = len(self.topology['power'])
        self.trans500_num = len(self.topology['500kv'])
        self.trans220_num = len(self.topology['220kv'])
        self.trans110_num = len(self.topology['110kv'])
        
        # relation:distance
        self.distance = {}
        key_list = [None, 'power', '500kv', '220kv', '110kv']
        for key in self.topology:
            if key == '110kv':
                break
            for node_id1 in self.topology[key]:
                for node_id2 in self.topology[key][node_id1]["relation"]:
                    if node_id1 < node_id2:
                        key2 = key_list[node_id2 // BASE]
                        # print(node_id1)
                        line_string = LineString(
                            [
                                Point(
                                    self.topology[key][node_id1]['pos'][1], 
                                    self.topology[key][node_id1]['pos'][0]
                                ), 
                                Point(
                                    self.topology[key2][node_id2]['pos'][1], 
                                    self.topology[key2][node_id2]['pos'][0]
                                )
                            ]
                        )
                        self.distance[(node_id1, node_id2)] = geod.geometry_length(line_string) / 1e4
        
        # 110kv及以上正常功率
        self.power_110kv_up_valid = {}
        # 创建110kv上下游集合
        self.relation_data_110kv = {}
        for key in self.topology['110kv']:
            self.relation_data_110kv[key] = {'relaiton_10':set(), 'relation_220':set()}
            power_key = 0
            for node in self.topology['110kv'][key]['relation']:
                if node // BASE == 5:
                    self.relation_data_110kv[key]['relaiton_10'].add(node)
                    power_key += self.power_10kv_valid[node]
                else:
                    self.relation_data_110kv[key]['relation_220'].add(node)
            #110kv点功率
            self.power_110kv_up_valid[key] = power_key
        # print('110kv总功率',sum(self.power_110kv_up_valid.values()))

        # 正常潮流计算矩阵
        self.bus_mat_valid, self.gene_mat_valid, self.branch_mat_valid = self.get_flow_mat()
        # print(self.bus_mat_valid.shape, self.gene_mat_valid.shape, self.branch_mat_valid.shape)
        
        self.bus_mat = self.bus_mat_valid.copy()
        self.branch_mat = self.branch_mat_valid.copy()
        self.gene_mat = self.gene_mat_valid.copy()

        # 220kv以上各点功率
        self.power_110kv_up_valid.update(self.flow_calculate(init=True))

    # init=True时，重置，否则恢复到上一个状态
    def reset(self, init=True):
        if init:
            self.facility_state_dict = copy.deepcopy(self.facility_state_dict_valid)
            self.bus_mat = self.bus_mat_valid.copy()
            self.branch_mat = self.branch_mat_valid.copy()
            self.gene_mat = self.gene_mat_valid.copy()
            self.power_10kv = self.power_10kv_valid.copy()
            self.power_110kv_up = self.power_110kv_up_valid.copy()
        else:
            self.facility_state_dict = copy.deepcopy(self.facility_state_dict_record)
            self.bus_mat = self.bus_mat_record.copy()
            self.branch_mat = self.branch_mat_record.copy()
            self.gene_mat = self.gene_mat_record.copy()
            self.power_10kv = self.power_10kv_record.copy()
            self.power_110kv_up = self.power_110kv_up_record.copy()
    
    def record(self):
        self.facility_state_dict_record = copy.deepcopy(self.facility_state_dict)
        self.bus_mat_record = self.bus_mat.copy()
        self.branch_mat_record = self.branch_mat.copy()
        self.gene_mat_record = self.gene_mat.copy()
        self.power_10kv_record = self.power_10kv.copy()
        self.power_110kv_up_record = self.power_110kv_up.copy()

    def ruin(self, destory_list):
        self.record()
        use_flow = []
        # print(destory_list)
        for id in destory_list:
            if id in self.facility_state_dict[id // BASE]["valid"]:
                self.facility_state_dict[id // BASE]["ruined"].add(id)
                self.facility_state_dict[id // BASE]["valid"].remove(id)
                if id // BASE == 5:
                    self.power_10kv[id] = 0
                elif id // BASE == 4:
                    self.power_110kv_up[id] = 0
                    self.facility_state_dict[5]["stopped"] |= self.relation_data_110kv[id]['relaiton_10']
                    self.facility_state_dict[5]["valid"] -= self.relation_data_110kv[id]['relaiton_10']
                    for id_10 in self.relation_data_110kv[id]['relaiton_10']:
                        self.power_10kv[id_10] = 0
                else:
                    self.power_110kv_up[id] = 0
                    use_flow.append(id)

        if len(use_flow) > 0:
            power_up_220kv = self.flow_calculate(use_flow)
            for key in power_up_220kv:
                if power_up_220kv[key] == 0 and key in self.facility_state_dict[key // BASE]["valid"]:
                    # print('220kv断开', key)
                    self.facility_state_dict[key // BASE]["cascaded"].add(key)
                    self.facility_state_dict[key // BASE]["valid"].remove(key)
            self.power_110kv_up.update(power_up_220kv)

            invalid_220kv = self.facility_state_dict[3]["ruined"] | self.facility_state_dict[3]["cascaded"]
            for id_110 in self.relation_data_110kv:
                if id_110 in self.facility_state_dict[4]["valid"] and self.relation_data_110kv[id_110]['relation_220'] <= invalid_220kv:
                    self.facility_state_dict[4]["stopped"].add(id_110)
                    self.facility_state_dict[4]["valid"].remove(id_110)
                    self.power_110kv_up[id_110] = 0
                    self.facility_state_dict[5]["stopped"] |= self.relation_data_110kv[id_110]['relaiton_10']
                    self.facility_state_dict[5]["valid"] -= self.relation_data_110kv[id_110]['relaiton_10']
                    for id_10 in self.relation_data_110kv[id_110]['relaiton_10']:
                        self.power_10kv[id_10] = 0
        return sum(self.power_10kv.values())

    def flow_calculate(self, use_flow = [], init = False):
        """
        潮流计算，返回损坏的220kv以上节点和220kv以上节点功率
        """
        # 级联循环
        count = 0
        flag = 1
        destory_first = True
        destory_power = []
        destory_220kv = []
        destory_110kv = []
        for key in use_flow:
            if key // BASE == 4:
                destory_110kv.append(key)
            elif key // BASE == 3:
                destory_220kv.append(key)
            else:
                destory_power.append(key)
        while_num = 0
        while flag:
            if while_num > 20:
                break
            if destory_first:
                self.delete_power(destory_power)
                flag_power = np.sum(self.gene_mat[self.trans500_num:self.trans500_num+self.power_num, self.index_config['GEN_STATUS']])
                if flag_power < 0.1:
                    break
                self.delete_220kv(destory_220kv)
                self.delete_110kv(destory_110kv)
                destory_first = False
            else:
                if cascade_power.size:
                    count += 1
                self.delete_power(cascade_power)
                flag_power = np.sum(self.gene_mat[self.trans500_num:self.trans500_num+self.power_num, self.index_config['GEN_STATUS']])
                if flag_power < 0.1:
                    break
                self.delete_220kv(cascade_220kv)
                self.delete_110kv()


            with HiddenPrints():
                ppc = {
                    "version": '2',
                    "baseMVA": 100.0,
                    "bus": self.bus_mat.copy(),
                    "gen": self.gene_mat.copy(),
                    "branch": self.branch_mat.copy()
                }
                ppopt = ppoption(OUT_GEN=0)
                result, _ = runpf(ppc, ppopt, fname = 'test')

            if init:
                # 保存power信息，id，正常功率，运行功率，比值
                self.info_gene = np.zeros((self.trans500_num+self.power_num, 4))
                self.info_gene[:,[0, 1]] = result["gen"][:,[0,1]]
                # 保存220kv信息，id，正常功率，运行功率，比值
                self.info_220kv = np.zeros((self.trans220_num, 4))
                index_220kv = np.where(result['branch'][:,self.index_config["TAP"]]==2)[0]
                self.info_220kv[:,[0, 1]] = result["branch"][index_220kv][:,[0,13]]
                break
            
            index_220kv = np.where(result['branch'][:,self.index_config["TAP"]]==2)[0]
            # print('*'*50)
            # print('220kv', index_220kv.shape)
            self.info_220kv[:,2] = result["branch"][index_220kv][:,self.index_config["PF"]]
            self.info_220kv[:,3] = abs(self.info_220kv[:,2] / self.info_220kv[:,1])
            self.info_gene[:,2] = result["gen"][:,1]
            self.info_gene[:,3] = abs(self.info_gene[:,2] / self.info_gene[:,1])
            cascade_220kv = np.where(
                (self.info_220kv[:,3] >= self.basic_config['up_220'])
            )[0]
            cascade_power = np.where(
                (self.info_gene[:,3] >= self.basic_config['up_power'])
            )[0]
            for i, id in enumerate(cascade_power):
                if id < self.trans500_num:
                    cascade_power[i] += BASE * 2
                else:
                    cascade_power[i] += BASE * 1 - self.trans500_num
            # print("cascade_power", cascade_power)
            # print("cascade_220kv", cascade_220kv)
            # print(cascade_power, cascade_220kv)
            # print("info_220kv", self.info_220kv[cascade_220kv])
            flag = len(cascade_220kv) + len(cascade_power)
            while_num += 1

        update_power_dict = {}
        
        if flag_power < 0.1:
            for i in range(self.power_num):
                id = i + BASE
                update_power_dict[id] = 0
            for i in range(self.trans500_num):
                id = i + BASE * 2
                update_power_dict[id] = 0
            for i in range(self.trans220_num):
                id = i + BASE * 3
                update_power_dict[id] = 0
        else:
            # 更新destory_power中的220kv和500kv功率, 保存220kv及以上{id:功率}到update_power_dict
            update_trans500_power = result["gen"][0:self.trans500_num, self.index_config["PG"]]
            update_gene_power = result["gen"][self.trans500_num:self.trans500_num+self.power_num, self.index_config["PG"]]
            index_220kv = np.where(result['branch'][:,self.index_config["TAP"]]==2)[0]
            update_trans220_power = result["branch"][index_220kv][:, self.index_config["PF"]]

            for i in range(len(update_gene_power)):
                id = i + BASE
                update_power_dict[id] = update_gene_power[i] * 1e6
            for i in range(len(update_trans500_power)):
                id = i + BASE * 2
                update_power_dict[id] = update_trans500_power[i] * 1e6
            for i in range(len(update_trans220_power)):
                id = i + BASE * 3
                update_power_dict[id] = update_trans220_power[i] * 1e6

        return update_power_dict

    def delete_power(self, destory_power_list):
        # 处理电源
        for id in destory_power_list:
            if id // BASE == 2:
                self.gene_mat[id % BASE, self.index_config['GEN_STATUS']] = 0
                self.branch_mat[id % BASE, self.index_config['BR_STATUS']] = 0
            elif id // BASE == 1:
                self.gene_mat[id % BASE + self.trans500_num, self.index_config['GEN_STATUS']] = 0
                # self.branch_mat[id % BASE + self.trans500_num, self.index_config['BR_STATUS']] = 0

    def delete_220kv(self, destory_220kv_list):
        # 处理220kv
        for id in destory_220kv_list:
            connect_110kv = np.where(
                self.branch_mat[:,self.index_config["F_BUS"]] == id % BASE + 2 * (self.power_num + self.trans500_num) + self.trans220_num
            )[0]
            # print('len(connect_110kv)', len(connect_110kv))
            self.branch_mat = np.delete(self.branch_mat, connect_110kv, axis=0)
            # self.branch_mat[id % BASE + self.power_num + self.trans500_num, self.index_config['BR_STATUS']] = 0
        condi = np.where(self.branch_mat[:,self.index_config["T_BUS"]] > 250)[0]
        # print("condi",len(condi))

    def delete_110kv(self, destory_110kv = []):
        # 直接摧毁, 此时110kv branch，bus均有, 潮流计算矩阵在不断删除，智能用np.where查找
        if destory_110kv:
            for id in destory_110kv:
                bus_id = id + 2 * (self.trans500_num + self.power_num + self.trans220_num)
                condi_110kv = np.where(
                    self.branch_mat[:, 1] == bus_id
                )[0]
                self.branch_mat = np.delete(self.branch_mat, condi_110kv, axis=0)
                bus_110kv = np.where(
                    self.bus_mat[:,0] == bus_id
                )[0]
                self.bus_mat = np.delete(self.bus_mat, bus_110kv, axis=0)
        # 110kv的上游两个220kv如果被删除，删除110kv
        else:
            for id in range(self.trans110_num):#110kv数量
                bus_id = id + 2 * (self.trans500_num + self.power_num + self.trans220_num)
                condi_110kv = np.where(
                    self.branch_mat[:, 1] == bus_id
                )[0]
                if condi_110kv.size == 0:
                    # 删除bus
                    bus_110kv = np.where(
                        self.bus_mat[:,0] == bus_id
                    )[0]
                    if len(bus_110kv) != 0:#有可能已经删了
                        self.bus_mat = np.delete(self.bus_mat, bus_110kv, axis=0)
    
    def get_flow_mat(self):
        """
        由self.topology获取正常运行的flow_mat，每个拓扑只运行一次
        参看pypower文档
        """
        Bus_data = []
        Generator_data = []
        Branch_data = []

        # Bus data Generate
        Bus_num = 2 * (self.trans500_num + self.power_num + self.trans220_num) + self.trans110_num 
        for bus_id in range(Bus_num):
            type_id = 1
            Pd = 0
            Qd = 0
            # 与500kv连接的母线
            if bus_id < self.trans500_num:
                type_id = 3
                Vm = 5/1.1
            # 与发电厂连接的母线
            elif bus_id < self.trans500_num + self.power_num:
                type_id = 3
                Vm = 5/1.1
            # 500kv、发电厂出线以及220kv入线
            elif bus_id < 2 * (self.trans500_num + self.power_num) + self.trans220_num:
                Vm = 2
            # 220kv出线
            elif bus_id < 2 * (self.trans500_num + self.power_num + self.trans220_num):
                Vm = 1
            # 110kv入线
            else:
                index_110 = int(bus_id - 2 * (self.trans500_num + self.power_num + self.trans220_num)) + 4 * BASE
                Pd = self.power_110kv_up_valid[index_110] / 1e6
                Qd = Pd * math.sqrt((1 / self.basic_config["cos_phi"]) ** 2 - 1)
                Vm = 1
            Bus_data.append(
                [
                    bus_id,
                    type_id,
                    Pd,
                    Qd,
                    self.basic_config["Gs"],
                    self.basic_config["Bs"],
                    self.basic_config["area"],
                    Vm,
                    self.basic_config["Va"],
                    self.basic_config["baseKV"],
                    self.basic_config["Zone"],
                    Vm * 1.5,   #Vmax
                    Vm / 1.5    #Vmin
                ]
            )

        # Generator data generate
        Generator_num = self.trans500_num + self.power_num
        for bus_id in range(Generator_num):
            # 500kv电源
            if bus_id < self.trans500_num:
                Pg = 0
                Qg = 0
            # 发电厂
            else:
                Pg = 0
                Qg = 0
            Generator_data.append(
                [
                    bus_id,
                    Pg,
                    Qg,
                    self.basic_config["Qmax"],
                    self.basic_config["Qmin"],
                    self.basic_config["Vg"],
                    self.basic_config["mbase"],
                    self.basic_config["status"],
                    self.basic_config["Pmax"],
                    self.basic_config["Pmin"],
                    self.basic_config["Pc1"],
                    self.basic_config["Pc2"],
                    self.basic_config["Qc1min"],
                    self.basic_config["Qc1max"],
                    self.basic_config["Qc2min"],
                    self.basic_config["Qc2max"],
                    self.basic_config["ramp_agc"],
                    self.basic_config["ramp_10"],
                    self.basic_config["ramp_30"],
                    self.basic_config["ramp_q"],
                    self.basic_config["apf"],
                ]
            )

        # 变压器支路
        Transformer_num = self.trans500_num + self.power_num + self.trans220_num
        for i in range(Transformer_num):
            r = 0
            x = 1e-6
            b = 0
            if i < self.trans500_num + self.power_num:
                fbus = i
                tbus = fbus + self.trans500_num + self.power_num
                ratio = 5/2.2
            else:
                fbus = i + self.trans500_num + self.power_num
                tbus = fbus + self.trans220_num
                ratio = 2
            Branch_data.append(
                [
                    fbus,
                    tbus,
                    r,
                    x,
                    b,
                    self.basic_config["rateA"],
                    self.basic_config["rateB"],
                    self.basic_config["rateC"],
                    ratio,
                    self.basic_config["angle"],
                    self.basic_config["status"],
                    self.basic_config["angmin"],
                    self.basic_config["angmax"],
                ]
            )

        for relation in self.distance:
            if relation[1] // BASE == 4:
                fbus = relation[0] % BASE + 2 * (self.trans500_num + self.power_num) + self.trans220_num
                tbus = relation[1] % BASE + 2 * (self.trans500_num + self.power_num + self.trans220_num)
            elif relation[1] // BASE == 3:
                tbus = relation[1] % BASE + 2 * (self.trans500_num + self.power_num)
                if relation[0] // BASE == 3:
                    fbus = relation[0] % BASE + 2 * (self.trans500_num + self.power_num)
                elif relation[0] // BASE == 2:
                    fbus = relation[0] % BASE + self.trans500_num + self.power_num
                else:
                    fbus = relation[0] % BASE + 2 * (self.trans500_num) + self.power_num
            else:
                fbus = relation[0] % BASE
                tbus = relation[1] % BASE
            # distance = self.distance[relation]
            # r = 0.000023 * distance
            # x = 0.000031 * distance
            # b = 3.765 * distance * 1e-8
            r = 1e-3
            x = 1e-3
            b = 1e-8
            ratio = 1
            Branch_data.append(
                [
                    fbus,
                    tbus,
                    r,
                    x,
                    b,
                    self.basic_config["rateA"],
                    self.basic_config["rateB"],
                    self.basic_config["rateC"],
                    ratio,
                    self.basic_config["angle"],
                    self.basic_config["status"],
                    self.basic_config["angmin"],
                    self.basic_config["angmax"],
                ]
            )

        return np.array(Bus_data), np.array(Generator_data), np.array(Branch_data)

if __name__ == "__main__":
    # 配置文件
    with open("./datasets/config.yml") as f:
        config = yaml.safe_load(f)
    # 10kv功率，固定值
    with open('./datasets/power_10kv.json') as json_file:
        power_10 = json.load(json_file)
    # id转为int
    power_10 = str2int(power_10)
    # 读取拓扑数据，每读一个拓扑数据，就创建一个对象
    with open('./datasets/all_dict.json') as json_file:
        topology = json.load(json_file)
    # id转为int
    for key in topology:
        topology[key] = str2int(topology[key])
    elec = ElecNoStep(config, topology, power_10)
    initial_power = elec.ruin([])
    print(initial_power)

    nodes = [300000010, 300000021]
    current_power = elec.ruin(nodes)
    print(current_power)
    nodes = [300000035]
    current_power = elec.ruin(nodes)
    print(current_power)
    initial_power = elec.ruin([])
    print(initial_power)
    elec.reset()


    # print('all_dict500')
    # with open('./datasets/all_dict.json') as json_file:
    #     topology = json.load(json_file)
    # # id转为int
    # for key in topology:
    #     topology[key] = str2int(topology[key])
    # elec_no_step_500 = ElecNoStep(config, topology, power_10)

    # # 一个打击策略，打击两次，每次打击节点数量不同
    # once_destroy = [[100000008], [200000009], [200000010]]
    # for hit in once_destroy:
    #     power_summed = elec_no_step_500.ruin(hit)
    #     print(power_summed)
    # elec_no_step_500.reset()

