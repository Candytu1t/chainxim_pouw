import ast
import copy
import logging
import math
import random
import time
from array import array

import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt

import consensus
import functions
import global_var
import network
from data import Block, Chain, LocalChainTracker
from external import chain_growth, chain_quality, common_prefix, MAX_SUFFIX
from functions import for_name, INT_LEN, BYTE_ORDER
from miner import Miner, network_interface
from attack.adversary import Adversary



# 导入内存清理工具
try:
    from memory_leak_fix import cleanup_memory
    MEMORY_CLEANUP_AVAILABLE = True
except ImportError:
    def cleanup_memory(env_obj=None, consensus_obj=None):
        pass
    MEMORY_CLEANUP_AVAILABLE = False

logger = logging.getLogger(__name__)

MAX_INT = 2**32-1

class Environment(object):

    def __init__(self,  attack_param: dict = None,
                 consensus_param:dict = None, network_param:dict = None, 
                 genesis_blockheadextra:dict = None, genesis_blockextra:dict = None,
                 dataitem_param:dict = None):
        '''initiate the running environment

        Param
        -----
        attack_param: attack parameters (dict)
        consensus_param: consensus parameters (dict)
        network_param: network parameters (dict)
        genesis_blockheadextra: initialize variables in the head of genesis block (dict)
        genesis_blockextra: initialize variables in the genesis block (dict)
        dataitem_param: data item parameters (dict) 
        '''
        #environment parameters
        self.miner_num = global_var.get_miner_num()
        self.total_round = 0
        # load data item settings
        self.dataitem_params = dataitem_param
        if not dataitem_param['dataitem_enable']:
            self.dataitem_validator = None
            self.dataitem_params['max_block_capacity'] = 0
        else:
            self.dataitem_validator = set()
            if dataitem_param['dataitem_input_interval'] == 0:
                initial_dataitems = b''.join([self.generate_dataitem(1) for _ in range(dataitem_param['max_block_capacity'])])
                self.input_dataitem = array('Q', initial_dataitems).tobytes()

        # configure extra genesis block info
        consensus_type_str = global_var.get_consensus_type()
        consensus_type:consensus.Consensus = for_name(consensus_type_str)
        consensus_type.genesis_blockheadextra = genesis_blockheadextra
        consensus_type.genesis_blockextra = genesis_blockextra
        # Local chain tracker generation
        self.local_chain_tracker = LocalChainTracker()
        # generate miners
        self.miners:list[Miner] = []
        for miner_id in range(self.miner_num):
            miner = Miner(miner_id, consensus_param, dataitem_param['max_block_capacity'], dataitem_param['dataitem_input_interval'] == 0)
            miner.get_local_chain().set_switch_tracker_callback(self.local_chain_tracker.get_switch_tracker(miner_id))
            # miner.get_local_chain().set_merge_tracker_callback(self.local_chain_tracker.get_merge_tracker(miner_id))
            self.miners.append(miner)
        self.envir_create_global_chain()

        # generate network
        self.network:network.Network = for_name(global_var.get_network_type())(self.miners)

        # generate adversary
        self.adversary = Adversary(
            adver_num = attack_param['adver_num'], 
            attack_type = attack_param['attack_type'], 
            adversary_ids = attack_param['adversary_ids'], 
            network_type = self.network, 
            consensus_type = global_var.get_consensus_type(), 
            miner_list = self.miners, 
            # eclipse = attack_param['eclipse'], 
            global_chain = self.global_chain_prp,  # 传递提议者链（保持向后兼容）
            adver_consensus_param = copy.deepcopy(consensus_param), 
            attack_arg = attack_param['attack_arg'])
        # get honest miner IDs
        adversary_ids = self.adversary.get_adver_ids()
        self.honest_miner_ids = [miner.miner_id for miner in self.miners if miner.miner_id not in adversary_ids]
        self.confirm_delay = consensus_param['N']

        # configure oracles
        if consensus_type_str == 'consensus.SolidPoW':
            self.configure_oracles(consensus_param, adversary_ids)

        # set parameters for network
        self.network.set_net_param(**network_param)
        if dataitem_param['dataitem_enable']:
            self.network._dataitem_param = dataitem_param
        else:
            self.adversary.get_attack_type().behavior.ATTACKER_INPUT = None

        # add a line in chain data to distinguish adversaries from non-adversaries

        parameter_str = ('Parameters:\n' + 
            f'Miner Number: {self.miner_num} \n' + 
            f'Consensus Protocol: {consensus_type.__name__} \n' + 
            f'Network Type: {type(self.network).__name__} \n' + 
            f'Network Param:  {network_param} \n' + 
            f'Consensus Param: {consensus_param} \n')
        if adversary_ids:
            parameter_str += f'Adversary Miners: {adversary_ids} \n'
            parameter_str += f'Attack Execute Type: {self.adversary.get_attack_type_name()}'
            # parameter_str += f'  (Eclipse: {self.adversary.get_eclipse()}) \n'
            parameter_str += f"  (Adversary's q: {self.adversary.get_adver_q()}) \n"
        if isinstance(self.miners[0]._NIC, network_interface.NICWithTp):
            if self.dataitem_params['dataitem_enable']:
                parameter_str += f'Dataitem Param: {dataitem_param} \n'
            else:
                parameter_str += f'Block Size: {global_var.get_blocksize()} \n'
        print(parameter_str)
        with open(global_var.get_result_path() / 'parameters.txt', 'w+') as conf:
            print(parameter_str, file=conf)

    def configure_oracles(self, consensus_params:dict, adversary_ids:list):
        if consensus_params['q_distr'] == 'equal':
            q_list = [consensus_params['q_ave'] for _ in range(self.miner_num)]
        elif isinstance(eval(consensus_params['q_distr']), list):
            q_list = eval(consensus_params['q_distr'])
        else:   
            raise ValueError("q_distr should be a list or the string 'equal'")

        from consensus import RandomOracleRoot
        self.oracle_root = RandomOracleRoot()
        self.verifying_oracles = []
        self.mining_oracles = []
        attacker_oracle_verifying = self.oracle_root.get_verifying_oracle(adversary_ids)
        adversary_q = [q_list[id] for id in adversary_ids]
        attacker_oracle_mining = self.oracle_root.get_mining_oracle(sum(adversary_q))

        for id, miner in enumerate(self.miners):
            if id in adversary_ids:
                verifying_oracle = attacker_oracle_verifying
                mining_oracle = attacker_oracle_mining
            else:
                verifying_oracle = self.oracle_root.get_verifying_oracle([miner.miner_id])
                mining_oracle = self.oracle_root.get_mining_oracle(q_list[miner.miner_id])

            self.verifying_oracles.append(verifying_oracle)
            self.mining_oracles.append(mining_oracle)
            miner.consensus.set_random_oracle(mining_oracle=mining_oracle, verifying_oracle=verifying_oracle)

        self.adversary.get_attack_type().adver_consensus.set_random_oracle(mining_oracle=attacker_oracle_mining,
                                                                           verifying_oracle=attacker_oracle_verifying)

    def envir_create_global_chain(self):
        '''create global chain and its genesis block by copying
          local chain from the first miner.'''
        # 创建提议者链（PRP链）
        self.global_chain_prp = Chain()
        self.global_chain_prp.add_blocks(blocks=copy.deepcopy(self.miners[0].consensus.local_chain_prp.head))
        
        # 创建投票者链（VT链）列表
        self.global_vt_chains = []
        for vt_chain in self.miners[0].consensus.local_chain_vt:
            # 创建一个新的全局投票链实例
            global_chain_vt = Chain()
            # 复制对应vtChain的头部状态
            global_chain_vt.add_blocks(blocks=copy.deepcopy(vt_chain.head))
            # 将这个新创建并初始化的全局链添加到列表中
            self.global_vt_chains.append(global_chain_vt)
            
        # 保留旧的global_chain引用以保持向后兼容（指向提议者链）
        self.global_chain = self.global_chain_prp
        


    def generate_dataitem(self, round):
        dataitem = random.randint(1, MAX_INT).to_bytes(INT_LEN, BYTE_ORDER)
        dataitem += round.to_bytes(INT_LEN, BYTE_ORDER)
        self.dataitem_validator.add(dataitem)
        return dataitem        

    def on_round_start(self, round):
        self.local_chain_tracker.update_round(round)
        if getattr(self, 'oracle_root', None):
            self.oracle_root.reset_counters(self.mining_oracles)
        if self.dataitem_params['dataitem_enable']:
            if self.dataitem_params['dataitem_input_interval'] > 0:
                if round % self.dataitem_params['dataitem_input_interval'] == 1:
                    self.input_dataitem = self.generate_dataitem(round)
                else:
                    self.input_dataitem = b''
            else: # simplified dataitem generation without tracking
                new_dataitems = [self.generate_dataitem(round) for _ in range(self.dataitem_params['max_block_capacity'])]
                self.input_dataitem = b''.join(new_dataitems)
        else:
            self.input_dataitem = round.to_bytes(INT_LEN, BYTE_ORDER)

    def post_verification(self):
        '''post verification of the longest chain in the global_chain'''
        try:
            # construct consensus object with no verification limit
            consensus_obj:consensus.Consensus = copy.copy(self.miners[self.honest_miner_ids[0]].consensus)
            consensus_obj.miner_id = -1
            consensus_name = type(consensus_obj).__name__
            if consensus_name == 'SolidPoW':
                consensus_obj.set_random_oracle(None, self.oracle_root.get_verifying_oracle([]))
            
            # check the proposer chain
            longest_chain_prp = self.global_chain_prp.get_last_block()
            if longest_chain_prp and not consensus_obj.valid_chain(longest_chain_prp):
                return False
            
            # check voter chains
            for i, vt_chain in enumerate(self.global_vt_chains):
                longest_chain_vt = vt_chain.get_last_block()
                if longest_chain_vt and not consensus_obj.valid_chain(longest_chain_vt):
                    return False
            
            return True
            
        except Exception as e:
            return False

    def exec(self, num_rounds, max_height, process_bar_type, post_verification):

        '''
        调用当前miner的BackboneProtocol完成mining
        当前miner用add_block_direct添加上链
        之后gobal_chain用深拷贝的add_block_copy上链
        '''
        if process_bar_type != 'round' and process_bar_type != 'height':
            raise ValueError('process_bar_type should be \'round\' or \'height\'')
        ## 开始循环
        t_0 = time.time() # 记录起始时间
        cached_height = self.global_chain_prp.get_height()
        
        # 任务系统初始化（从chain-xim-master-kkt移植）
        qualified_votes_latency = {}
        data_to_prp_latency = {}
        threshold = 0
        vote_counts = global_var.get_vtcount()
        perm_prp = []
        perm_prp_tx = []
        m = global_var.get_vtnum()
        fork_prp = []
        dimension = global_var.get_N()
        N = global_var.get_vtnum()
        fanwei = global_var.get_fanwei()
        
        # # 从D盘读取任务文件
        # file_path = 'D:/task_8_2.txt'
        # with open(file_path, 'r') as f:
        #     data = f.read()
        
        # # 使用 ast.literal_eval 将字符串解析为 Python 对象
        # tasks_all = ast.literal_eval(f"{data}")
        # tasks = [tasks_all[i][:] for i in range(global_var.get_vtnum())]
        # tasks = [[0.1,0.1] for _ in range(N)]
        tasks = [np.random.uniform(-fanwei, fanwei, dimension).tolist() for _ in range(N)]
        
        # 保存生成的tasks到文件
        # with open('D:/task_8_2.txt', 'w') as f:
        #     f.write(str(tasks))
        t = tasks[:]
        global_var.set_tasks(t)
        renwudaodashijian = []
        sudu1 = []
        sudu2 = []
        tasks_wanchengshu = [0 for _ in range(global_var.get_vtnum())]
        number = global_var.get_number()
        blocknew1 = global_var.get_minority_block()
        blocknew2 = global_var.get_majority_block()
        vote_counts = {}
        vote_counts['B1'] = 0
        vote_counts['B2'] = 0
        
        try:
            for round in range(1, num_rounds+1):

                
                # 定期清理内存，防止内存泄漏（如果启用）
                try:
                    if global_var.get_memory_cleanup_enable() and MEMORY_CLEANUP_AVAILABLE and round % 50 == 0:
                        cleanup_memory(env_obj=self)
                except:
                    # 如果配置未初始化或其他错误，忽略清理
                    pass
                
                self.on_round_start(round)
                inputfromz = self.input_dataitem # 生成输入

                adver_tmpflag = 1
                if self.adversary.get_adver_num() != 0:
                    adverflag = random.randint(1,self.adversary.get_adver_num())
                
                for miner in self.miners:
                    miner.input_tape.append(("INSERT", inputfromz))
                    
                    # 攻击者
                    if miner._isAdversary:
                        if adver_tmpflag == adverflag:
                            self.adversary.excute_per_round(round = round)
                        adver_tmpflag = adver_tmpflag + 1
                        continue
                    
                    # 诚实矿工 run backbone protocol
                    # 记录BackboneProtocol执行时间
                    backbone_start_time = time.time()
                    new_msgs = miner.BackboneProtocol(round, tasks)
                    backbone_end_time = time.time()
                    backbone_duration = backbone_end_time - backbone_start_time
                    

                    
                    if new_msgs is not None:
                        # self.network.access_network(new_msgs,temp_miner.miner_id,round)
                        for msg in new_msgs:
                            if isinstance(msg, Block):
                                # print(f'get new msg{msg.name} in chain{msg.index}')
                                if msg.index == 1:  # 提议者区块
                                    self.global_chain_prp.add_block_forcibly(msg)
                                    global_var.set_arrive_time_prp(msg.name, round)
                                elif msg.index == 0:  # 交易区块
                                    tx = global_var.get_txnum()
                                    global_var.set_txho(tx)
                                    global_var.add_tx(f"TX{tx}")
                                    global_var.set_arrive_time_tx(msg.name, round)
                                elif msg.index < (global_var.get_vtnum() + 2):  # 投票者区块
                                    # 任务完成检查和处理
                                    if msg.blockhead.pouw_opt_x == [] or msg.blockhead.nonce3 != 0:
                                        tasks_wanchengshu[msg.index - 2] += 1
                                        # 生成新任务
                                        tasks[msg.index-2] = np.random.uniform(-fanwei, fanwei, dimension).tolist()
                                        msg.blockhead.benchmark_x = tasks[msg.index - 2]
                                        msg.blockhead.benchmark_y = functions.f1(msg.blockhead.benchmark_x)
                                        global_var.ADD_COMNUM()
                                    
                                    self.global_vt_chains[msg.index - 2].add_block_forcibly(msg)
                                    chain_id = msg.index - 2
                                    y = global_var.get_optimal(chain_id)
                                    opt = msg.blockhead.pouw_opt_y
                                    if y == 0 or opt < y:
                                        global_var.set_optimal(chain_id, opt)
                                        if opt not in global_var.get_opt_time(chain_id):
                                            global_var.set_opt_time(chain_id, opt, round)
                    miner.clear_tapes()
                    
                # diffuse(C)
                self.network.diffuse(round)
            
                # 全局链高度超过max_height之后就提前停止
                current_height = self.global_chain_prp.get_height()
                if current_height > max_height:
                    break
                # 根据process_bar_type决定进度条的显示方式
                if process_bar_type == 'round':
                    self.process_bar(round, num_rounds, t_0, 'round/s')
                elif current_height > cached_height and process_bar_type == 'height':
                    cached_height = current_height
                    self.process_bar(current_height, max_height, t_0, 'block/s')
        except KeyboardInterrupt:
            print("\nSimulation interrupted by user, saving existing results...")
        self.total_round = round
        assert post_verification or self.post_verification(), "Post verification failed"
        

        
        print(f"\nSimulation finished, global chains reach block height PRP:{self.global_chain_prp.get_height()}, VT:{[vt_chain.get_height() for vt_chain in self.global_vt_chains]} in {round} rounds")

        # ===== 任务完成时间与优化过程统计（对齐 chain-xim-master-thre 的实现思路）=====
        # 1) 统计每条 VT 链的优化过程（pouw_opt_y 随时间变化），并对时间做相对化
        all_chains_optimizations = []
        chain_index_iter = 0
        for vt_chain in self.global_vt_chains:
            chain_index_iter += 1
            print(f'now count vtchain {chain_index_iter}')
            current_block = vt_chain.get_last_block()
            if current_block is None or 'B0' in current_block.name:
                continue

            # 回溯到第一个满足 nonce3 != 0 的块（纯 PoW 块），作为一次任务的“结束点”
            while current_block and ('B0' in current_block.name or getattr(current_block.blockhead, 'nonce3', 0) == 0):
                current_block = current_block.parentblock

            if not current_block or 'B0' in current_block.name:
                continue

            chain_optimizations = {}

            # 从该块的父块开始回溯，收集优化块的 y 值及其时间戳
            current_block = current_block.parentblock
            while current_block and 'B0' not in current_block.name:
                pouw_opt_y = getattr(current_block.blockhead, 'pouw_opt_y', 0)
                if pouw_opt_y != 0:
                    chain_optimizations[pouw_opt_y] = getattr(current_block.blockhead, 'timestamp', 0)
                    current_block = current_block.parentblock
                else:
                    # 遇到 pouw_opt_y == 0 的块，尝试以其时间戳作为基准零点
                    base_timestamp = None
                    nonce3 = getattr(current_block.blockhead, 'nonce3', 0)
                    if nonce3 != 0 and 'B0' not in current_block.name:
                        base_timestamp = getattr(current_block.blockhead, 'timestamp', 0)
                    if base_timestamp is not None:
                        for opt_y in list(chain_optimizations.keys()):
                            chain_optimizations[opt_y] -= base_timestamp
                        benchmark_y = getattr(current_block.blockhead, 'benchmark_y', 0)
                        chain_optimizations[benchmark_y] = 0
                        sorted_chain_optimizations = OrderedDict(
                            sorted(chain_optimizations.items(), key=lambda item: item[1]))
                        all_chains_optimizations.append(sorted_chain_optimizations)
                        print(f'has append vtchain {chain_index_iter}')
                        chain_optimizations = {}
                        current_block = current_block.parentblock
                    else:
                        # 无有效基准时间，继续回溯
                        current_block = current_block.parentblock

            # 如果回溯到创世块仍未写入一次记录，则以任务基准解作为起点
            if current_block and 'B0' in current_block.name:
                task_all = global_var.get_tasks()
                if task_all and 0 <= (chain_index_iter - 1) < len(task_all):
                    benchmark_x = task_all[chain_index_iter - 1]
                    benchmark_y = functions.f1(benchmark_x)
                    chain_optimizations[benchmark_y] = 0
                    sorted_chain_optimizations = OrderedDict(
                        sorted(chain_optimizations.items(), key=lambda item: item[1]))
                    all_chains_optimizations.append(sorted_chain_optimizations)
                    print(f'has append vtchain {chain_index_iter}')

        # 2) 统计任务完成时间（两次连续的 nonce3 != 0 之间的时差），并计算平均速度
        task_durations = []
        for vt_chain in self.global_vt_chains:
            current_block = vt_chain.get_last_block()
            # 从链尾向前扫描，将相邻两次纯 PoW（nonce3 != 0）出块的时间差视为一次“任务完成时间”
            last_end_time = None
            while current_block and 'B0' not in current_block.name:
                if getattr(current_block.blockhead, 'nonce3', 0) != 0:
                    end_time = getattr(current_block.blockhead, 'timestamp', 0)
                    if last_end_time is None:
                        last_end_time = end_time
                    else:
                        # 邻接两次纯 PoW 之间的间隔
                        duration = last_end_time - end_time
                        if duration >= 0:
                            task_durations.append(duration)
                        last_end_time = end_time
                    current_block = current_block.parentblock
                else:
                    current_block = current_block.parentblock

        # 写出优化过程与任务完成时间
        tasknum = global_var.get_task_num()
        yuzhi_change = global_var.get_yuzhi_change()
        yuzhi = global_var.get_yuzhi()

        # 记录每条链的优化效果（时间已相对化）
        file_path_opt = f'D:/chainximopt_{tasknum}_{yuzhi_change}_{yuzhi}.txt'
        with open(file_path_opt, 'a', encoding='utf-8') as f:
            for i, chain_optimization in enumerate(all_chains_optimizations):
                f.write(f"chain {i} OPT TIME: {dict(chain_optimization)}\n")

        # 记录任务完成时间序列

        file_path_time = f'D:/chainximopt_{tasknum}_time.txt'
        with open(file_path_time, 'a', encoding='utf-8') as f:
            f.write(f"solve time = {task_durations}\n")


        # 速度与相关性指标
        total_duration = sum(task_durations)
        valid_round_count = len(task_durations)
        average_duration = total_duration / valid_round_count if valid_round_count > 0 else 0
        average_speed = 1 / average_duration if average_duration > 0 else 0


        with open('D:/suduthre.txt', 'a', encoding='utf-8') as f:
            f.write(f"speed is {average_speed}\n")



        last_keys = [list(d.keys())[-1] for d in all_chains_optimizations if len(d) > 0]
        mean_task_durations = float(np.mean(task_durations)) if len(task_durations) > 0 else 0.0
        mean_last_keys = float(np.mean(last_keys)) if len(last_keys) > 0 else 0.0
        with open('D:/rguanxi.txt', 'a', encoding='utf-8') as f:
            f.write(f"mean_task_durations: {mean_task_durations}\nmean_last_keys: {mean_last_keys}\n")


        # 3) 生成图：
        # 3.1 任务完成时间直方图
        if len(task_durations) > 0:
            plt.figure(figsize=(8, 4.5))
            plt.hist(task_durations, bins=30, color='#4C72B0', alpha=0.85)
            plt.xlabel('Task completion time (rounds)')
            plt.ylabel('Frequency')
            plt.title('Distribution of task completion time')
            plt.grid(True, alpha=0.3)
            out_path = global_var.get_result_path() / 'task_completion_time_hist.svg'
            plt.savefig(out_path)
            if global_var.get_show_fig():
                plt.show()
            plt.close()


        # 3.2 优化过程阶梯图（每条链 y 随相对时间的阶梯变化）
        if len(all_chains_optimizations) > 0:
            plt.figure(figsize=(9.5, 5.0))
            for i, chain_optimization in enumerate(all_chains_optimizations):
                if len(chain_optimization) == 0:
                    continue
                times = list(chain_optimization.values())
                ys = list(chain_optimization.keys())
                # 为了阶梯效果更明显，确保从 0 开始
                if len(times) > 0 and times[0] != 0:
                    times = [0] + times
                    ys = [ys[0]] + ys
                plt.step(times, ys, where='post', label=f'chain {i}')
            plt.xlabel('Relative time (rounds)')
            plt.ylabel('Objective value y')
            plt.title('Optimization progress over time')
            plt.grid(True, alpha=0.3)
            plt.legend(ncol=2, fontsize=8)
            out_path2 = global_var.get_result_path() / 'optimization_progress_step.svg'
            plt.savefig(out_path2)
            if global_var.get_show_fig():
                plt.show()
            plt.close()


    def assess_common_prefix(self, type:str = 'cdf'):
        def assess_common_prefix_pdf_per_round(self:Environment, miner_local_chain_tip:list[Block], longest_honest_chain:Block):
            # Common Prefix Property
            # 以全局链为标准视角 与其他矿工的链视角对比

            # 每轮结束时，各个矿工的链与common prefix相差区块个数的分布
            cp_pdf = np.zeros((1, MAX_SUFFIX))

            cp = longest_honest_chain
            for i in self.honest_miner_ids:
                cp = common_prefix(cp, miner_local_chain_tip[i])
            len_cp = cp.get_height()
            for i in range(0, self.miner_num):
                len_suffix = miner_local_chain_tip[i].get_height() - len_cp
                if len_suffix >= 0 and len_suffix < MAX_SUFFIX:
                    cp_pdf[0, len_suffix] = cp_pdf[0, len_suffix] + 1
            return cp_pdf

        def assess_common_prefix_cdf_per_round(self:Environment, miner_local_chain_tip:list[Block], longest_honest_chain:Block):
            # 一种新的计算common prefix的方法，更加接近Bitcoin backbone
            # 每轮结束后，砍掉主链后
            cp_cdf_k = np.zeros((1, MAX_SUFFIX))
            cp_k = longest_honest_chain
            cp_stat = np.zeros((1, self.miner_num))
            for k in range(MAX_SUFFIX):
                # 当所有矿工的链都达标后，后面的都不用算了，降低计算复杂度
                if cp_k is None or np.sum(cp_stat) == len(self.honest_miner_ids):  
                    cp_cdf_k[0, k] += self.miner_num-self.adversary.get_adver_num()
                    continue
                cp_sum_k = 0
                for i in self.honest_miner_ids:
                    if cp_stat[0, i] == 1:
                        cp_sum_k += 1
                        continue
                    if cp_k == common_prefix(cp_k, miner_local_chain_tip[i]):
                        cp_stat[0, i] = 1
                        cp_sum_k += 1
                cp_cdf_k[0, k] += cp_sum_k
                cp_k = cp_k.parentblock
            return cp_cdf_k
        
        if type == 'pdf':
            assess_cp_per_round = assess_common_prefix_pdf_per_round
            cp_xdf_update = np.zeros((1, MAX_SUFFIX))
            cp_xdf_update[0, 0] = self.miner_num
        elif type == 'cdf':
            assess_cp_per_round = assess_common_prefix_cdf_per_round
            cp_xdf_update = np.ones((1, MAX_SUFFIX)) * len(self.honest_miner_ids)
        else:
            raise ValueError('type should be pdf or cdf')

        cp_xdf = np.zeros((1, MAX_SUFFIX))
        # 跟踪矿工本地链链尾
        miner_local_chain_tip = [miner.consensus.local_chain_prp.head for miner in self.miners]

        event_collector = self.collect_until_next_round() # 收集下一个存在事件的轮次上的所有事件
        previous_round = 0
        while current_events := next(event_collector, None):
            rounds_elapsed = current_events[0].switch_round - previous_round
            previous_round = current_events[0].switch_round
            cp_xdf += rounds_elapsed * cp_xdf_update # 重复利用上一个事件的结果
            
            for event in current_events:
                miner_local_chain_tip[event.subject] = self.miners[event.subject].consensus.local_chain_prp.search_block_by_hash(event.blockhash)
            honest_chain_tip = [tip for miner, tip in enumerate(miner_local_chain_tip) if miner in self.honest_miner_ids]
            longest_honest_chain = max(honest_chain_tip, key=lambda x: x.get_height())
            cp_xdf_update = assess_cp_per_round(self, miner_local_chain_tip, longest_honest_chain)
        return cp_xdf
        
    def collect_until_next_round(self):
        event_iter = iter(self.local_chain_tracker.chain_switch_events)
        current_round = self.local_chain_tracker.chain_switch_events[0].switch_round
        events = []
        try:
            while True:
                event:LocalChainTracker.ChainSwitchEvent = next(event_iter)
                if event.switch_round > current_round:
                    yield events
                    current_round = event.switch_round
                    events = []
                elif event.switch_round < current_round:
                    raise ValueError('Events are not stored incrementally, something is wrong')
                events.append(event)
        except StopIteration:
            yield events
            return

    def view(self, quantile) -> dict:
        # 展示一些仿真结果
        print('\n')
        # print("Global Tree Structure:", "")
        # self.global_chain.ShowStructure1()
        # print("End of Global Tree", "")

        # Evaluation Results
        honest_miners = filter(lambda x: x.miner_id in self.honest_miner_ids, self.miners)
        stats = self.global_chain_prp.CalculateStatistics(self.total_round, list(honest_miners), self.adversary.get_adver_ids(),
                                                          self.confirm_delay, self.dataitem_params, self.dataitem_validator, quantile,
                                                          self.adversary.get_eclipsed_ids(), self.local_chain_tracker.chain_switch_events)
        stats.update({'total_round':self.total_round})
        # Chain Growth Property
        growth = 0
        num_honest = 0
        for i in range(self.miner_num):
            if not self.miners[i]._isAdversary:
                growth = growth + chain_growth(self.miners[i].consensus.local_chain_prp)
                # 加上所有投票者链的增长
                for vt_chain in self.miners[i].consensus.local_chain_vt:
                    growth = growth + chain_growth(vt_chain)
                num_honest = num_honest + 1
        growth = growth / num_honest
        stats.update({
            'average_chain_growth_in_honest_miners\'_chain': growth
        })
        # Common Prefix Property
        if global_var.get_common_prefix_enable():
            cp_pdf = self.assess_common_prefix('pdf')
            cp_cdf_k = self.assess_common_prefix('cdf')
            timestamp_of_last_block = self.global_chain_prp.get_last_block().blockhead.timestamp
            stats.update({
                'common_prefix_pdf': cp_pdf/cp_pdf.sum(),
                'consistency_rate':cp_pdf[0,0]/(cp_pdf.sum()),
                'common_prefix_cdf_k': cp_cdf_k/(len(self.honest_miner_ids)*timestamp_of_last_block)
            })

        # Network Property
        stats.update({'block_propagation_times': {} })
        if not isinstance(self.network,network.SynchronousNetwork):
            self.network: network.StochPropNetwork
            ave_block_propagation_times = self.network.cal_block_propagation_times()
            stats.update({
                'block_propagation_times': ave_block_propagation_times
            })
        inv_count = 0
        sync_full_chain_count = 0
        send_data_count = 0
        for miner in self.miners:
            inv_count += miner._NIC.inv_count
            sync_full_chain_count += miner._NIC.sync_full_chain_count
            send_data_count += miner._NIC.send_data_count
        stats.update({
            'inv_count': inv_count,
            'sync_full_chain_count': sync_full_chain_count,
            'send_data_count': send_data_count
        })
        
        for k,v in stats.items():
            if type(v) is float:
                stats.update({k:round(v,8)})

        # show the results in the terminal
        # Chain Growth Property
        print('Chain Growth Property:')
        print(stats["num_of_generated_blocks"], "blocks are generated in",
              self.total_round, "rounds, in which", stats["num_of_stale_blocks"], "are stale blocks.")
        print("Average chain growth in honest miners' chain:", round(growth, 3))
        print("Number of Forks:", stats["num_of_forks"])
        print("Fork rate:", stats["fork_rate"])
        print("Stale rate:", stats["stale_rate"])
        print("Average block time (main chain):", stats["average_block_time_main"], "rounds/block")
        print("Average block time (total):", stats["average_block_time_total"], "rounds/block")
        print("Block throughput (main chain):", stats["block_throughput_main"], "blocks/round")
        if self.dataitem_params['dataitem_enable']:
            print("Throughput of valid dataitems:", stats['valid_dataitem_throughput'], "items/round")
            print("Throughput of valid dataitems in MB:", 
                stats['valid_dataitem_throughput']*self.dataitem_params['dataitem_size'], "MB/round")
            print("Input dataitem rate:", stats["input_dataitem_rate"], "items/round")
            print("Average dataitems per block:", stats["block_average_size"], "items/block")
        else:
            print("Throughput in MB (main chain):", stats["throughput_main_MB"], "MB/round")
            print("Block throughput (total):", stats["block_throughput_total"], "blocks/round")
            print("Throughput in MB (total):", stats["throughput_total_MB"], "MB/round")
        # Common Prefix Property
        if global_var.get_common_prefix_enable():
            print('Common Prefix Property:')
            print('The common prefix pdf:')
            print(stats["common_prefix_pdf"])
            print('Consistency rate:', stats["consistency_rate"])
            print('The common prefix cdf with respect to k:')
            print(stats["common_prefix_cdf_k"])
        print("")
        # Chain Quality Property
        print('Chain_Quality Property:', stats["chain_quality_property"])
        print('Ratio of blocks contributed by malicious players:', stats["ratio_of_blocks_contributed_by_malicious_players"])
        if self.dataitem_params['dataitem_enable']:
            print('Ratio of dataitems contributed by malicious players:', 1 - stats["valid_dataitem_rate"])
        # Attack Property
        if self.adversary.get_info():
            print('The simulation data of', self.adversary.get_attack_type_name() , 'is as follows', ':\n', self.adversary.get_info())
            stats.update(self.adversary.get_info())
            print('Double spending success times:', stats["double_spending_success_times"])
            
        # Network Property
        if not isinstance(self.network,network.SynchronousNetwork):
            print('Block propagation times:', ave_block_propagation_times)
        if isinstance(self.network,network.TopologyNetwork) or isinstance(self.network,network.AdHocNetwork):
            print('Count of INV interactions:', inv_count)
            print('Count of full chain synchronization:', sync_full_chain_count)
            print('Count of data sending:', send_data_count)
        return stats

    def view_and_write(self, quantile):
        # stats = self.view(quantile)
        
        self.global_chain_prp.printchain2txt()
        # 也打印投票者链的信息
        for i, vt_chain in enumerate(self.global_vt_chains):
            vt_chain.printchain2txt(f"global_vt_chain_{i}.txt")

        # save the results in the evaluation results.txt
        # RESULT_PATH = global_var.get_result_path()
        # with open(RESULT_PATH / 'evaluation results.txt', 'a+',  encoding='utf-8') as f:
        #     blocks_round = ['block_throughput_main', 'block_throughput_total']
        #     MB_round = ['throughput_main_MB', 'throughput_total_MB']
        #     rounds_block = ['average_block_time_main', 'average_block_time_total']

        #     for k,v in stats.items():
        #         if k in blocks_round:
        #             print(f'{k}: {v} blocks/round', file=f)
        #         elif k in MB_round:
        #             print(f'{k}: {v} MB/round', file=f)
        #         elif k in rounds_block:
        #             print(f'{k}: {v} rounds/block', file=f)
        #         else:
        #             print(f'{k}: {v}', file=f)

        # if global_var.get_compact_outputfile():
        #     return stats


        self.local_chain_tracker.dump_events(global_var.get_chain_data_path() / 'chain_dump.bin',
                                             [miner.consensus.local_chain_prp for miner in self.miners],
                                             self.honest_miner_ids)

        # show or save figures
        # self.global_chain.ShowStructure(self.miner_num)
        # block interval distribution
        self.miners[0].consensus.local_chain_prp.GetBlockIntervalDistribution()

        other_list = self.adversary.get_eclipsed_ids()
        
        # 使用全局变量中存储的投票计数（如果存在）
        vote_counts = global_var.get_vtcount()
        
        # 计算每个高度上投票最多的区块
        max_votes_per_height = {}
        for block in self.global_chain_prp:
            if block.height not in max_votes_per_height or \
                    vote_counts.get(block.name, 0) > vote_counts.get(max_votes_per_height[block.height].name, 0) or \
                    (vote_counts.get(block.name, 0) == vote_counts.get(max_votes_per_height[block.height].name, 0) and 
                     block.blockhash < max_votes_per_height[block.height].blockhash):
                max_votes_per_height[block.height] = block
        
        # 先绘制提议者链（包含投票计数信息）
        dot = self.global_chain_prp.ShowStructureWithGraphviz(dot=None, chain_id='prpchain', prpchain=None,
                                                              vote_counts=vote_counts,
                                                              max_votes_per_height=max_votes_per_height)
        
        # 然后绘制所有投票者链
        for i in range(global_var.get_vtnum()):
            if i < len(self.global_vt_chains):  # 确保索引不越界
                dot = self.global_vt_chains[i].ShowStructureWithGraphviz(dot=dot, chain_id=f'vtchain_{i}', prpchain=None)
        
        # 渲染最终图形
        if dot is not None:
            try:
                dot.render(directory=global_var.get_result_path() / "blockchain_visualization", format='svg',
                          filename='blockchain_structure', cleanup=True, view=global_var.get_show_fig())
            except Exception as e:
                print(f"绘图渲染失败: {e}")
                # 如果渲染失败，至少调用原来的方法作为备选
                self.global_chain_prp.ShowStructureWithGraphviz(other_list=other_list)
        if isinstance(self.network,network.TopologyNetwork):
            # 利用 isinstance 指定类型 方便调用类方法gen_routing_gragh_from_json()
            # self.network.save_rest_routing_process()
            self.network.gen_routing_gragh_from_json()
        # return stats
    
    def process_bar(self,process,total,t_0,unit='round/s'):
        bar_len = 50
        percent = (process)/total
        cplt = "■" * math.ceil(percent*bar_len)
        uncplt = "□" * (bar_len - math.ceil(percent*bar_len))
        time_len = time.time()-t_0+0.0000000001
        time_cost = time.gmtime(time_len)
        vel = process/(time_len)
        time_eval = time.gmtime(total/(vel+0.001))#Events: see events.log 
        print("\r{}{}  {:.5f}%  {}/{}  {:.2f} {}  {}:{}:{}>>{}:{}:{}  "\
        .format(cplt, uncplt, percent*100, process, total, vel, unit, time_cost.tm_hour, time_cost.tm_min, time_cost.tm_sec,\
            time_eval.tm_hour, time_eval.tm_min, time_eval.tm_sec),end="", flush=True)
        return vel

        
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    times_prob = {0:0, 0.03: 0.944, 0.05: 1.804, 0.08: 2.767, 0.1: 3.308, 0.2: 5.421, 
              0.4: 8.797, 0.5: 10.396, 0.6: 12.071, 0.7: 13.956, 0.8: 16.217, 0.9: 19.377, 
              0.93: 20.777, 0.95: 21.971, 0.98: 24.683, 1.0: 29.027}
    times_vec ={0:0, 0.1: 1, 0.2: 2, 0.3:3, 0.4: 4, 0.5: 5.0, 0.6: 6.0, 0.7: 7.0, 0.8: 8.0, 0.9: 9.0, 1.0: 10} 


    times_tp = {0:0, 0.03: 9.297, 0.05: 12.257, 0.08: 15.889, 0.1: 17.185, 0.2: 21.459, 0.4: 26.592, 0.5: 28.493, 0.6: 30.343, 0.7: 32.232, 0.8: 34.231, 0.9: 36.993, 0.93: 38.194, 0.95: 39.256, 0.98: 41.946, 1.0: 46.706}
    times_outage = {0:0, 0.03: 9.627, 0.05: 12.988, 0.08: 16.45, 0.1: 17.94, 0.2: 22.193, 0.4: 27.405, 0.5: 29.353, 0.6: 31.262, 0.7: 33.265, 0.8: 35.496, 0.9: 38.663, 0.93: 40.298, 0.95: 41.889, 0.98: 45.681, 1.0: 52.662}
    times_wo_parition = {0:0,  0.05: 21.713, 0.08: 27.103, 0.1: 29.326, 0.2: 42.763, 0.4: 61.751, 0.5: 68.969, 0.6: 80.483, 0.8: 92.329, 0.9: 103.141, 0.93: 107.383, 0.95: 110.757, 0.98: 119.123, 1.0: 145.109}
    times_dynamic ={0:0, 0.03: 18.237, 0.05: 24.153, 0.08: 30.627, 0.1: 34.757, 0.2: 56.545, 0.4: 81.044, 0.5: 84.115, 0.6: 103.68, 0.7: 128.212, 0.8: 147.1, 0.9: 196.187, 0.93: 238.383, 0.95: 360.152, 0.98: 2226.653, 1.0: 3848.5}
    rcv_rates = list(times_tp.keys())
    t = list(times_tp.values())
    t = [tt/t[-1] for tt in t]
    plt.plot(t,rcv_rates,"--o", label= "Topology Network")

    rcv_rates = list(times_outage.keys())
    t = list(times_outage.values())
    t = [tt/t[-1] for tt in t]
    plt.plot(t,rcv_rates,"--o", label= "Topology Network + Link Outage")

    rcv_rates = list(times_wo_parition.keys())
    t = list(times_wo_parition.values())
    t = [tt/t[-1] for tt in t]
    plt.plot(t,rcv_rates,"--o", label= "Topology Network + Dynamic")

    rcv_rates = list(times_dynamic.keys())
    t = list(times_dynamic.values())
    t = [tt/t[-1] for tt in t]
    plt.plot(t,rcv_rates,"--o", label= "Topology Network + Dynamic + 3Patitions")

    rcv_rates = list(times_prob.keys())
    t = list(times_prob.values())
    t = [tt/t[-1] for tt in t]
    plt.plot(t,rcv_rates,"--^", label= "Stochastic Propagation Network")
    rcv_rates = list(times_vec.keys())
    t = list(times_vec.values())
    t = [tt/t[-1] for tt in t]
    plt.plot(t,rcv_rates,"--*", label= "Deterministic Propagation Network")
    plt.xlabel("Normalized number of rounds passed")
    plt.ylabel("Ratio of received miners")
    plt.legend()
    plt.grid()
    plt.show()