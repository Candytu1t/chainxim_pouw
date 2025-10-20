import math
import struct
import time
import json
from typing import List, Tuple

import functions
from functions import f1, ff1, merkle_root

import data
import numpy as np
import global_var
from functions import BYTE_ORDER, INT_LEN, HASH_LEN, hash_bytes

from .consensus_abc import Consensus




class PoUW(Consensus):
    """Proof of Useful Work (PoUW) 共识机制
    
    基于旧项目chain-xim-master-kkt中的PoW实现，移植而来的PoUW共识算法。
    支持多链架构（proposer链和多个voter链）以及多阶段挖矿过程。
    """

    # 为不同类型的链设置不同的创世区块参数
    genesis_blockheadextra_prp = {}
    genesis_blockextra_prp = {}
    genesis_blockheadextra_vt = {}
    genesis_blockextra_vt = {}

    class BlockHead(Consensus.BlockHead):
        '''适用于PoUW共识协议的区块头'''
        __slots__ = ['target', 'target2', 'target3', 'nonce', 'nonce2', 'nonce3', 'gradient', 'benchmark_x', 'benchmark_y']
        
        def __init__(self, preblock: Consensus.Block = None, timestamp=0, content=b'', miner_id=-1,
                     target=(2**(8*HASH_LEN) - 1).to_bytes(HASH_LEN, BYTE_ORDER),
                     target2=(2 ** (8 * HASH_LEN) - 1).to_bytes(HASH_LEN, BYTE_ORDER),
                     target3=(2 ** (8 * HASH_LEN) - 1).to_bytes(HASH_LEN, BYTE_ORDER),
                     nonce=0, nonce2=0, nonce3=0, parentMT_root='', contentMT_root='', 
                     pouw_opt_x=None, pouw_opt_y=0, gradient=0.0, benchmark_x=None, benchmark_y=0):
            if pouw_opt_x is None:
                pouw_opt_x = []
            if benchmark_x is None:
                benchmark_x = []
            # 调用父类构造函数 - 只传递父类需要的参数
            super().__init__(preblock, timestamp, content, miner_id)
            # 手动设置PoUW特有的属性
            self.parentMT_root = parentMT_root
            self.contentMT_root = contentMT_root
            self.pouw_opt_x = pouw_opt_x
            self.pouw_opt_y = pouw_opt_y
            self.target = target  # 难度目标1
            self.target2 = target2  # 难度目标2
            self.target3 = target3  # 难度目标3
            self.nonce = nonce  # 随机数1
            self.nonce2 = nonce2  # 随机数2
            self.nonce3 = nonce3  # 随机数3
            self.gradient = gradient  # 梯度信息
            self.benchmark_x = benchmark_x  # 基准解
            self.benchmark_y = benchmark_y  # 基准解的函数值

        def calculate_blockhash(self) -> bytes:
            """计算区块哈希（严格复现挖矿阶段对 content 的处理：尝试 int 转换，否则用 0）"""
            # 与 mining_consensus 中一致：content 尝试用 int(x)，失败则回退为 0
            try:
                x_int = int(self.content)
            except Exception:
                x_int = 0
            content_bytes = x_int.to_bytes(INT_LEN, BYTE_ORDER, signed=True)

            intermediate_hasher = hash_bytes(
                self.miner.to_bytes(INT_LEN, BYTE_ORDER, signed=True) +
                self.parentMT_root.encode() +
                self.contentMT_root.encode() +
                content_bytes)
            hasher = intermediate_hasher.copy()
            hasher.update(self.nonce.to_bytes(INT_LEN, BYTE_ORDER))
            currenthash = hasher.digest()
            return currenthash

    class Block(Consensus.Block):
        '''适用于PoUW共识协议的区块'''
        __slots__ = []
        
        def __init__(self, *args, **kwargs):
            # 处理位置参数的兼容性
            if len(args) >= 1 and not isinstance(args[0], str):
                # 如果第一个参数不是字符串，说明是旧的调用方式 (blockhead, preblock, ...)
                blockhead = args[0]
                preblock = args[1] if len(args) > 1 else kwargs.get('preblock', None)
                isadversary = args[2] if len(args) > 2 else kwargs.get('isadversary', False)
                blocksize_MB = args[3] if len(args) > 3 else kwargs.get('blocksize_MB', 2)
                
                # 调用父类构造函数
                super().__init__(blockhead, preblock, isadversary, blocksize_MB)
                
                # 设置PoUW特有属性
                self.index = kwargs.get('index', 0)
                self.vtonprp = kwargs.get('vtonprp', [])
                self.txpool = kwargs.get('txpool', [])
                self.prppool = kwargs.get('prppool', [])
            else:
                # 新的调用方式 (name=..., blockhead=..., ...)
                name = args[0] if len(args) > 0 else kwargs.get('name', None)
                blockhead = args[1] if len(args) > 1 else kwargs.get('blockhead', None)
                preblock = args[2] if len(args) > 2 else kwargs.get('preblock', None)
                isadversary = args[3] if len(args) > 3 else kwargs.get('isadversary', False)
                blocksize_MB = args[4] if len(args) > 4 else kwargs.get('blocksize_MB', 2)
                
                # 如果没有提供name，使用默认的区块编号生成方式
                if name is None:
                    import global_var
                    block_number = global_var.get_block_number() if preblock else 0
                    name = f"B{block_number}"
                
                # 设置高度和创世标志
                height = preblock.height+1 if preblock else 0
                is_genesis = False if preblock else True
                
                # 直接调用data.Block构造函数
                data.Block.__init__(self, name, blockhead, height, isadversary, is_genesis, blocksize_MB)
                
                # 设置PoUW特有属性
                self.index = kwargs.get('index', 0)
                self.vtonprp = kwargs.get('vtonprp', [])
                self.txpool = kwargs.get('txpool', [])
                self.prppool = kwargs.get('prppool', [])

    def __init__(self, miner_id, consensus_params: dict):
        super().__init__(miner_id=miner_id)
        
        # 矿工ID相关
        self.miner_id = miner_id
        
        # 基础计数器
        self.ctr = 0  # 计数器1
        self.ctr2 = 0  # 计数器2
        self.ctr3 = 0  # 计数器3
        
        # 难度目标设置
        self.target = bytes.fromhex(consensus_params['target'])
        self.target2 = bytes.fromhex(consensus_params.get('target2', consensus_params['target']))
        self.target3 = bytes.fromhex(consensus_params.get('target3', consensus_params['target']))
        
        # Merkle树根
        self.parentroot = None
        self.contentroot = None
        self.bh = None
        
        # 投票相关
        self.voter_parent_balance = None
        self.gradient = 0
        
        # 初始化多链结构
        self.local_chain_prp = data.Chain(miner_id)  # proposer链
        self.local_chain_vt = [data.Chain(miner_id) for _ in range(global_var.get_vtnum())]  # voter链
        
        # 创建创世区块
        self.create_genesis_block(self.local_chain_prp, self.genesis_blockheadextra_prp, self.genesis_blockextra_prp, 'prpchain')
        
        for i, vtchain in enumerate(self.local_chain_vt):
            self.create_genesis_block(vtchain, self.genesis_blockheadextra_vt, self.genesis_blockextra_vt, f'vtchain_{i}')
        
        # 消息接收队列
        self._receive_tape_tx = []  # 交易消息
        self._receive_tape_prp = []  # 提案消息
        self._receive_tape_vt = []  # 投票消息
        
        # 投票相关
        self.votesOnPrpBks = [[] for _ in range(global_var.get_vtnum())]
        self._block_buffer = {}  # 区块缓存
        
        # 未引用的交易和提案
        self.unref_tx = []
        self.unref_prp = []
        
        # PoUW优化相关
        dimension = global_var.get_N()
        fanwei = global_var.get_fanwei()
        self.pouw_opt_x = [np.random.uniform(-fanwei, fanwei, dimension).tolist() for _ in range(global_var.get_vtnum())]
        # self.pouw_opt_x = [[1.05, -1.71] for _ in range(global_var.get_vtnum())]
        self.pouw_opt_y = [functions.f1(x) for x in self.pouw_opt_x]
        self.pouw_opt_stage = 1  # 当前优化阶段
        self.segment_index = None  # 当前段索引
        
        # 优化过程记录
        self.initial_point = []
        self.final_point = []
        self.fail_count = 0
        self.opt_start = 0
        self.pow_start = 0
        
        # 算力分布设置
        if consensus_params['q_distr'] == 'equal':
            self.q = consensus_params['q_ave']
        else:
            q_distr = eval(consensus_params['q_distr'])
            if isinstance(q_distr, list):
                self.q = q_distr[miner_id]
            else:
                raise ValueError("q_distr should be a list or the string 'equal'")
        
        # 初始化初始点
        number = global_var.get_number()
        if number < len(self.pouw_opt_x):
            solution = self.pouw_opt_x[number]
            self.initial_point = [solution[:]]
        else:
            self.initial_point = []
        
        self.last_point_time = 0

    def setparam(self, **consensus_params):
        '''设置PoUW参数'''
        self.target = bytes.fromhex(consensus_params.get('target', self.target.hex()))
        self.target2 = bytes.fromhex(consensus_params.get('target2', self.target2.hex()))
        self.target3 = bytes.fromhex(consensus_params.get('target3', self.target3.hex()))
        self.q = consensus_params.get('q', self.q)

    def remove_unreftx(self, item):
        """移除未引用的交易"""
        if item in self.unref_tx:
            self.unref_tx.remove(item)

    def remove_unrefprp(self, item):
        """移除未引用的提案"""
        if item in self.unref_prp:
            self.unref_prp.remove(item)

    def distance_limit(self, x1, x2):
        """计算两点间的欧几里得距离"""
        if len(x1) != len(x2):
            return 100  # 长度不匹配时返回大值
        ds = 0
        for i in range(0, len(x1)):
            ds += (x1[i] - x2[i]) * (x1[i] - x2[i])
        return math.sqrt(ds)

    def get_voter_parent_balancing(self, minority_proposer, chainid):
        """获取平衡投票的父块"""
        initial_block = self.local_chain_vt[chainid].last_block
        while initial_block and 'B0' not in initial_block.name:
            initial_block = initial_block.parentblock
        
        if 'B0' in self.local_chain_vt[chainid].last_block.name:
            return initial_block
        
        voter_root = None
        for block in self.local_chain_vt[chainid].block_set.values():
            if block.height == 1:
                for vtblock in block.vtonprp:
                    if minority_proposer.name == vtblock.name:
                        voter_root = block
                        break
        
        if voter_root:
            search_list = [voter_root]
            highest_voter = voter_root
            while search_list:
                voter_tmp = search_list.pop(0)
                search_list.extend(voter_tmp.next)
                if voter_tmp.height > highest_voter.height:
                    highest_voter = voter_tmp
            return highest_voter
        else:
            return initial_block

    def get_last_proposer_reference(self, last_voter_block):
        """获取上一个引用过的提案块"""
        while last_voter_block is not None and last_voter_block.vtonprp == []:
            last_voter_block = last_voter_block.parentblock
        
        last_proposer_height = 0
        last_proposer_block = None
        if last_voter_block is not None:
            for block_i in last_voter_block.vtonprp:
                if block_i.height > last_proposer_height:
                    last_proposer_height = block_i.height
                    last_proposer_block = block_i
        
        return last_proposer_block, last_proposer_height

    def mining_consensus(self, miner_id, isadversary, x, round, tasks=None):
        '''PoUW挖矿算法
        
        实现了多阶段的PoUW挖矿过程：
        阶段1：segment选择
        阶段2：优化计算
        阶段3：nonce2计算
        阶段4：纯PoW计算
        
        Args:
            miner_id: 矿工ID
            isadversary: 是否为对手
            x: 区块内容
            round: 当前轮次
            tasks: 优化任务列表
            
        Returns:
            tuple: (挖出的区块列表, 挖矿成功标识)
        '''
        blocksnew = []
        pow_success = False
        
        # 获取全局参数
        yuzhi = global_var.get_yuzhi()
        kkt_yuzhi = global_var.get_kkt_yuzhi()
        repeat_times = global_var.get_repeat_time()
        dimension = global_var.get_N()
        number = global_var.get_number()
        fanwei = global_var.get_fanwei()
        
        # 创建基础调试日志文件
        import datetime

        
        # 执行挖矿循环
        for _ in range(int(self.q)):
            prpContent = []
            txContent = []
            vtprehashes = [0] * global_var.get_vtnum()
            vtContent = self.votesOnPrpBks
            
            b_last_vt = [None for _ in range(global_var.get_vtnum())]
            target_int = int.from_bytes(self.target, 'big')
            segment_range = target_int // (global_var.get_vtnum() + 2)
            
            
            # 获取各链的最后一个块
            b_last_prp = self.local_chain_prp.get_last_block()
            prpprehash = b_last_prp.blockhash
            txprehash = prpprehash
            
            for vt_index, vt_chain in enumerate(self.local_chain_vt):
                b_last_vt[vt_index] = vt_chain.get_last_block()
                vtprehashes[vt_index] = b_last_vt[vt_index].blockhash
            
            # 阶段1：segment选择
            if self.pouw_opt_stage == 1:
                # 准备内容
                for r_tape_tx in self.unref_tx:
                    txContent.append(r_tape_tx)
                for r_tape_prp in self.unref_prp:
                    prpContent.append(r_tape_prp)
                
                # 计算Merkle树根
                parentMT = [txprehash, prpprehash] + vtprehashes
                parentMT_root = merkle_root(parentMT)
                contentMT = [txContent] + [prpContent] + [vtContent]
                contentMT_root = merkle_root(contentMT)
                
                # 准备哈希计算
                miner_id_int = miner_id
                
                # 统一使用整数content，保持与旧版实现一致
                # x应为整数，若不是则回退为0
                try:
                    x_int = int(x)
                except Exception:
                    x_int = 0
                x_bytes = x_int.to_bytes(INT_LEN, BYTE_ORDER, signed=True)
                
                intermediate_hasher = hash_bytes(
                    miner_id_int.to_bytes(INT_LEN, BYTE_ORDER, signed=True) +
                    parentMT_root.encode() +
                    contentMT_root.encode() + 
                    x_bytes
                )
                
                
                # 尝试找到合适的哈希
                i = 0
                while i < 1:
                    i += 1
                    self.ctr += 1
                    hasher = intermediate_hasher.copy()
                    hasher.update(self.ctr.to_bytes(INT_LEN, BYTE_ORDER))
                    currenthash = hasher.digest()
                    currenthash_int = int.from_bytes(currenthash, 'big')
                    
                    if currenthash_int < target_int:
                        segment_index = currenthash_int // segment_range
                        print(f"miner {miner_id}:{segment_index}")
                        
                        # 根据segment_index处理不同类型的区块
                        if segment_index == 0:  # 交易区块
                            if not isadversary and global_var.get_attack_execute_type() != 'BalanceAttack':
                                pow_success = True
                                blockhead = self.BlockHead(None, round, x, miner_id,
                                                         self.target, self.target2, self.target3,
                                                         self.ctr, self.ctr2, 0, parentMT_root, contentMT_root)
                                global_var.rec_tx()
                                tx = global_var.get_txnum()
                                block_name = f'TX{tx}'
                                blocknew = self.Block(name=block_name, blockhead=blockhead, preblock=None, isadversary=isadversary, 
                                                    blocksize_MB=global_var.get_blocksize(), index=segment_index)
                                


                                
                                self.unref_tx.append(blocknew.name)
                                blocksnew.append(blocknew)
                        
                        elif segment_index == 1:  # 提案区块
                            if not isadversary and global_var.get_attack_execute_type() != 'BalanceAttack':
                                pow_success = True
                                blockhead = self.BlockHead(b_last_prp, round, x, miner_id,
                                                         self.target, self.target2, self.target3,
                                                         self.ctr, self.ctr2, 0, parentMT_root, contentMT_root)
                                blocknew = self.Block(blockhead=blockhead, preblock=b_last_prp, isadversary=isadversary, 
                                                    blocksize_MB=global_var.get_blocksize(), index=segment_index, 
                                                    vtonprp=[], txpool=txContent, prppool=prpContent)
                                

                                
                                print(f"wa chu le proposer block : {blocknew.name}")

                                
                                # 为新提案块投票
                                for i, vt_chain in enumerate(self.local_chain_vt):
                                    if not vt_chain.has_voted(blocknew.height):
                                        vt_chain.vote(blocknew.height, blocknew)
                                        self.votesOnPrpBks[i].append(blocknew)
                                
                                self.unref_tx = []
                                self.unref_prp = []
                                blocksnew.append(blocknew)
                        
                        elif segment_index < (global_var.get_vtnum() + 2):  # 投票区块

                            
                            # 处理平衡攻击
                            if global_var.get_attack_execute_type() == 'BalanceAttack' and isadversary:
                                minority_block = global_var.get_minority_block()
                                self.voter_parent_balance = self.get_voter_parent_balancing(minority_block, segment_index - 2)
                            
                            # 设置挖矿开始时间
                            if segment_index - 2 == number:
                                global_var.set_miner_in(round, miner_id)
                                self.opt_start = round
                            
                            self.fail_count = 0
                            
                            # 获取上一个解决方案
                            if b_last_vt[segment_index - 2].blockhead.pouw_opt_x != []:
                                last_solution = b_last_vt[segment_index - 2].blockhead.pouw_opt_x
                                opt_initial = functions.f1(last_solution)
                            else:
                                last_solution = tasks[segment_index - 2]
                                opt_initial = functions.f1(last_solution)
                            print(f'miner {miner_id}get task:{last_solution} and ini:{opt_initial}')
                            
                            # 计算梯度并检查KKT条件
                            gradient = functions.ff1(self.pouw_opt_x[segment_index - 2])
                            grad_norm = math.sqrt(sum(g ** 2 for g in gradient))
                            distance = self.distance_limit(last_solution, self.pouw_opt_x[segment_index - 2])
                            
                            
                            # 判断是否满足直接出块条件
                            if (self.pouw_opt_y[segment_index - 2] < opt_initial - yuzhi and 
                                self.pouw_opt_x[segment_index - 2] != [] and 
                                grad_norm < kkt_yuzhi and distance > 1):
                                self.pouw_opt_stage = 3  # 直接进入阶段3
                                

                                
                                if segment_index - 2 == number:
                                    global_var.set_dis_suc_pure(self.initial_point)
                                    global_var.set_dis_suc_pure_minerid(miner_id)
                                    global_var.set_dis_suc_round_pure(round)
                                    global_var.add_opt_result_pure(self.pouw_opt_y[segment_index - 2])
                                    self.pow_start = round
                            else:
                                self.pouw_opt_stage = 2  # 进入优化阶段
                                

                            
                            self.bh = currenthash
                            self.segment_index = segment_index
                            self.parentroot = parentMT_root
                            self.contentroot = contentMT_root
                            break
                    else:
                        pass
            
            # 阶段2：优化计算
            elif self.pouw_opt_stage == 2:
                segment_index = self.segment_index  # 使用保存的segment_index
                if segment_index - 2 == number:
                    if self.initial_point == []:
                        self.initial_point = [self.pouw_opt_x[self.segment_index - 2][:]]
                        self.last_point_time = round
                
                # 获取优化参数
                learning_rate = 1
                beta = 0.333
                sigma = 0.1
                max_adjustments = 20
                
                # 获取基准解
                if b_last_vt[self.segment_index - 2].blockhead.pouw_opt_x != []:
                    last_solution = b_last_vt[self.segment_index - 2].blockhead.pouw_opt_x
                    opt_initial = functions.f1(last_solution)
                else:
                    tasks = global_var.get_tasks()
                    if tasks and self.segment_index - 2 < len(tasks):
                        last_solution = tasks[self.segment_index - 2]
                        opt_initial = functions.f1(last_solution)
                    else:
                        last_solution = self.pouw_opt_x[self.segment_index - 2]
                        opt_initial = functions.f1(last_solution)
                
                # 执行优化
                panduan = 0
                for _ in range(1):
                    opt_before = self.pouw_opt_y[self.segment_index - 2]
                    gradient = functions.ff1(self.pouw_opt_x[self.segment_index - 2])
                    adjustment_count = 0
                    
                    # Armijo线搜索
                    while True:
                        new_solution = [xi - learning_rate * gi for xi, gi in
                                      zip(self.pouw_opt_x[self.segment_index - 2], gradient)]
                        f_x_new = functions.f1(new_solution)
                        f_x = self.pouw_opt_y[self.segment_index - 2]
                        
                        if f_x_new <= f_x - sigma * learning_rate * sum(g ** 2 for g in gradient):
                            self.pouw_opt_x[self.segment_index - 2] = new_solution
                            self.pouw_opt_y[self.segment_index - 2] = f_x_new
                            print(f'miner {miner_id} opt_after_1: {self.pouw_opt_y[self.segment_index - 2]} x:{self.pouw_opt_x[self.segment_index - 2]} and opt_before_1: {f_x} x:{self.pouw_opt_x[self.segment_index - 2]}')
                            panduan = 0
                            break
                        
                        learning_rate *= beta
                        adjustment_count += 1
                        
                        if adjustment_count >= max_adjustments:
                            self.pouw_opt_x[self.segment_index - 2] = new_solution
                            self.pouw_opt_y[self.segment_index - 2] = f_x_new
                            panduan = 1
                            print(f'miner {miner_id} panduan: {panduan} because adjustments')
                            break
                    
                    opt_after = self.pouw_opt_y[self.segment_index - 2]
                    print(f'miner {miner_id} opt_after: {opt_after} x:{self.pouw_opt_x[self.segment_index - 2]} and opt_before: {opt_before} x:{last_solution}')
                    
                    # 检查优化是否成功
                    if panduan == 0:
                        if opt_after < opt_before and all(-fanwei < self.pouw_opt_x[self.segment_index - 2][i] < fanwei 
                                                        for i in range(len(self.pouw_opt_x[self.segment_index - 2]))):
                            panduan = 0
                            print(f'miner {miner_id} panduan: {panduan} in 1')
                            break
                        else:
                            print(f'miner {miner_id} panduan: {panduan} because opt_after: {opt_after} and opt_before: {opt_before}')
                            panduan = 1
                            break
                    else:
                        panduan = 1
                        print(f'miner {miner_id} panduan: {panduan} because opt_after: {opt_after} and opt_before: {opt_before}')
                        break
                
                # 记录优化过程
                if self.segment_index - 2 == number:
                    if self.pouw_opt_x[self.segment_index - 2] != []:
                        solution = self.pouw_opt_x[self.segment_index - 2]
                        self.initial_point.append(solution[:])
                        self.last_point_time = round
                
                # 判断优化结果
                if panduan == 0:
                    # 根据优化方法选择不同的判断条件
                    optimization_method = global_var.get_optimization_method()
                    
                    if optimization_method == 'threshold':
                        # ======= 阈值方法处理逻辑 =======
                        height = b_last_vt[self.segment_index - 2].height
                        yuzhi = global_var.calculate_dynamic_threshold(height)
                        distance = self.distance_limit(last_solution, self.pouw_opt_x[self.segment_index - 2])
                        if 'B0' in b_last_vt[self.segment_index - 2].name:
                            distance = 100
                        
                        improvement = opt_initial - self.pouw_opt_y[self.segment_index - 2]  # 改善量（旧解 - 新解）
                        can_proceed = (improvement > yuzhi)
                        print(f'miner {miner_id} can_proceed: {can_proceed} and improvement: {improvement} and yuzhi: {yuzhi} and distance: {distance}')
                        
                        if can_proceed:
                            # 阈值方法成功，进入阶段3
                            self.pouw_opt_stage = 3
                            if self.segment_index - 2 == number:
                                opt_time = self.opt_start
                                use_time = round - opt_time
                                global_var.add_time_opt_suc(use_time)
                                self.pow_start = round
                                global_var.set_dis_suc_pure(self.initial_point)
                                global_var.set_dis_suc_pure_minerid(miner_id)
                                global_var.set_dis_suc_round_pure(round)
                                global_var.add_opt_result_pure(self.pouw_opt_y[self.segment_index - 2])
                        else:
                            # 阈值方法失败处理
                            print(f'miner {miner_id} threshold method failed')
                            if self.segment_index - 2 == number:
                                global_var.set_dis(self.initial_point)
                                global_var.set_dis_minerid(miner_id)
                                global_var.set_dis_round(round)
                                self.initial_point = []
                            
                            self.fail_count += 1
                            if self.fail_count >= repeat_times:
                                self.fail_count = 0
                                self.pouw_opt_stage = 4  # 进入纯PoW阶段
                                if self.segment_index - 2 == number:
                                    opt_start = self.opt_start
                                    opt_time = round - opt_start
                                    global_var.add_time_opt(opt_time)
                                    self.opt_start = 0
                                    self.pow_start = round
                            else:
                                # 重新初始化
                                self.pouw_opt_x[self.segment_index - 2] = np.random.uniform(-fanwei, fanwei, dimension).tolist()
                                self.pouw_opt_y[self.segment_index - 2] = functions.f1(self.pouw_opt_x[self.segment_index - 2])
                                if self.segment_index - 2 == number:
                                    self.initial_point = []
                                    new_solution = self.pouw_opt_x[self.segment_index - 2]
                                    self.initial_point = [new_solution[:]]
                                    self.last_point_time = round
                    
                    else:  # KKT方法
                        # ======= KKT方法处理逻辑 =======
                        gradient = functions.ff1(self.pouw_opt_x[self.segment_index - 2])
                        grad_norm = math.sqrt(sum(g ** 2 for g in gradient))
                        
                        if grad_norm < kkt_yuzhi:
                            # 满足KKT条件，才判断是否可以进入下一阶段
                            distance = self.distance_limit(last_solution, self.pouw_opt_x[self.segment_index - 2])
                            if 'B0' in b_last_vt[self.segment_index - 2].name:
                                distance = 100
                            improvement = opt_initial - self.pouw_opt_y[self.segment_index - 2]
                            
                            if (improvement > yuzhi) and (distance > 1):
                                # KKT方法成功，进入阶段3
                                self.pouw_opt_stage = 3
                                if self.segment_index - 2 == number:
                                    opt_time = self.opt_start
                                    use_time = round - opt_time
                                    global_var.add_time_opt_suc(use_time)
                                    self.pow_start = round
                                    global_var.set_dis_suc_pure(self.initial_point)
                                    global_var.set_dis_suc_pure_minerid(miner_id)
                                    global_var.set_dis_suc_round_pure(round)
                                    global_var.add_opt_result_pure(self.pouw_opt_y[self.segment_index - 2])
                                print(f'miner {miner_id} KKT method succeeded: improvement={improvement}, yuzhi={yuzhi}, distance={distance}')
                            else:
                                # 满足KKT条件但不满足阈值/距离条件，认为失败
                                print(f'miner {miner_id} KKT method failed: improvement={improvement}, yuzhi={yuzhi}, distance={distance}')
                                if self.segment_index - 2 == number:
                                    global_var.set_dis(self.initial_point)
                                    global_var.set_dis_minerid(miner_id)
                                    global_var.set_dis_round(round)
                                    self.initial_point = []
                                
                                self.fail_count += 1
                                if self.fail_count >= repeat_times:
                                    self.fail_count = 0
                                    self.pouw_opt_stage = 4  # 进入纯PoW阶段
                                    if self.segment_index - 2 == number:
                                        opt_start = self.opt_start
                                        opt_time = round - opt_start
                                        global_var.add_time_opt(opt_time)
                                        self.opt_start = 0
                                        self.pow_start = round
                                else:
                                    # 重新初始化
                                    self.pouw_opt_x[self.segment_index - 2] = np.random.uniform(-fanwei, fanwei, dimension).tolist()
                                    self.pouw_opt_y[self.segment_index - 2] = functions.f1(self.pouw_opt_x[self.segment_index - 2])
                                    if self.segment_index - 2 == number:
                                        self.initial_point = []
                                        new_solution = self.pouw_opt_x[self.segment_index - 2]
                                        self.initial_point = [new_solution[:]]
                                        self.last_point_time = round
                        else:
                            # 不满足KKT条件，继续优化（不认为失败，继续下一轮优化）
                            print(f'miner {miner_id} KKT condition not met: grad_norm={grad_norm}, kkt_yuzhi={kkt_yuzhi}, continue optimizing')
                            # 注意：这里不增加fail_count，因为KKT方法需要等到收敛才判断是否失败
                else:
                    # 优化失败处理
                    
                    if self.segment_index - 2 == number:
                        global_var.set_dis(self.initial_point)
                        global_var.set_dis_minerid(miner_id)
                        global_var.set_dis_round(round)
                        self.initial_point = []
                    
                    self.fail_count += 1

                    if self.fail_count >= repeat_times:

                        self.fail_count = 0
                        self.pouw_opt_stage = 4  # 进入纯PoW阶段
                        if self.segment_index - 2 == number:
                            opt_start = self.opt_start
                            opt_time = round - opt_start
                            global_var.add_time_opt(opt_time)
                            self.opt_start = 0
                            self.pow_start = round
                    else:
                        # 重新初始化
                        self.pouw_opt_x[self.segment_index - 2] = np.random.uniform(-fanwei, fanwei, dimension).tolist()
                        self.pouw_opt_y[self.segment_index - 2] = functions.f1(self.pouw_opt_x[self.segment_index - 2])
                        if self.segment_index - 2 == number:
                            self.initial_point = []
                            new_solution = self.pouw_opt_x[self.segment_index - 2]
                            self.initial_point = [new_solution[:]]
                            self.last_point_time = round
            
            # 阶段3：nonce2计算（有用工作证明）
            elif self.pouw_opt_stage == 3:
                posthash_success = False
                target_int_2 = int.from_bytes(self.target2, 'big')
                currenthash = self.bh
                
                for _ in range(1):
                    posthash = hash_bytes(
                        currenthash + 
                        struct.pack(f'{len(self.pouw_opt_x[self.segment_index - 2])}d', *self.pouw_opt_x[self.segment_index - 2]) + 
                        self.ctr2.to_bytes(INT_LEN, BYTE_ORDER)
                    ).digest()
                    
                    if int.from_bytes(posthash, 'big') < target_int_2:
                        posthash_success = True
                        break
                    self.ctr2 += 1
                
                if posthash_success:
                    
                    pow_success = True
                    if not isadversary or (isadversary and global_var.get_attack_execute_type() != 'BalanceAttack'):

                        
                        blockhead = self.BlockHead(
                            b_last_vt[self.segment_index - 2], round, x, miner_id,
                            self.target, self.target2, self.target3, self.ctr, self.ctr2, 0,
                            self.parentroot, self.contentroot,
                            self.pouw_opt_x[self.segment_index - 2], self.pouw_opt_y[self.segment_index - 2]
                        )
                        blocknew = self.Block(
                            blockhead=blockhead, preblock=b_last_vt[self.segment_index - 2], 
                            isadversary=isadversary, blocksize_MB=global_var.get_blocksize(), 
                            index=self.segment_index, vtonprp=vtContent[self.segment_index - 2]
                        )
                        

                        

                        

                    elif global_var.get_attack_execute_type() == 'BalanceAttack':
                        voter_parent = self.voter_parent_balance
                        blockhead = self.BlockHead(
                            voter_parent, round, x, miner_id,
                            self.target, self.target2, self.target3, self.ctr, self.ctr2, 0,
                            self.parentroot, self.contentroot,
                            self.pouw_opt_x[self.segment_index - 2], self.pouw_opt_y[self.segment_index - 2]
                        )
                        blocknew = self.Block(
                            blockhead=blockhead, preblock=voter_parent, isadversary=isadversary,
                            blocksize_MB=global_var.get_blocksize(), index=self.segment_index, vtonprp=[]
                        )
                        blocknew.height = voter_parent.height + 1
                        
                        last_proposer, last_proposer_height = self.get_last_proposer_reference(voter_parent)
                        if last_proposer:
                            blocknew.vtonprp = []
                        else:
                            minority_block = global_var.get_minority_block()
                            blocknew.vtonprp = [minority_block]
                    
                    # 清理状态
                    self.votesOnPrpBks[self.segment_index - 2] = []
                    vtContent[self.segment_index - 2] = []
                    self.pouw_opt_stage = 1
                    
                    if self.segment_index - 2 == number:
                        global_var.set_miner_out(round, miner_id)
                    
                    self.parentroot = None
                    self.contentroot = None
                    self.ctr2 = 0
                    self.fail_count = 0
                    
                    if self.segment_index - 2 == number:
                        global_var.set_dis_suc(self.initial_point)
                        global_var.set_dis_suc_minerid(miner_id)
                        global_var.set_dis_suc_round(round)
                        global_var.add_opt_result(self.pouw_opt_y[self.segment_index - 2])
                        self.initial_point = []
                        self.last_point_time = 0
                        pow_time = self.pow_start
                        use_time = round - pow_time
                        global_var.add_time_pow_suc(use_time)
                        self.pow_start = 0
                    
                    self.segment_index = -1
                    blocksnew.append(blocknew)
            
            # 阶段4：纯PoW计算
            elif self.pouw_opt_stage == 4:
                i = 0
                target_int_3 = int.from_bytes(self.target3, 'big')
                
                while i < 1:
                    self.ctr3 += 1
                    pow_hash = hash_bytes(self.bh + self.ctr3.to_bytes(INT_LEN, BYTE_ORDER)).digest()
                    pow_hash_int = int.from_bytes(pow_hash, 'big')
                    
                    if pow_hash_int < target_int_3:

                        pow_success = True
                        if not isadversary or (isadversary and global_var.get_attack_execute_type() != 'BalanceAttack'):
                            blockhead = self.BlockHead(
                                b_last_vt[self.segment_index - 2], round, x, miner_id,
                                self.target, self.target2, self.target3, self.ctr, self.ctr2, self.ctr3,
                                self.parentroot, self.contentroot, [], 0
                            )
                            blocknew = self.Block(
                                blockhead=blockhead, preblock=b_last_vt[self.segment_index - 2], 
                                isadversary=isadversary, blocksize_MB=global_var.get_blocksize(), 
                                index=self.segment_index, vtonprp=vtContent[self.segment_index - 2]
                            )
                            

                            

                            

                        elif global_var.get_attack_execute_type() == 'BalanceAttack':
                            voter_parent = self.voter_parent_balance
                            blockhead = self.BlockHead(
                                voter_parent, round, x, int.from_bytes(miner_id, BYTE_ORDER, signed=True),
                                self.target, self.target2, self.target3, self.ctr, self.ctr2, self.ctr3,
                                self.parentroot, self.contentroot, [], 0
                            )
                            blocknew = self.Block(
                                blockhead=blockhead, preblock=voter_parent, isadversary=isadversary,
                                blocksize_MB=global_var.get_blocksize(), index=self.segment_index, vtonprp=[]
                            )
                            blocknew.height = voter_parent.height + 1
                            
                            last_proposer, last_proposer_height = self.get_last_proposer_reference(voter_parent)
                            if last_proposer:
                                blocknew.vtonprp = []
                            else:
                                minority_block = global_var.get_minority_block()
                                blocknew.vtonprp = [minority_block]
                        
                        # 清理状态
                        self.votesOnPrpBks[self.segment_index - 2] = []
                        vtContent[self.segment_index - 2] = []
                        self.pouw_opt_stage = 1
                        
                        if self.segment_index - 2 == number:
                            global_var.set_miner_out(round, miner_id)
                        
                        self.ctr3 = 0
                        self.ctr2 = 0
                        self.fail_count = 0
                        
                        # 重新初始化优化变量
                        self.pouw_opt_x[self.segment_index - 2] = np.random.uniform(-fanwei, fanwei, dimension).tolist()
                        self.pouw_opt_y[self.segment_index - 2] = functions.f1(self.pouw_opt_x[self.segment_index - 2])
                        
                        if self.segment_index - 2 == number:
                            self.initial_point = []
                            new_solution = self.pouw_opt_x[self.segment_index - 2]
                            self.initial_point = [new_solution[:]]
                            self.last_point_time = round
                            pow_time = self.pow_start
                            use_time = round - pow_time
                            global_var.add_time_pow(use_time)
                            self.pow_start = 0
                        
                        self.segment_index = -1
                        blocksnew.append(blocknew)
                        i += 1
                    else:
                        i += 1
        
        return (blocksnew, pow_success)

    def local_state_update(self):
        # algorithm 2 比较自己的chain和收到的chain并相应更新本地链
        # output:
        #   lastblock 最长链的最新一个区块
        new_update = False  # 有没有更新
        #touched_hash_list = []
        for incoming_block in self._receive_tape_prp:
            if not isinstance(incoming_block, Consensus.Block):
                continue
            # print(f'verify success, get block{incoming_block.name} on index{incoming_block.index}')
            if self.valid_block(incoming_block):
                prehash = incoming_block.blockhead.prehash
                if insert_point := self.local_chain_prp.search_block_by_hash(prehash):
                    conj_block = self.local_chain_prp.add_blocks(blocks=[incoming_block], insert_point=insert_point)
                    fork_tip, _ = self.synthesize_fork(conj_block)
                    #for block in touched_block:
                    #    touched_hash_list.append(block.blockhash)
                    depthself = self.local_chain_prp.get_height()
                    depth_incoming_block = fork_tip.get_height()
                    if depthself < depth_incoming_block:
                        self.local_chain_prp.set_last_block(fork_tip)
                        new_update = True
                        # self.pouw_opt_stage = 1
                        # self.ctr = 0
                        # self.ctr2 = 0
                        # self.ctr3 = 0
                else:
                    self._block_buffer.setdefault(prehash, [])
                    self._block_buffer[prehash].append(incoming_block)
        for incoming_block in self._receive_tape_vt:
            if not isinstance(incoming_block, Consensus.Block):
                continue
            # print(f'verify success, get block{incoming_block.name} on index{incoming_block.index}')
            if self.valid_block(incoming_block):
                prehash = incoming_block.blockhead.prehash
                if insert_point := self.local_chain_vt[incoming_block.index-2].search_block_by_hash(prehash):
                    conj_block = self.local_chain_vt[incoming_block.index-2].add_blocks(blocks=[incoming_block], insert_point=insert_point)
                    fork_tip, _ = self.synthesize_fork(conj_block)
                    #for block in touched_block:
                    #    touched_hash_list.append(block.blockhash)
                    depthself = self.local_chain_vt[incoming_block.index-2].get_height()
                    depth_incoming_block = fork_tip.get_height()
                    if depthself < depth_incoming_block:
                        self.local_chain_vt[incoming_block.index-2].set_last_block(fork_tip)
                        new_update = True
                        # self.pouw_opt_stage = 1
                        # self.ctr = 0
                        # self.ctr2 = 0
                        # self.ctr3 = 0

                else:
                    self._block_buffer.setdefault(prehash, [])
                    self._block_buffer[prehash].append(incoming_block)
        

        
        #self._block_buffer = {k: v for k, v in self._block_buffer.items() if k not in touched_hash_list}

        return self.local_chain_prp, new_update, self.local_chain_vt

    def valid_chain(self, lastblock: Consensus.Block):
        """验证链是否合法"""
        chain_vali = True
        block_count = 0
        
        if chain_vali and lastblock:
            blocktmp = lastblock
            ss = blocktmp.calculate_blockhash()
            
            while chain_vali and blocktmp is not None:
                block_count += 1
                
                hash_val = blocktmp.calculate_blockhash()
                
                if hash_val != ss:
                    chain_vali = False
                    break
                
                block_vali = self.valid_block(blocktmp)
                if not block_vali:
                    chain_vali = False
                    break
                
                ss = blocktmp.blockhead.prehash
                blocktmp = blocktmp.parentblock
                
                # 防止无限循环
                if block_count > 1000:
                    break
        
        return chain_vali

    def valid_block(self, block: Consensus.Block):
        '''
        验证单个区块是否PoW合法
        param:
            block 要验证的区块 type:Block
        return:
            block_vali 合法标识 type:bool
        '''
        btemp = block
        target = btemp.blockhead.target
        hash_val = btemp.calculate_blockhash()

        if block.index == 0 or block.index == 1:
            # 简单哈希验证
            if hash_val >= target:

                
                print(f'valid unsuccess because hash: {hash_val} and target: {target}')
                return False
            else:
                return True
        else:
            # 第一步：验证哈希是否小于target
            if hash_val >= target:

                
                print(f'valid unsuccess because hash: {hash_val} and target: {target} miner{btemp.blockhead.miner}'
                      f'parent{btemp.blockhead.parentMT_root.encode()} content {btemp.blockhead.contentMT_root.encode()} x {btemp.blockhead.content} nonce{btemp.blockhead.nonce}')
                return False

            # 第二步：验证posthash是否小于target2
            if btemp.blockhead.nonce3 == 0:
                currenthash = hash_val
                posthash = hash_bytes(
                    currenthash + struct.pack(f'{len(btemp.blockhead.pouw_opt_x)}d', *btemp.blockhead.pouw_opt_x) + btemp.blockhead.nonce2.to_bytes(INT_LEN,
                                                                                                         BYTE_ORDER)
                ).digest()
                if int.from_bytes(posthash, 'big') >= int.from_bytes(btemp.blockhead.target2, 'big'):

                    

                    return False

                # 第三步：验证pouw_opt_x在优化函数计算下的结果是否等于pouw_opt_y
                pouw_opt_x = btemp.blockhead.pouw_opt_x
                calculated_pouw_opt_y = functions.f1(pouw_opt_x)

                if calculated_pouw_opt_y != btemp.blockhead.pouw_opt_y:

                    

                    return False

            else:
                target3 = btemp.blockhead.target3
                pow_hash = hash_bytes(hash_val + btemp.blockhead.nonce3.to_bytes(INT_LEN, BYTE_ORDER)).digest()
                if int.from_bytes(pow_hash, 'big') >= int.from_bytes(target3, 'big'):

                    

                    return False
                if block.index in global_var.get_start_time():
                    if block.index not in global_var.get_com_time():
                        start_time = global_var.get_start_time()[block.index]
                        time_cost = btemp.blockhead.timestamp - start_time
                        global_var.set_com_time(block.index, time_cost)
            return True

    def receive_block(self, rcvblock):
        """接收区块的处理逻辑"""
        segment_index = rcvblock.index
        number = global_var.get_number()
        fanwei = global_var.get_fanwei()
        
        if segment_index == 0:  # 交易区块
            if rcvblock in self._receive_tape_tx:
                return False
            if block_list := self._block_buffer.get(rcvblock.blockhead.prehash, None):
                for block in block_list:
                    if block.blockhash == rcvblock.blockhash:
                        return False
            self._receive_tape_tx.append(rcvblock)
            self.unref_tx.append(rcvblock.name)
            return True
        
        elif segment_index == 1:  # 提案区块
            if rcvblock in self._receive_tape_prp:
                return False
            if block_list := self._block_buffer.get(rcvblock.blockhead.prehash, None):
                for block in block_list:
                    if block.blockhash == rcvblock.blockhash:
                        return False
            if self.in_local_chain_prp(rcvblock):
                return False
            
            self._receive_tape_prp.append(rcvblock)
            self.unref_prp.append(rcvblock.name)
            
            # 处理投票
            now_lastblock_prp = self.local_chain_prp.get_last_block()
            if rcvblock.height == now_lastblock_prp.height + 1:
                for i, vt_chain in enumerate(self.local_chain_vt):
                    if not vt_chain.has_voted(rcvblock.height):
                        vt_chain.vote(rcvblock.height, rcvblock)
                        self.votesOnPrpBks[i].append(rcvblock)
            
            # 移除已引用的交易和提案
            for block in rcvblock.txblock:
                self.remove_unreftx(block)
            for block in rcvblock.prpblock:
                self.remove_unrefprp(block)
            
            return True
        
        elif segment_index < (global_var.get_vtnum() + 2):  # 投票区块
            if rcvblock in self._receive_tape_vt:
                return False
            if block_list := self._block_buffer.get(rcvblock.blockhead.prehash, None):
                for block in block_list:
                    if block.blockhash == rcvblock.blockhash:
                        return False
            if self.in_local_chain_vt(rcvblock):
                return False
            
            self._receive_tape_vt.append(rcvblock)
            
            # 更新投票记录
            self.votesOnPrpBks[segment_index - 2] = [item for item in self.votesOnPrpBks[segment_index - 2] 
                                                    if item not in rcvblock.vtonprp]
            
            for prp_block in rcvblock.vtonprp:
                if self.in_local_chain_prp(prp_block):
                    vtc = self.local_chain_vt[segment_index - 2]
                    if vtc.has_voted(prp_block.height):
                        old_block_name = vtc.votes[prp_block.height]
                        voteon = self.votesOnPrpBks[segment_index - 2]
                        if old_block_name in voteon:
                            voteon.remove(old_block_name)
                            vtc.votes[prp_block.height] = prp_block
                    else:
                        vtc.vote(prp_block.height, prp_block)
            
            # 处理优化状态更新
            if rcvblock.blockhead.nonce3 != 0:
                if segment_index - 2 == number:
                    if len(self.initial_point) > 1 and self.pouw_opt_stage == 2:
                        global_var.set_dis(self.initial_point)
                        global_var.set_dis_minerid(self.miner_id)
                        global_var.set_dis_round(rcvblock.blockhead.timestamp)
                    self.initial_point = []
                
                dimension = global_var.get_N()
                self.pouw_opt_x[segment_index - 2] = np.random.uniform(-fanwei, fanwei, dimension).tolist()
                self.pouw_opt_y[segment_index - 2] = functions.f1(self.pouw_opt_x[segment_index - 2])
                
                if segment_index - 2 == number:
                    new_solution = self.pouw_opt_x[segment_index - 2]
                    self.initial_point = [new_solution[:]]
                
                if segment_index == self.segment_index:
                    self.pouw_opt_stage = 1
                    if segment_index - 2 == number:
                        global_var.set_miner_out(rcvblock.blockhead.timestamp, self.miner_id)
                        self.opt_start = 0
                        self.pow_start = 0
                    self.ctr2 = 0
                    self.ctr3 = 0
                    self.fail_count = 0
                    self.segment_index = -1
            else:
                # 根据优化方法选择不同的判断逻辑
                optimization_method = global_var.get_optimization_method()
                should_update = False
                
                if optimization_method == 'threshold':
                    # 阈值方法：比当前解好一个阈值就更新
                    height = rcvblock.height if hasattr(rcvblock, 'height') else 0
                    yuzhi = global_var.calculate_dynamic_threshold(height)
                    should_update = (rcvblock.blockhead.pouw_opt_y <= self.pouw_opt_y[segment_index - 2] - yuzhi or 
                                   self.pouw_opt_x[segment_index - 2] == [])
                else:
                    # KKT方法：只有更好的解才更新
                    should_update = rcvblock.blockhead.pouw_opt_y <= self.pouw_opt_y[segment_index - 2]
                
                if should_update:
                    self.pouw_opt_x[segment_index - 2] = rcvblock.blockhead.pouw_opt_x
                    self.pouw_opt_y[segment_index - 2] = rcvblock.blockhead.pouw_opt_y
                    
                    if segment_index - 2 == number:

                        if len(self.initial_point) > 1 and self.pouw_opt_stage == 2:
                            global_var.set_dis(self.initial_point)
                            global_var.set_dis_minerid(self.miner_id)
                            global_var.set_dis_round(rcvblock.blockhead.timestamp)
                        new_solution = self.pouw_opt_x[segment_index - 2]
                        self.initial_point = [new_solution[:]]
                        self.last_point_time = rcvblock.blockhead.timestamp

                
                if segment_index == self.segment_index:
                    self.pouw_opt_stage = 1
                    if segment_index - 2 == number:
                        global_var.set_miner_out(rcvblock.blockhead.timestamp, self.miner_id)
                    self.ctr2 = 0
                    self.ctr3 = 0
                    self.fail_count = 0
                    self.segment_index = -1
            
            return True
        
        return False

    def in_local_chain_prp(self, block):
        """检查区块是否在提案链中"""
        if self.local_chain_prp.search_block(block) is None:
            return False
        return True

    def in_local_chain_vt(self, block):
        """检查区块是否在指定投票链中"""
        index = block.index
        if self.local_chain_vt[index-2].search_block(block) is None:
            return False
        return True

    def consensus_process(self, isadversary, x, round, tasks=None):
        """共识过程主函数"""
        newblocks, success = self.mining_consensus(self.miner_id, isadversary, x, round, tasks)
        
        if success is False:
            return None, False
        else:
            newblocks_after = []
            for newblock in newblocks:
                if newblock.index == 0:
                    pass
                elif newblock.index == 1:
                    newblock = self.local_chain_prp.add_blocks(blocks=newblock)
                else:
                    newblock = self.local_chain_vt[newblock.index - 2].add_blocks(blocks=newblock)
                
                # 记录日志
                import logging
                logger = logging.getLogger(__name__)
                logger.info("round %d, M%d mined %s", round, self.miner_id, newblock.name)
                
                newblocks_after.append(newblock)
            
            return newblocks_after, True