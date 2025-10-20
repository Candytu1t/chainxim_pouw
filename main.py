import configparser
import logging
import time
import random
import os

import numpy as np

import global_var
from environment import Environment


def get_time(f):
    def inner(*arg, **kwarg):
        s_time = time.time()
        res = f(*arg, **kwarg)
        e_time = time.time()
        print('耗时：{}秒'.format(e_time - s_time))
        return res
    return inner

@get_time
def run(Z:Environment, total_round: int, max_height: int,
        process_bar_type, post_verification:bool, quantile: float):
    Z.exec(total_round, max_height, process_bar_type, post_verification)
    return Z.view_and_write(quantile)

def config_log(env_config:dict):
    """配置日志"""
    level_str = env_config.get('log_level', 'info')
    if level_str == 'error':
        global_var.set_log_level(logging.ERROR)
    elif level_str == 'warning':
        global_var.set_log_level(logging.WARNING)
    elif level_str == 'info':
        global_var.set_log_level(logging.INFO)
    elif level_str == 'debug':
        global_var.set_log_level(logging.DEBUG)
    else:
        raise ValueError("config error log_level must be set as " +
                         "error/warning/info/debug, cur setting:%s", level_str)
    logging.basicConfig(filename=global_var.get_result_path() / 'events.log',
                        level=global_var.get_log_level(), filemode='w')


def main(**args):
    '''主程序'''

    # 读取配置文件
    config = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
    config.optionxform = lambda option: option
    config.read(args.get('config'),encoding='utf-8')
    env_config = dict(config['EnvironmentSettings'])
    #设置全局变量
    miner_num = args.get('miner_num') or int(env_config['miner_num'])
    network_type = args.get('network_type') or env_config['network_type']
    global_var.__init__(args.get('result_path'))
    global_var.set_consensus_type(
        args.get('consensus_type') or env_config['consensus_type'])
    global_var.set_network_type(network_type)
    global_var.set_miner_num(miner_num)
    global_var.set_blocksize(
        args.get('blocksize') or int(env_config['blocksize']))
    global_var.set_show_fig(
        args.get('show_fig') or config.getboolean('EnvironmentSettings','show_fig'))
    global_var.set_compact_outputfile(
        config.getboolean('EnvironmentSettings','compact_outputfile')
        if not args.get('no_compact_outputfile') else False)
    global_var.set_common_prefix_enable(
        config.getboolean('EnvironmentSettings','common_prefix_enable'))
    
    # 设置性能分析器开关
    global_var.set_performance_analyzer_enable(
        config.getboolean('EnvironmentSettings','performance_analyzer_enable'))
    global_var.set_memory_cleanup_enable(
        config.getboolean('EnvironmentSettings','memory_cleanup_enable'))
    
    # 配置日志
    config_log(env_config)
    
    # 设置共识协议参数
    consensus_settings = dict(config['ConsensusSettings'])
    if global_var.get_consensus_type() in ['consensus.PoW', 'consensus.VirtualPoW', 'consensus.SolidPoW']:
        q_ave = args.get('q_ave') or float(consensus_settings['q_ave'])
        global_var.set_ave_q(q_ave)
        q_distr = args.get('q_distr') or consensus_settings['q_distr']
        if q_distr == 'rand':
            q_distr = get_random_q_gaussian(miner_num,q_ave)
        average_block_time = args.get('average_block_time') or float(consensus_settings['average_block_time'])
        if average_block_time == 0:
            target = args.get('target') or consensus_settings['target']
            global_var.set_PoW_target(target)
        else:
            target =  f"{round(2**256/miner_num/q_ave/average_block_time):064x}"
            global_var.set_PoW_target(target)
        consensus_param = {'target':target, 'q_ave':q_ave, 'q_distr':q_distr, 'N': args.get('N') or int(consensus_settings['N'])}
    elif global_var.get_consensus_type() == 'consensus.PoUW':
        # PoUW共识特殊参数处理
        q_ave = args.get('q_ave') or float(consensus_settings['q_ave'])
        global_var.set_ave_q(q_ave)
        q_distr = args.get('q_distr') or consensus_settings['q_distr']
        if q_distr == 'rand':
            q_distr = get_random_q_gaussian(miner_num,q_ave)
        
        # 设置PoUW相关全局参数
        global_var.set_vtnum(args.get('vtnum') or int(consensus_settings.get('vtnum', 5)))
        global_var.set_N(args.get('optimization_dimension') or int(consensus_settings.get('optimization_dimension', 2)))
        global_var.set_fanwei(args.get('search_range') or float(consensus_settings.get('search_range', 20)))
        global_var.set_task_num(args.get('task_num') or int(consensus_settings.get('task_num', 1)))
        global_var.set_kkt_yuzhi(args.get('kkt_threshold') or float(consensus_settings.get('kkt_threshold', 0.001)))
        global_var.set_repeat_time(args.get('retry_times') or int(consensus_settings.get('retry_times', 10)))
        
        # 设置优化方法和阈值参数
        global_var.set_optimization_method(args.get('optimization_method') or consensus_settings.get('optimization_method', 'kkt'))
        global_var.set_yuzhi(args.get('threshold_base') or float(consensus_settings.get('threshold_base', 4000)))
        global_var.set_yuzhi_change(args.get('threshold_change_rate') or float(consensus_settings.get('threshold_change_rate', 0.8)))
        global_var.set_yuzhi_min(args.get('threshold_min') or float(consensus_settings.get('threshold_min', 4000)))
        
        # 处理难度目标
        average_block_time = args.get('average_block_time') or float(consensus_settings['average_block_time'])
        if average_block_time == 0:
            target = args.get('target') or consensus_settings['target']
            target2 = args.get('target2') or consensus_settings.get('target2', target)
            target3 = args.get('target3') or consensus_settings.get('target3', target)
        else:
            target = f"{round(2**256/miner_num/q_ave/average_block_time):064x}"
            target2 = args.get('target2') or consensus_settings.get('target2', target)
            target3 = args.get('target3') or consensus_settings.get('target3', target)
        
        global_var.set_PoW_target(target)
        
        consensus_param = {
            'target': target,
            'target2': target2, 
            'target3': target3,
            'q_ave': q_ave,
            'q_distr': q_distr,
            'N': args.get('N') or int(consensus_settings['N'])
        }
    else:
        consensus_param = {}
        for key, value in consensus_settings.items():
            consensus_param[key] = args.get(key) or value

    # 设置网络参数
    network_param = {}
    # StochPropNetwork
    if network_type == 'network.StochPropNetwork':
        bdnet_settings = dict(config["StochPropNetworkSettings"])
        network_param = {
            'rcvprob_start': (args.get('rcvprob_start') 
                              if args.get('rcvprob_start') is not None  
                              else float(bdnet_settings['rcvprob_start'])),
            'rcvprob_inc': (args.get('rcvprob_inc') 
                            if args.get('rcvprob_inc') is not None
                            else float(bdnet_settings['rcvprob_inc'])),
            'stat_prop_times': (args.get('stat_prop_times') or 
                                eval(bdnet_settings['stat_prop_times']))
        }
    # DeterPropNetwork
    elif network_type == 'network.DeterPropNetwork':
        pvnet_settings = dict(config["DeterPropNetworkSettings"])
        network_param = {'prop_vector':(args.get('prop_vector') or 
                                        eval(pvnet_settings['prop_vector']))}
    # TopologyNetwork
    elif network_type == 'network.TopologyNetwork':
        net_setting = 'TopologyNetworkSettings'
        bool_params  = ['show_label', 'save_routing_graph', 'save_routing_history',
                        'dynamic','enable_resume_transfer']
        float_params = ['ave_degree', 'bandwidth_honest', 'bandwidth_adv',
                        'outage_prob','avg_tp_change_interval','edge_add_prob',
                        'edge_remove_prob','max_allowed_partitions']
        for bparam in bool_params:
            network_param.update({bparam: args.get(bparam) or 
                                 config.getboolean(net_setting, bparam)})
        for fparam in float_params:
            network_param.update({fparam: args.get(fparam) or 
                                  config.getfloat(net_setting, fparam)})
        network_param.update({
            'init_mode': (args.get('init_mode') or 
                          config.get(net_setting, 'init_mode')),
            'topology_path': (args.get('topology_path') or 
                              config.get(net_setting, 'topology_path', fallback=None)),
            'stat_prop_times': (args.get('stat_prop_times') or 
                                eval(config.get(net_setting, 'stat_prop_times'))),
            'rand_mode': (args.get('rand_mode') or
                          config.get(net_setting, 'rand_mode'))
        })
    # AdHocNetwork
    elif network_type == 'network.AdHocNetwork':
        net_setting = 'AdHocNetworkSettings'
        bool_params  = ['enable_large_scale_fading']
        float_params = ['ave_degree', 'region_width', 'comm_range','move_variance',
                        'outage_prob','segment_size','bandwidth_max'] # 'min_move', 'max_move'
        for bparam in bool_params:
            network_param.update({bparam: args.get(bparam) or 
                                 config.getboolean(net_setting, bparam)})
        for fparam in float_params:
            network_param.update({fparam: args.get(fparam) or 
                                  config.getfloat(net_setting, fparam)})
        network_param.update({
            'init_mode': (args.get('init_mode') or config.get(net_setting, 'init_mode')),
            'stat_prop_times': (args.get('stat_prop_times') or eval(config.get(net_setting, 'stat_prop_times'))),
            'path_loss_level': (args.get('path_loss_level') or config.get(net_setting, 'path_loss_level')),
        })
        # global_var.set_segmentsize(config.getfloat(net_setting, "segment_size"))

    # 设置attack参数
    attack_setting = dict(config['AttackSettings'])
    global_var.set_attack_execute_type(args.get('attack_type') or attack_setting.get('attack_type', 'HonestMining'))
    attack_param = {
        'adver_num'    : (args.get('adver_num') if args.get('adver_num') is not None 
                          else int(attack_setting['adver_num'])),
        'attack_type'  : (args.get('attack_type') if args.get('attack_type') is not None
                          else attack_setting.get('attack_type', 'HonestMining')),
        'attack_arg'   : eval(attack_setting.get('attack_arg')) if attack_setting.get('attack_arg') is not None
                          else {},
        'adversary_ids': (args.get('adver_lists') if args.get('adver_lists') is not None
                          else eval(attack_setting.get('adver_lists') or 'None')),
    }
    attack_param['attack_arg'].update(args.get('attack_arg', {}))
    attack_param['attack_arg'].update({'N': consensus_param.get('N', 1)})

    # 设置dataitem相关的配置
    dataitem_setting = dict(config['DataItemSettings'])
    dataitem_param = {
        'dataitem_enable': args.get('dataitem_enable') or config.getboolean('DataItemSettings','dataitem_enable'),
        'max_block_capacity': (args.get('max_block_capacity') or 
                               int(dataitem_setting['max_block_capacity'])),
        'dataitem_size': (args.get('dataitem_size') or 
                          int(dataitem_setting['dataitem_size'])),
        'dataitem_input_interval': (args.get('dataitem_input_interval') or 
                                    int(dataitem_setting['dataitem_input_interval'])),
    }

    # =============== 批量运行支持（可选） ===============
    # 从配置文件读取批量设置（如存在）
    def _cfg(section, key, fallback=None):
        try:
            return config.get(section, key, fallback=fallback)
        except Exception:
            return fallback
    def _cfg_bool(section, key, fallback=False):
        try:
            return config.getboolean(section, key, fallback=fallback)
        except Exception:
            return fallback

    batch_enable = args.get('batch_enable') or _cfg_bool('BatchSettings', 'batch_enable', False)

    if batch_enable:
        # 限制并行库线程数，避免资源打满
        max_cores = args.get('max_cores') or (_cfg('BatchSettings', 'max_cores') and int(_cfg('BatchSettings', 'max_cores'))) or 6
        total_cores = os.cpu_count() or max_cores
        usable_cores = min(int(max_cores), int(total_cores))
        os.environ["OMP_NUM_THREADS"] = str(usable_cores)
        os.environ["MKL_NUM_THREADS"] = str(usable_cores)
        os.environ["NUMEXPR_NUM_THREADS"] = str(usable_cores)
        os.environ["OPENBLAS_NUM_THREADS"] = str(usable_cores)

        def _parse_list(val, cast=float):
            if val is None:
                return None
            if isinstance(val, list):
                return val
            s = str(val).strip()
            if not s:
                return None
            try:
                # 优先按 Python 表达式解析
                lst = eval(s, {"__builtins__": {}}, {})
                if isinstance(lst, list):
                    return [cast(x) for x in lst]
            except Exception:
                pass
            # 退化为逗号分隔
            try:
                return [cast(x.strip()) for x in s.split(',') if x.strip()]
            except Exception:
                return None

        # 读取批量参数列表（支持命令行或配置文件BatchSettings；未提供则使用当前配置的单值）
        # mlist: voter链数量列表
        mlist = (_parse_list(args.get('mlist'), int) or
                 _parse_list(_cfg('BatchSettings', 'mlist'), int) or
                 [global_var.get_vtnum()])
        # nlist: 矿工数量列表（n，miner_num）
        miner_nums = (_parse_list(args.get('nlist'), int) or
                      _parse_list(_cfg('BatchSettings', 'nlist'), int) or
                      [global_var.get_miner_num()])
        # Nlist: 优化维度列表（N，optimization dimension）
        Nlist = (_parse_list(args.get('Nlist'), int) or
                 _parse_list(_cfg('BatchSettings', 'Nlist'), int) or
                 [global_var.get_N()])
        task_numlist = (_parse_list(args.get('task_numlist'), int) or
                        _parse_list(_cfg('BatchSettings', 'task_numlist'), int) or
                        [global_var.get_task_num()])
        confirm_depths = (_parse_list(args.get('confirm_depths'), int) or
                          _parse_list(_cfg('BatchSettings', 'confirm_depths'), int) or
                          [consensus_param.get('N', 1)])
        adver_nums = (_parse_list(args.get('adver_nums'), int) or
                      _parse_list(_cfg('BatchSettings', 'adver_nums'), int) or
                      [attack_param['adver_num']])
        retry_list = (_parse_list(args.get('retry_list'), int) or
                      _parse_list(_cfg('BatchSettings', 'retry_list'), int) or
                      [global_var.get_repeat_time()])
        yuzhis = (_parse_list(args.get('yuzhis'), float) or
                  _parse_list(_cfg('BatchSettings', 'yuzhis'), float) or
                  [global_var.get_yuzhi()])
        yuzhi_changes = (_parse_list(args.get('yuzhi_changes'), float) or
                         _parse_list(_cfg('BatchSettings', 'yuzhi_changes'), float) or
                         [global_var.get_yuzhi_change()])
        iterations = int(args.get('iterations') or (_cfg('BatchSettings', 'iterations') or 1))
        # 选择优化方法列表（kkt/threshold），支持列表或逗号分隔
        opt_methods = (_parse_list(args.get('optimization_method'), str) or
                       _parse_list(_cfg('BatchSettings', 'optimization_method'), str) or
                       [consensus_settings.get('optimization_method', 'kkt')])

        total_round_base = args.get('total_round') or int(env_config['total_round'])
        max_height_base = (args.get('total_height') or int(env_config.get('total_height') or 2**31 - 2))
        process_bar_type_base = (args.get('process_bar_type') or env_config.get('process_bar_type'))
        post_verification = args.get('disable_post_verification')
        quantile = config.getfloat('EnvironmentSettings', 'consensus_miner_quantile')

        batch_count = 0
        for task_num in task_numlist:
            for yuzhi in yuzhis:
                for yuzhi_change in yuzhi_changes:
                    for m in mlist:
                        for n_miners in miner_nums:
                            for N_dim in Nlist:
                                for opt_method in opt_methods:
                                    for confirm_depth in confirm_depths:
                                        for retry_times in retry_list:
                                            for adver_num in adver_nums:
                                                for _ in range(iterations):
                                                    # 为每次仿真设置全局参数
                                                    global_var.set_vtnum(m)
                                                    # N 为优化维度
                                                    global_var.set_N(N_dim)
                                                    # 优化方法
                                                    global_var.set_optimization_method(opt_method)
                                                    global_var.set_task_num(task_num)
                                                    global_var.set_yuzhi(yuzhi)
                                                    global_var.set_yuzhi_change(yuzhi_change)
                                                    global_var.set_repeat_time(retry_times)
                                                    # n 为矿工数量
                                                    global_var.set_miner_num(n_miners)

                                                    # 按 m 动态设置 target（兼容 chain-xim-master-thre 的策略）
                                                    new_target = None
                                                    if m == 20:
                                                        new_target = '05A1CAC083126E978D4FDF3B645A1CAC083126E978D4FDF3B645A1CAC083126E'
                                                    elif m == 3:
                                                        new_target = '0147AE147AE147AE147AE147AE147AE147AE147AE147AE147AE147AE147AE147'
                                                    elif m == 30:
                                                        new_target = '083126E978D4FDF3B645A1CAC083126E978D4FDF3B645A1CAC083126E978D4FD'
                                                    elif m == 40:
                                                        new_target = '0AC083126E978D4FDF3B645A1CAC083126E978D4FDF3B645A1CAC083126E978D'
                                                    elif m == 150:
                                                        new_target = '26E978D4FDF3B645A1CAC083126E978D4FDF3B645A1CAC083126E978D4FDF3B6'
                                                    elif m == 50:
                                                        new_target = '0D4FDF3B645A1CAC083126E978D4FDF3B645A1CAC083126E978D4FDF3B645A1C'
                                                    elif m == 100:
                                                        new_target = '1A1CAC083126E978D4FDF3B645A1CAC083126E978D4FDF3B645A1CAC083126E9'
                                                    elif m == 10:
                                                        new_target = '03126E978D4FDF3B645A1CAC083126E978D4FDF3B645A1CAC083126E978D4FDF'
                                                    elif m == 5:
                                                        new_target = '01CAC083126E978D4FDF3B645A1CAC083126E978D4FDF3B645A1CAC083126E97'
                                                    elif m == 11:
                                                        new_target = '0353F7CED916872B020C49BA5E353F7CED916872B020C49BA5E353F7CED91687'
                                                    elif m == 21:
                                                        new_target = '05E353F7CED916872B020C49BA5E353F7CED916872B020C49BA5E353F7CED916'
                                                    elif m == 1:
                                                        new_target = '01CAC083126E978D4FDF3B645A1CAC083126E978D4FDF3B645A1CAC083126E97'

                                                    # 写出关键信息（速度与参数关系）到 D 盘（与参考实现保持一致）
                                                    try:
                                                        with open('D:/suduthre.txt', 'a', encoding='utf-8') as f:
                                                            f.write(f"miner: {n_miners} and chain_number:{m} and task_num:{task_num}\n")
                                                    except Exception:
                                                        pass
                                                    try:
                                                        with open('D:/rguanxi.txt', 'a', encoding='utf-8') as f:
                                                            f.write(f"m,n:{m},{n_miners},r: {retry_times}\n")
                                                    except Exception:
                                                        pass

                                                    # 动态范围/阈值（按任务类型）
                                                    global_var.set_yuzhi_change(yuzhi_change)
                                                    if task_num == 7 or task_num == 17:
                                                        global_var.set_fanwei(600)
                                                        global_var.set_yuzhi(yuzhi)
                                                        global_var.set_yuzhi_min(1)
                                                    elif task_num == 8:
                                                        global_var.set_fanwei(5.12)
                                                        global_var.set_yuzhi(yuzhi)
                                                        global_var.set_yuzhi_min(0.1)
                                                    elif task_num == 9:
                                                        global_var.set_fanwei(500)
                                                        global_var.set_yuzhi(yuzhi)
                                                        global_var.set_yuzhi_min(1)
                                                    elif task_num == 14:
                                                        global_var.set_fanwei(100)

                                                    # 将 miner_num、chain_num 等写到链路输出文件（与参考实现兼容）
                                                    try:
                                                        tasknum_cur = global_var.get_task_num()
                                                        yzchg_cur = global_var.get_yuzhi_change()
                                                        yz_cur = global_var.get_yuzhi()
                                                        fp1 = f'D:/chainximopt_{tasknum_cur}_{yzchg_cur}_{yz_cur}.txt'
                                                        with open(fp1, 'a', encoding='utf-8') as f:
                                                            f.write(f"miner_num:{n_miners}\n")
                                                        fp2 = f'D:/chainximopt_{tasknum_cur}_time.txt'
                                                        with open(fp2, 'a', encoding='utf-8') as f:
                                                            f.write(f"miner_num:{n_miners}\nchain_num:{m}\n")
                                                    except Exception:
                                                        pass

                                                # 构造本次运行用的参数对象（保持其他配置一致）
                                                    attack_param_loop = dict(attack_param)
                                                    attack_param_loop['adver_num'] = adver_num

                                                    consensus_param_loop = dict(consensus_param)
                                                    consensus_param_loop['N'] = confirm_depth
                                                    # 同步设置确认深度到全局（供统计/外部函数使用）
                                                    global_var.set_depth(confirm_depth)
                                                    if new_target:
                                                        consensus_param_loop['target'] = new_target

                                                    # 每次运行重新实例化环境
                                                    Z_loop = Environment(attack_param_loop, consensus_param_loop, network_param, {}, {}, dataitem_param)
                                                    total_round = total_round_base
                                                    max_height = max_height_base
                                                    process_bar_type = process_bar_type_base

                                                    try:
                                                        run(Z_loop, total_round, max_height, process_bar_type, post_verification, quantile)
                                                    except AssertionError as e:
                                                        # 跳过单次失败但继续批量
                                                        print(f"[Batch] Post verification failed, skip. detail: {e}")
                                                    except Exception as e:
                                                        print(f"[Batch] Unexpected error, skip: {e}")
                                                    finally:
                                                        batch_count += 1

        print(f"Batch simulations finished. Total runs: {batch_count}")
        return

    # =============== 单次运行（默认） ===============
    genesis_blockheadextra = {}
    genesis_blockextra = {}

    Z = Environment(attack_param, consensus_param, network_param,
                    genesis_blockheadextra, genesis_blockextra, dataitem_param)
    total_round = args.get('total_round') or int(env_config['total_round'])
    max_height = (args.get('total_height') or 
                  int(env_config.get('total_height') or 2**31 - 2))
    process_bar_type = (args.get('process_bar_type') 
                        or env_config.get('process_bar_type'))

    return run(Z, total_round, max_height, process_bar_type, args.get('disable_post_verification'),
               config.getfloat('EnvironmentSettings','consensus_miner_quantile'))

def get_random_q_gaussian(miner_num,q_ave):
    '''
    随机设置各个节点的hash rate,满足均值为q_ave,方差为1的高斯分布
    且满足全网总算力为q_ave*miner_num
    '''
    # 生成均值为ave_q，方差为0.2*q_ave的高斯分布
    q_dist = np.random.normal(q_ave, 0.2*q_ave, miner_num)
    # 归一化到总和为total_q，并四舍五入为整数
    total_q = q_ave * miner_num
    q_dist = total_q / np.sum(q_dist) * q_dist
    q_dist = np.round(q_dist).astype(int)
    # 修正，如果和不为total_q就把差值分摊在最小值或最大值上
    if np.sum(q_dist) != total_q:
        diff = total_q - np.sum(q_dist)
        for _ in range(abs(diff)):
            sign_diff = np.sign(diff)
            idx = np.argmin(q_dist) if sign_diff > 0 else np.argmax(q_dist)
            q_dist[idx] += sign_diff
    return str(list(q_dist))


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    program_description = 'ChainXim, a blockchain simulator developed by XinLab\
, simulates and assesses blockchain system with various consensus protocols \
under different network conditions. Security evaluation of blockchain systems \
could be performed with attackers designed in the simulator'
    parser = argparse.ArgumentParser(description=program_description)
    parser.add_argument('--config', '-c', help='The path to the config file.', type=str, default='system_config.ini')
    # EnvironmentSettings
    env_setting = parser.add_argument_group('EnvironmentSettings','Settings for Environment')
    env_setting.add_argument('--process_bar_type', help='Set the style of process bar: round/height',type=str)
    env_setting.add_argument('--total_round', help='Total rounds before simulation stops.', type=int)
    env_setting.add_argument('--total_height', help='Total block height generated before simulation stops.', type=int)
    env_setting.add_argument('--miner_num', help='The total miner number in the network.', type=int)
    env_setting.add_argument('--consensus_type',help='The consensus class imported during simulation',type=str)
    env_setting.add_argument('--network_type',help='The network class imported during simulation',type=str)
    env_setting.add_argument('--blocksize', help='The size of each block in MB. Only effective when dataitem_enable=False and network_type is TopologyNetwork or AdhocNetwork.',type=float)
    env_setting.add_argument('--show_fig', help='Show figures during simulation.',action='store_true')
    env_setting.add_argument('--no_compact_outputfile', action='store_true',
                             help='Simplify log and result outputs to reduce disk space consumption. Use no_compact_outputfile to disable this feature.')
    # ConsensusSettings
    consensus_setting = parser.add_argument_group('ConsensusSettings', 'Settings for Consensus Protocol')
    consensus_setting.add_argument('--q_ave', help='The average number of hash trials in a round.',type=int)
    consensus_setting.add_argument('--q_distr', help='distribution of hash rate across all miners.\
                        \'equal\': all miners have equal hash rate;\
                        \'rand\': q satisfies gaussion distribution.',type=str)
    consensus_setting.add_argument('-N', help='The number of block confirmations required for the transactoins in a block to be considered valid. Used to calculate attack success rate.', type=int)
    # Ensure only one of difficulty and average_block_time can be set
    group = consensus_setting.add_mutually_exclusive_group()
    group.add_argument('--difficulty', help='The number of zero prefix of valid block hash.\
                    A metric for Proof of Work difficulty.', type=int)
    group.add_argument('--average_block_time', help='The average time interval between two blocks.\
                    If average_block_time=0, then target is set as the difficulty.', type=float)
    # AttackModeSettings
    attack_setting = parser.add_argument_group('AttackModeSettings','Settings for Attack')
    attack_setting.add_argument('--adver_num',help='The total number of attackers. If adver_num is non-zero and adversary_ids not specified, then attackers are randomly selected.',type=int)
    attack_setting.add_argument('--adver_lists', help='Specify id of adversaries. If adversary_ids is set, `adver_num` will not take effect.', type=str)
    attack_setting.add_argument('--attack_type', help='The name of attack type defined in attack mode.',type=str)
    attack_setting.add_argument('-Ng', help='Parameter Ng for DoubleSpending', type=int)
    attack_setting.add_argument('--eclipse_target', help='The target miner id for eclipse attack.', type=str)
    # StochPropNetworkSettings
    stoch_setting = parser.add_argument_group('StochPropNetworkSettings','Settings for StochPropNetwork')
    stoch_setting.add_argument('--rcvprob_start', help='Initial receive probability when a block access network.',type=float)
    stoch_setting.add_argument('--rcvprob_inc',help='Increment of rreceive probability per round.', type=float)
    # CommonSettings for both TopologyNetwork and AdhocNetwork
    common_setting = parser.add_argument_group('CommonSettings', 'Common settings for both TopologyNetwork and AdhocNetwork')
    init_mode_help_text = ''' Options:coo/adj/rand.
    coo: load adjecent matrix in COOrdinate format.
    adj: load adjecent matrix.
    rand: randomly generate a network. AdhocNetwork only support this.'''
    common_setting.add_argument('--init_mode', help='Initialization mode for the network.' + init_mode_help_text, type=str)
    common_setting.add_argument('--topology_path', help='The relative path to the topology file in a csv format specified by init_mode.', type=str)
    common_setting.add_argument('--ave_degree', help='Set the average degree of the network.', type=float)
    common_setting.add_argument('--outage_prob', help='The outage probability of each link.', type=float)
    # TopologyNetworkSettings
    topology_setting = parser.add_argument_group('TopologyNetworkSettings', 'Settings for TopologyNetwork')
    topology_setting.add_argument('--show_label', help='Show edge labels on network and routing graph.', action='store_true')
    topology_setting.add_argument('--save_routing_graph', help='Generate routing graph at the end of simulation or not.', action='store_true')
    rand_mode_help_text = '''Options:homogeneous/binomial.
    homogeneous: try to keep the degree of each node the same.
    binomial: set up edges with probability ave_degree/(miner_num-1). '''
    topology_setting.add_argument('--rand_mode', help=rand_mode_help_text, type=str)
    topology_setting.add_argument('--bandwidth_honest', help='Set bandwidth between honest miners and between the honest and adversaries(MB/round)', type=float)
    topology_setting.add_argument('--bandwidth_adv', help='Set bandwidth between adversaries(MB/round)', type=float)
    topology_setting.add_argument('--dynamic', help='Whether the network topology can dynamically change.', action='store_true')
    topology_setting.add_argument('--avg_tp_change_interval', help='The average interval of topology changing.', type=float)
    topology_setting.add_argument('--edge_remove_prob', help='The probability of each links being removed in each round.', type=float)
    topology_setting.add_argument('--edge_add_prob', help='The probability of each links being added in each round.', type=float)
    topology_setting.add_argument('--max_allowed_partitions', help='The maximum number of network partitions being allowed.', type=int)
    # AdhocNetworkSettings
    adhoc_setting = parser.add_argument_group('AdhocNetworkSettings', 'Settings for AdhocNetwork')
    adhoc_setting.add_argument('--region_width', help='Width of the region for the network.', type=float)
    adhoc_setting.add_argument('--comm_range', help='Communication range.', type=float)
    adhoc_setting.add_argument('--move_variance', help='Variance of the movement when position updates in Gaussian random walk.', type=float)
    adhoc_setting.add_argument('--enable_large_scale_fading', help='If large-scale fading is enabled, the segment size will be adjusted automatically according to the fading model.', action='store_true')
    adhoc_setting.add_argument('--path_loss_level', help='Path loss level. low/medium/high', type=str)
    adhoc_setting.add_argument('--bandwidth_max', help='The max bandwidth is the bandwidth within the range of comm_range/100 in MB/round.', type=float)
    # DataItemSettings
    dataitem_setting = parser.add_argument_group('DataItemSettings', 'Settings for DataItem')
    dataitem_setting.add_argument('--dataitem_enable', help='Enable dataitem generation.', action='store_true')
    dataitem_setting.add_argument('--max_block_capacity', help='''The maximum number of dataitems that a block can contain. 
                                                                  max_block_capacity=0 will disable the dataitem mechanism.''', type=int)
    dataitem_setting.add_argument('--dataitem_size', help='The size of each dataitem in MB.', type=int)
    dataitem_setting.add_argument('--dataitem_input_interval', help='''The interval of dataitem input in rounds.
                                                                       dataitem_input_interval=0 will disable the dataitem queue of each miner.''', type=int)

    parser.add_argument('--result_path',help='The path to output results', type=str)
    parser.add_argument('--disable_post_verification', action='store_true',
                        help='Disable post verification of the longest chain in the global chain right after the main loop in exec().')

    # Batch run options
    batch_setting = parser.add_argument_group('BatchSettings', 'Batch simulation settings (also supported in ini under [BatchSettings])')
    batch_setting.add_argument('--batch_enable', action='store_true', help='Enable batch simulations with parameter sweeps.')
    batch_setting.add_argument('--mlist', help='List of voter chain counts, e.g., "[5,10,20]" or "5,10,20".', type=str)
    batch_setting.add_argument('--nlist', help='List of miner counts (n), e.g., "[10,20]" or "10,20".', type=str)
    batch_setting.add_argument('--Nlist', help='List of optimization dimensions (N), e.g., "[2,5]" or "2,5".', type=str)
    batch_setting.add_argument('--task_numlist', help='List of task numbers to sweep, e.g., "[1,8]".', type=str)
    batch_setting.add_argument('--confirm_depths', help='List of confirmation depths (consensus_param.N).', type=str)
    batch_setting.add_argument('--adver_nums', help='List of adversary counts to sweep.', type=str)
    batch_setting.add_argument('--retry_list', help='List of retry_times for optimization (global_var.repeat_times).', type=str)
    batch_setting.add_argument('--yuzhis', help='List of threshold base values to sweep.', type=str)
    batch_setting.add_argument('--yuzhi_changes', help='List of threshold change rates to sweep.', type=str)
    batch_setting.add_argument('--iterations', help='Repeat times per parameter combo.', type=int)
    batch_setting.add_argument('--max_cores', help='Max cores for BLAS/OpenMP backends during batch.', type=int)
    batch_setting.add_argument('--optimization_method', help='List of optimization methods, e.g., "[kkt,threshold]" or "kkt,threshold".', type=str)

    args = vars(parser.parse_args())
    args['result_path'] = args['result_path'] and Path(args['result_path'])
    if difficulty := args.get('difficulty'):
        target_bin = int('F'*64, 16) >> difficulty
        args['target'] = f"{target_bin:0>64X}"
    else:
        args['target'] = None

    if args.get('adver_lists') is not None:
        args['adver_lists'] = eval(args.get('adver_lists', 'None'))
    args['attack_arg'] = {}
    if args.get('Ng') is not None:
        args['attack_arg']['Ng'] = args.get('Ng')
    if args.get('eclipse_target') is not None:
        args['attack_arg']['eclipse_target'] = eval(args.get('eclipse_target'))

    main(**args)
