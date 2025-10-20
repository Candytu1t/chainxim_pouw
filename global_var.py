'''
    全局变量
'''
import logging
import time
from pathlib import Path


def __init__(result_path:Path = None): 
    # current_time = time.strftime("%Y%m%d-%H%M%S")
    RESULT_FOLDER = Path.cwd() / 'Results' / time.strftime("%Y%m%d")
    RESULT_FOLDER.mkdir(parents=True,exist_ok = True)   
    RESULT_PATH=result_path or RESULT_FOLDER  / time.strftime("%H%M%S")
    RESULT_PATH.mkdir(parents=True,exist_ok = True)   
    NET_RESULT_PATH=RESULT_PATH / 'Network Results'
    NET_RESULT_PATH.mkdir(parents=True,exist_ok = True)
    CHAIN_DATA_PATH=RESULT_PATH / 'Chain Data'
    CHAIN_DATA_PATH.mkdir(parents=True,exist_ok = True)
    ATTACK_RESULT_PATH=RESULT_PATH / 'Attack Result'
    ATTACK_RESULT_PATH.mkdir(parents=True,exist_ok = True)
    '''
    初始化
    '''
    global _var_dict
    _var_dict = {}
    _var_dict['MINER_NUM']=0
    _var_dict['POW_TARGET']=''
    _var_dict['AVE_Q']=0
    _var_dict['CONSENSUS_TYPE']='consensus.PoW'
    _var_dict['NETWORK_TYPE']='network.FullConnectedNetwork'
    _var_dict['BLOCK_NUMBER'] = 0
    _var_dict['RESULT_PATH'] = RESULT_PATH
    _var_dict['NET_RESULT_PATH'] = NET_RESULT_PATH
    _var_dict['ATTACK_RESULT_PATH'] = ATTACK_RESULT_PATH
    _var_dict['CHAIN_DATA_PATH'] = CHAIN_DATA_PATH
    _var_dict['BLOCKSIZE'] = 2
    _var_dict['SEGMENTSIZE'] = 0
    _var_dict['LOG_LEVEL'] = logging.INFO
    _var_dict['Show_Fig'] = False
    _var_dict['COMPACT_OUTPUT'] = True
    _var_dict['ATTACK_EXECUTE_TYPE']='execute_sample0'
    _var_dict['CHECK_POINT'] = None
    _var_dict['COMMON_PREFIX_ENABLE'] = False
    # PoUW相关变量
    _var_dict['VTNUM'] = 20  # Voter链数量
    _var_dict['N'] = 2  # 优化问题维度
    _var_dict['fanwei'] = 20  # 搜索范围
    _var_dict['number'] = 1  # 当前矿工关注的链ID
    _var_dict['task_num'] = 1  # 任务类型
    _var_dict['task_yuzhi'] = 0.001  # 优化阈值
    _var_dict['repeat_times'] = 10  # 重试次数
    _var_dict['kkt_yuzhi'] = 0.001  # KKT条件阈值
    _var_dict['OPT_RESULT'] = []  # 优化结果
    _var_dict['OPT_RESULT_pure'] = []  # 纯PoW优化结果
    _var_dict['unref_tx'] = []  # 未引用的交易块
    _var_dict['unref_prp'] = []  # 未引用的提案块
    _var_dict['VOTE_COUNT'] = {}  # 投票计数
    _var_dict['ARRIVE_TIME_TX'] = {}  # 交易块到达时间
    _var_dict['ARRIVE_TIME_PRP'] = {}  # 提案块到达时间
    _var_dict['TXPOOL'] = []  # 交易池
    _var_dict['TX_NUM'] = 0  # 交易数量
    _var_dict['TX_NUM_HO'] = 0  # 诚实交易数量
    _var_dict['Throughput count'] = {}  # 吞吐量计数
    _var_dict['COM_TIME'] = {}  # 完成时间
    _var_dict['COM_NUM'] = 0  # 完成数量
    _var_dict['start_TIME'] = {}  # 开始时间
    _var_dict['vt_optimal_y'] = {}  # 投票链最优值
    _var_dict['vt_optimal_arrivetime'] = {}  # 投票链最优到达时间
    _var_dict['tasks'] = []  # 任务列表
    _var_dict['speed'] = 0  # 速度
    _var_dict['distribution'] = []  # 分布数据
    _var_dict['distribution_minerid'] = []  # 分布矿工ID
    _var_dict['distribution_suc'] = []  # 成功分布
    _var_dict['distribution_suc_minerid'] = []  # 成功分布矿工ID
    _var_dict['distribution_suc_round'] = []  # 成功分布轮次
    _var_dict['distribution_suc_pure'] = []  # 纯PoW成功分布
    _var_dict['distribution_suc_pure_minerid'] = []  # 纯PoW成功分布矿工ID
    _var_dict['distribution_suc_round_pure'] = []  # 纯PoW成功分布轮次
    _var_dict['distribution_round'] = []  # 分布轮次
    _var_dict['depth'] = 1  # 确认深度（默认1）
    _var_dict['minority_block'] = None  # 少数块
    _var_dict['majority_block'] = None  # 多数块
    _var_dict['miner_in_miner'] = []  # 矿工进入矿工
    _var_dict['miner_in_round'] = []  # 矿工进入轮次
    _var_dict['miner_out_miner'] = []  # 矿工退出矿工
    _var_dict['miner_out_round'] = []  # 矿工退出轮次
    _var_dict['work_time_opt'] = []  # 优化工作时间
    _var_dict['work_time_opt_suc'] = []  # 成功优化工作时间
    _var_dict['work_time_pow'] = []  # PoW工作时间
    _var_dict['work_time_pow_suc'] = []  # 成功PoW工作时间
    _var_dict['PERFORMANCE_ANALYZER_ENABLE'] = True  # 是否启用性能分析器
    _var_dict['MEMORY_CLEANUP_ENABLE'] = True  # 是否启用内存清理
    
    # 初始化阈值方法相关变量
    _var_dict['optimization_method'] = 'kkt'  # 'kkt' 或 'threshold'
    _var_dict['yuzhi'] = 4000  # 基础阈值
    _var_dict['yuzhi_change'] = 0.8  # 阈值变化率
    _var_dict['yuzhi_min'] = 4000  # 最小阈值

def set_common_prefix_enable(common_prefix_enable):
    '''设置是否启用common prefix pdf type:bool'''
    _var_dict['COMMON_PREFIX_ENABLE'] = common_prefix_enable

def get_common_prefix_enable():
    '''是否启用common prefix pdf计算'''
    return _var_dict['COMMON_PREFIX_ENABLE']

def set_log_level(log_level):
    '''设置日志级别'''
    _var_dict['LOG_LEVEL'] = log_level

def get_log_level():
    '''获得日志级别'''
    return _var_dict['LOG_LEVEL']

def set_consensus_type(consensus_type):
    '''定义共识协议类型 type:str'''
    _var_dict['CONSENSUS_TYPE'] = consensus_type
def get_consensus_type():
    '''获得共识协议类型'''
    return _var_dict['CONSENSUS_TYPE']

def set_miner_num(miner_num):
    '''定义矿工数量 type:int'''
    _var_dict['MINER_NUM'] = miner_num
def get_miner_num():
    '''获得矿工数量'''
    return _var_dict['MINER_NUM']

def set_PoW_target(PoW_target):
    '''定义pow目标 type:str'''
    _var_dict['POW_TARGET'] = PoW_target
def get_PoW_target():
    '''获得pow目标'''
    return _var_dict['POW_TARGET']

def set_ave_q(ave_q):
    '''定义pow,每round最多hash计算次数 type:int'''
    _var_dict['AVE_Q'] = ave_q
def get_ave_q():
    '''获得pow,每round最多hash计算次数'''
    return _var_dict['AVE_Q']

def get_block_number():
    '''获得产生区块的独立编号'''
    _var_dict['BLOCK_NUMBER'] = _var_dict['BLOCK_NUMBER'] + 1
    return _var_dict['BLOCK_NUMBER']

def get_result_path():
    return _var_dict['RESULT_PATH']

def get_net_result_path():
    return _var_dict['NET_RESULT_PATH']

def get_chain_data_path():
    return _var_dict['CHAIN_DATA_PATH']

def get_attack_result_path():
    return _var_dict['ATTACK_RESULT_PATH']

def set_network_type(network_type):
    '''定义网络类型 type:str'''
    _var_dict['NETWORK_TYPE'] = network_type
def get_network_type():
    '''获得网络类型'''
    return _var_dict['NETWORK_TYPE']

def set_blocksize(blocksize):
    _var_dict['BLOCKSIZE'] = blocksize

def get_blocksize():
    return _var_dict['BLOCKSIZE']

def set_segmentsize(segmentsize):
    _var_dict['SEGMENTSIZE'] = segmentsize

def get_segmentsize():
    return _var_dict['SEGMENTSIZE']

def set_show_fig(show_fig):
    _var_dict['Show_Fig'] = show_fig

def get_show_fig():
    return _var_dict['Show_Fig']

def set_compact_outputfile(compact_outputfile):
    _var_dict['COMPACT_OUTPUT'] = compact_outputfile

def get_compact_outputfile():
    return _var_dict['COMPACT_OUTPUT']

def set_attack_execute_type(attack_execute_type):
    '''定义攻击类型 type:str'''
    _var_dict['ATTACK_EXECUTE_TYPE'] = attack_execute_type

def get_attack_execute_type():
    '''定义攻击类型'''
    return _var_dict['ATTACK_EXECUTE_TYPE']


# PoUW相关函数
def get_vtnum():
    '''获得投票链数量'''
    return _var_dict['VTNUM']

def set_vtnum(n):
    '''设置投票链数量'''
    _var_dict['VTNUM'] = n

def get_N():
    '''获得优化问题维度'''
    return _var_dict['N']

def set_N(n):
    '''设置优化问题维度'''
    _var_dict['N'] = n

def get_fanwei():
    '''获得搜索范围'''
    return _var_dict['fanwei']

def set_fanwei(x):
    '''设置搜索范围'''
    _var_dict['fanwei'] = x

def get_number():
    '''获得当前矿工关注的链ID'''
    return _var_dict['number']

def set_number(x):
    '''设置当前矿工关注的链ID'''
    _var_dict['number'] = x

def get_task_num():
    '''获得任务类型'''
    return _var_dict['task_num']

def set_task_num(x):
    '''设置任务类型'''
    _var_dict['task_num'] = x

def get_yuzhi():
    '''获得优化阈值'''
    return _var_dict['task_yuzhi']

def set_yuzhi(x):
    '''设置优化阈值'''
    _var_dict['task_yuzhi'] = x

def get_repeat_time():
    '''获得重试次数'''
    return _var_dict['repeat_times']

def set_repeat_time(r):
    '''设置重试次数'''
    _var_dict['repeat_times'] = r

def get_kkt_yuzhi():
    '''获得KKT条件阈值'''
    return _var_dict['kkt_yuzhi']

def set_kkt_yuzhi(x):
    '''设置KKT条件阈值'''
    _var_dict['kkt_yuzhi'] = x

def add_opt_result(x):
    '''添加优化结果'''
    _var_dict['OPT_RESULT'].append(x)

def get_opt_result():
    '''获得优化结果'''
    return _var_dict['OPT_RESULT']

def add_opt_result_pure(x):
    '''添加纯PoW优化结果'''
    _var_dict['OPT_RESULT_pure'].append(x)

def get_opt_result_pure():
    '''获得纯PoW优化结果'''
    return _var_dict['OPT_RESULT_pure']

def get_txnum():
    '''获得交易数量'''
    return _var_dict['TX_NUM']

def rec_tx():
    '''记录交易'''
    _var_dict['TX_NUM'] = _var_dict['TX_NUM'] + 1

def get_minority_block():
    '''获得少数块'''
    return _var_dict['minority_block']

def set_minority_block(block):
    '''设置少数块'''
    _var_dict['minority_block'] = block

def get_majority_block():
    '''获得多数块'''
    return _var_dict['majority_block']

def set_majority_block(block):
    '''设置多数块'''
    _var_dict['majority_block'] = block

def set_miner_in(round_num, miner):
    '''设置矿工进入'''
    _var_dict['miner_in_miner'].append(miner)
    _var_dict['miner_in_round'].append(round_num)

def get_miner_in_miner():
    '''获得进入的矿工'''
    return _var_dict['miner_in_miner']

def get_miner_in_round():
    '''获得进入的轮次'''
    return _var_dict['miner_in_round']

def set_miner_out(round_num, miner):
    '''设置矿工退出'''
    _var_dict['miner_out_miner'].append(miner)
    _var_dict['miner_out_round'].append(round_num)

def get_miner_out_miner():
    '''获得退出的矿工'''
    return _var_dict['miner_out_miner']

def get_miner_out_round():
    '''获得退出的轮次'''
    return _var_dict['miner_out_round']

def set_dis(x):
    '''设置分布数据'''
    _var_dict['distribution'].append(x)

def get_dis():
    '''获得分布数据'''
    return _var_dict['distribution']

def set_dis_minerid(x):
    '''设置分布矿工ID'''
    _var_dict['distribution_minerid'].append(x)

def get_dis_minerid():
    '''获得分布矿工ID'''
    return _var_dict['distribution_minerid']

def set_dis_round(x):
    '''设置分布轮次'''
    _var_dict['distribution_round'].append(x)

def get_dis_round():
    '''获得分布轮次'''
    return _var_dict['distribution_round']

def set_dis_suc(x):
    '''设置成功分布'''
    _var_dict['distribution_suc'].append(x)

def get_dis_suc():
    '''获得成功分布'''
    return _var_dict['distribution_suc']

def set_dis_suc_minerid(x):
    '''设置成功分布矿工ID'''
    _var_dict['distribution_suc_minerid'].append(x)

def get_dis_suc_minerid():
    '''获得成功分布矿工ID'''
    return _var_dict['distribution_suc_minerid']

def set_dis_suc_round(x):
    '''设置成功分布轮次'''
    _var_dict['distribution_suc_round'].append(x)

def get_dis_suc_round():
    '''获得成功分布轮次'''
    return _var_dict['distribution_suc_round']

def set_dis_suc_pure(x):
    '''设置纯PoW成功分布'''
    _var_dict['distribution_suc_pure'].append(x)

def get_dis_suc_pure():
    '''获得纯PoW成功分布'''
    return _var_dict['distribution_suc_pure']

def set_dis_suc_pure_minerid(x):
    '''设置纯PoW成功分布矿工ID'''
    _var_dict['distribution_suc_pure_minerid'].append(x)

def get_dis_suc_pure_minerid():
    '''获得纯PoW成功分布矿工ID'''
    return _var_dict['distribution_suc_pure_minerid']

def set_dis_suc_round_pure(x):
    '''设置纯PoW成功分布轮次'''
    _var_dict['distribution_suc_round_pure'].append(x)

def get_dis_suc_round_pure():
    '''获得纯PoW成功分布轮次'''
    return _var_dict['distribution_suc_round_pure']

def set_depth(d):
    '''设置确认深度'''
    _var_dict['depth'] = d

def get_depth():
    '''获取确认深度'''
    return _var_dict.get('depth', 1)

def add_time_opt(x):
    '''添加优化工作时间'''
    _var_dict['work_time_opt'].append(x)

def get_time_opt():
    '''获得优化工作时间'''
    return _var_dict['work_time_opt']

def add_time_opt_suc(x):
    '''添加成功优化工作时间'''
    _var_dict['work_time_opt_suc'].append(x)

def get_time_opt_suc():
    '''获得成功优化工作时间'''
    return _var_dict['work_time_opt_suc']

def add_time_pow(x):
    '''添加PoW工作时间'''
    _var_dict['work_time_pow'].append(x)

def get_time_pow():
    '''获得PoW工作时间'''
    return _var_dict['work_time_pow']

def add_time_pow_suc(x):
    '''添加成功PoW工作时间'''
    _var_dict['work_time_pow_suc'].append(x)

def get_time_pow_suc():
    '''获得成功PoW工作时间'''
    return _var_dict['work_time_pow_suc']

def get_start_time():
    '''获得开始时间'''
    return _var_dict['start_TIME']

def set_start_time(block_name, round_num):
    '''设置开始时间'''
    _var_dict['start_TIME'][block_name] = round_num

def get_com_time():
    '''获得完成时间'''
    return _var_dict['COM_TIME']

def set_com_time(block_name, time):
    '''设置完成时间'''
    _var_dict['COM_TIME'][block_name] = time

def get_tasks():
    '''获得任务列表'''
    return _var_dict['tasks']

def set_tasks(tasks):
    '''设置任务列表'''
    _var_dict['tasks'] = tasks

def get_vtcount():
    '''获得投票计数'''
    return _var_dict['VOTE_COUNT']

def set_vtcount(vote_count):
    '''设置投票计数'''
    _var_dict['VOTE_COUNT'] = vote_count

def write_vtcount(vote_count):
    '''写入投票计数'''
    _var_dict['VOTE_COUNT'].update(vote_count)

def set_performance_analyzer_enable(enable):
    '''设置是否启用性能分析器'''
    _var_dict['PERFORMANCE_ANALYZER_ENABLE'] = enable

def get_performance_analyzer_enable():
    '''获得是否启用性能分析器'''
    return _var_dict['PERFORMANCE_ANALYZER_ENABLE']

def set_memory_cleanup_enable(enable):
    '''设置是否启用内存清理'''
    _var_dict['MEMORY_CLEANUP_ENABLE'] = enable

def get_memory_cleanup_enable():
    '''获得是否启用内存清理'''
    return _var_dict['MEMORY_CLEANUP_ENABLE']

# ====== 任务系统相关函数 (从chain-xim-master-kkt移植) ======

def set_arrive_time_tx(block_name, round):
    """更新交易区块到达的轮次"""
    _var_dict['ARRIVE_TIME_TX'][block_name] = round

def set_arrive_time_prp(block_name, round):
    """更新提案区块到达的轮次"""
    _var_dict['ARRIVE_TIME_PRP'][block_name] = round

def get_txnum():
    """获取交易数量"""
    return _var_dict['TX_NUM']

def set_txho(txho):
    """设置诚实交易数量"""
    _var_dict['TX_NUM_HO'] = txho

def add_tx(tx):
    """添加交易到交易池"""
    _var_dict['TXPOOL'].append(tx)

def get_txpool():
    """获取交易池"""
    return _var_dict['TXPOOL']

def rec_tx():
    """增加交易计数"""
    _var_dict['TX_NUM'] = _var_dict['TX_NUM'] + 1

def ADD_COMNUM():
    """增加完成数量计数"""
    _var_dict['COM_NUM'] = _var_dict['COM_NUM'] + 1

def get_comnum():
    """获取完成数量"""
    return _var_dict['COM_NUM']

def get_optimal(chain_id):
    """获取投票链的最优值"""
    return _var_dict['vt_optimal_y'].get(chain_id, 0)

def set_optimal(chain_id, y):
    """设置投票链的最优值"""
    _var_dict['vt_optimal_y'][chain_id] = y

def get_opt_time(chain_id):
    """获取投票链的最优到达时间"""
    return _var_dict['vt_optimal_arrivetime'].get(chain_id, {})

def set_opt_time(chain_id, y, time):
    """设置投票链的最优到达时间"""
    if chain_id not in _var_dict['vt_optimal_arrivetime']:
        _var_dict['vt_optimal_arrivetime'][chain_id] = {}
    _var_dict['vt_optimal_arrivetime'][chain_id][y] = time

def get_ar_time_tx():
    """获取交易区块到达时间"""
    return _var_dict['ARRIVE_TIME_TX']

def get_ar_time_prp():
    """获取提案区块到达时间"""
    return _var_dict['ARRIVE_TIME_PRP']

def set_com_time(block_name, round):
    """设置完成时间"""
    _var_dict['COM_TIME'][block_name] = round

def get_com_time():
    """获取完成时间"""
    return _var_dict['COM_TIME']

def set_start_time(block_name, round):
    """设置开始时间"""
    _var_dict['start_TIME'][block_name] = round

def get_start_time():
    """获取开始时间"""
    return _var_dict['start_TIME']

# ============= 阈值方法相关变量 =============

def set_optimization_method(method):
    """设置优化方法: 'kkt' 或 'threshold'"""
    _var_dict['optimization_method'] = method

def get_optimization_method():
    """获取优化方法"""
    return _var_dict.get('optimization_method', 'kkt')

def set_yuzhi(yuzhi):
    """设置基础阈值"""
    _var_dict['yuzhi'] = yuzhi

def get_yuzhi():
    """获取基础阈值"""
    return _var_dict.get('yuzhi', 4000)

def set_yuzhi_min(yuzhi_min):
    """设置最小阈值"""
    _var_dict['yuzhi_min'] = yuzhi_min

def get_yuzhi_min():
    """获取最小阈值"""
    return _var_dict.get('yuzhi_min', 4000)

def set_yuzhi_change(yuzhi_change):
    """设置阈值变化率"""
    _var_dict['yuzhi_change'] = yuzhi_change

def get_yuzhi_change():
    """获取阈值变化率"""
    return _var_dict.get('yuzhi_change', 0.8)

def calculate_dynamic_threshold(height):
    """计算动态阈值（随高度变化）"""
    yuzhi = get_yuzhi()
    yuzhi_change = get_yuzhi_change()
    yuzhi_min = get_yuzhi_min()
    
    dynamic_yuzhi = yuzhi * (yuzhi_change ** height)
    if dynamic_yuzhi < yuzhi_min:
        dynamic_yuzhi = yuzhi_min
    
    return dynamic_yuzhi