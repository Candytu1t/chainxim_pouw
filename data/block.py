import copy
from itertools import chain
from abc import ABCMeta, abstractmethod

from functions import BYTE_ORDER, INT_LEN, hash_bytes

from .message import Message


class BlockHead(metaclass=ABCMeta):
    __slots__ = ['prehash', 'timestamp', 'content', 'miner', 'parentMT_root', 'contentMT_root', 'pouw_opt_x', 'pouw_opt_y']
    __omit_keys = {} # The items to omit when printing the object
    
    def __init__(self, prehash=b'', timestamp=0, content=b'', Miner=-1, parentMT_root='', contentMT_root='',
                 pouw_opt_x:list=[], pouw_opt_y=0):
        self.prehash:bytes = prehash  # 前一个区块的hash
        self.timestamp:int = timestamp  # 时间戳
        self.content = content  # 内容 - 可以是bytes或int
        self.miner:int = Miner  # 矿工
        self.parentMT_root:str = parentMT_root  # 父区块默克尔树根
        self.contentMT_root:str = contentMT_root  # 内容默克尔树根
        self.pouw_opt_x:list = pouw_opt_x  # PoUW优化变量x
        self.pouw_opt_y:float = pouw_opt_y  # PoUW优化变量y
    
    @abstractmethod
    def calculate_blockhash(self) -> bytes:
        '''
        计算区块的hash（基础类：仅用于最简区块头；共识具体实现会覆盖）
        保持对content类型的兼容（支持bytes/int/str/list/dict等）
        '''
        import json

        def _content_to_bytes(content) -> bytes:
            if content is None:
                return (0).to_bytes(INT_LEN, BYTE_ORDER, signed=True)
            if isinstance(content, (bytes, bytearray)):
                return bytes(content)
            if isinstance(content, int):
                return content.to_bytes(INT_LEN, BYTE_ORDER, signed=True)
            if isinstance(content, str):
                return content.encode('utf-8')
            try:
                import numpy as _np
                if isinstance(content, _np.ndarray):
                    content = content.tolist()
            except Exception:
                pass
            try:
                s = json.dumps(content, sort_keys=True, separators=(',', ':'))
                return s.encode('utf-8')
            except Exception:
                return repr(content).encode('utf-8')

        content_bytes = _content_to_bytes(self.content)
        data_bytes = self.miner.to_bytes(INT_LEN, BYTE_ORDER, signed=True) + \
                     content_bytes + \
                     self.prehash
        return hash_bytes(data_bytes).digest()

    def __repr__(self) -> str:
        bhlist = []
        keys = chain.from_iterable(list(getattr(s, '__slots__', []) for s in self.__class__.__mro__) + list(getattr(self, '__dict__', {}).keys()))
        for k in keys:
            if k not in self.__omit_keys:
                v = getattr(self, k)
                bhlist.append(k + ': ' + (str(v) if not isinstance(v, bytes) else v.hex()))
        return '\n'.join(bhlist)


class Block(Message):
    __slots__ = ['__blockhead', 'height', 'blockhash', 'isAdversaryBlock', 'next', 'parentblock', 'isGenesis', 
                 'index', 'vtonprp', 'txpool', 'prppool', 'txblock', 'prpblock']
    __omit_keys = {'segment_num'} # The items to omit when printing the object

    def __init__(self, name=None, blockhead: BlockHead = None, height=None, 
                 isadversary=False, isgenesis=False, blocksize_MB=2, index=None, 
                 vtonprp:list=[], txpool:list=[], prppool:list=[]):
        super().__init__(name, blocksize_MB)
        self.__blockhead = blockhead
        self.height = height
        self.blockhash = blockhead.calculate_blockhash() if blockhead else b''
        self.isAdversaryBlock = isadversary
        self.next = []  # 子块列表
        self.parentblock:Block = None  # 母块
        self.isGenesis = isgenesis
        # PoUW相关属性
        self.index = index  # 段索引 (0=tx, 1=prp, 2+=vt)
        self.vtonprp = vtonprp  # 投票对提案的引用
        self.txpool = txpool  # 交易池
        self.prppool = prppool  # 提案池
        self.txblock = []  # 交易块列表
        self.prpblock = []  # 提案块列表
        # 单位:MB 随机 0.5~1 MB
        
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        slots = chain.from_iterable((getattr(s, '__slots__', [])) for s in self.__class__.__mro__)
        for k in slots:
            if cls.__name__ == 'Block':
                if k == 'next':
                    setattr(result, k, [])
                    continue
                if k == 'parentblock':
                    setattr(result, k, None)
                    continue
            if k == '__name':
                key = '_Message__name'
            elif k == '__blockhead':
                key = '_Block__blockhead'
            else:
                key = k
            setattr(result, key, copy.deepcopy(getattr(self, key), memo))

        if var_dict := getattr(self, '__dict__', None):
            for k, v in var_dict.items():
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    
    def __repr__(self) -> str:
        def _formatter(d, mplus=1):
            m = max(map(len, list(d.keys()))) + mplus
            s = '\n'.join([k.rjust(m) + ': ' + 
                           _indenter(str(v) if not isinstance(v, bytes) 
                           else v.hex(), m+2) for k, v in d.items()])
            return s
        def _indenter(s, n=0):
            split = s.split("\n")
            indent = " "*n
            return ("\n" + indent).join(split)
        
        slots = chain.from_iterable(getattr(s, '__slots__', []) for s in self.__class__.__mro__)
        var_dict = {}
        for k in slots:
            if k == '__name':
                key = '_Message__name'
            elif k == '__blockhead':
                key = '_Block__blockhead'
            else:
                key = k
            var_dict[k] = getattr(self, key)

        if hasattr(self, '__dict__'):
            var_dict.update(self.__dict__)
        var_dict.update({'next': [b.name for b in self.next if self.next], 
                      'parentblock': self.parentblock.name if self.parentblock is not None else None})
        for omk in self.__omit_keys:
            if omk in var_dict:
                del var_dict[omk]
        return '\n'+ _formatter(var_dict)

    @property
    def blockhead(self):
        return self.__blockhead

    def calculate_blockhash(self):
        self.blockhash = self.blockhead.calculate_blockhash()
        return self.blockhash

    def get_height(self):
        return self.height