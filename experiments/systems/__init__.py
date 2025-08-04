#!/usr/bin/env python3
"""
系统实现模块 - 四个对比系统的实现

包含：
- SystemA: 基线串行系统
- SystemB: KV缓存预填充系统（本文方案）
- SystemC: 理想化端到端系统
- SystemA': 仅流式ASR系统（消融研究用）
"""

from .system_a_baseline import SystemA_BaselineSequential
from .system_b_proposed import SystemB_ProposedKVCache
from .system_c_endtoend import SystemC_EndToEndOracle
from .system_a_prime import SystemA_Prime_StreamingASROnly

__all__ = [
    'SystemA_BaselineSequential',
    'SystemB_ProposedKVCache', 
    'SystemC_EndToEndOracle',
    'SystemA_Prime_StreamingASROnly'
]