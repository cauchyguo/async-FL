import math
import numpy as np
import random

from schedule.AbstractSchedule import AbstractSchedule
from utils.GlobalVarGetter import GlobalVarGetter

class TiFLSchedule(AbstractSchedule):
    def __init__(self, config):
        super().__init__(config)
    
    def schedule(self, client_list):
        """
        在TiFL算法中，我们选择每个层级中的所有客户端
        
        Args:
            client_list: 当前层级中的客户端列表
            
        Returns:
            选中的客户端列表
        """
        # 在TiFL中，我们选择层级中的所有客户端参与训练
        return client_list
