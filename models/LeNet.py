import torch.nn as nn
import torch.nn.functional as func
from quantize import QuantConv2d,QuantLinear
import xlrd
import numpy as np
'''
#导入近似乘法器查找表
def import_excel_matrix(path):
  table = xlrd.open_workbook(path).sheets()[0] # 获取第一个sheet表
  row = table.nrows # 行数
  col = table.ncols # 列数
  datamatrix = [] # 生成一个nrows行*ncols列的初始矩阵
  for i in range(col): # 对列进行遍历
    cols = table.row_values(i) # 把list转换为矩阵进行矩阵操作
    datamatrix.append(cols)# 按列把数据存进矩阵中
  return datamatrix
data_file = u'./data/mul_app_8x8_350.xlsx' # Excel文件存储位置
#data_file = 'C:\\Users\\chonghang\\Desktop\\retrain\\data\\mul_acc_8x8.xlsx' # Excel文件存储位置
mul_lut = import_excel_matrix(data_file)
mul_lut = np.array(mul_lut)
mul_lut = np.array(mul_lut).astype(np.float32)
'''
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        # self.conv1 = QuantConv2d(3, 6, 5, mul_lut, stride=1, padding=2, a_bits=8, w_bits=8, q_type=0, q_level=1,
        #             weight_observer=1, flag=1),
        # self.conv2 = QuantConv2d(6, 16, 5, mul_lut, stride=1, padding=2, a_bits=8, w_bits=8, q_type=0, q_level=1,
        #             weight_observer=1, flag=1),
        # self.fc1 = QuantLinear(400,120,mul_lut,a_bits=8, w_bits=8, q_type=0, q_level=1,weight_observer=1,flag=1),#flag = 1时使用近似乘法器， flag=0时使用精确乘法器
        # self.fc2 = QuantLinear(120,84,mul_lut,a_bits=8, w_bits=8, q_type=0, q_level=1,weight_observer=1,flag=1),#flag = 1时使用近似乘法器， flag=0时使用精确乘法器
        # self.fc3 = QuantLinear(84,10,mul_lut,a_bits=8, w_bits=8, q_type=0, q_level=1,weight_observer=1,flag=1),#flag = 1时使用近似乘法器， flag=0时使用精确乘法器
    def forward(self, x):
        x = func.relu(self.conv1(x))
        x = func.max_pool2d(x, 2)
        x = func.relu(self.conv2(x))
        x = func.max_pool2d(x, 2)
        x = x.contiguous().view(x.size(0), -1)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x
