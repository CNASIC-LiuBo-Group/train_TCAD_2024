import numpy as np
import torch
import torchvision
import xlrd

from torchvision import transforms as transforms
from models import *
from torch import optim
from misc import progress_bar


# 导入近似乘法器LUT
from models.ResNet import *
from models.ResNet import resnet18_quant
from models.resnet_layer import resnet18_layer


def import_Lut(path):
    fin = open(path,'r')
    Lut = []
    Lut_string = fin.readlines()
    for lines in Lut_string:
        temp = lines.split(',')
        lines = [float(x) for x in temp]
        Lut.append(lines)
    fin.close()
    np_type = np.array(Lut).astype(np.float32)
    return np_type

    
def import_excel_matrix(path):
    table = xlrd.open_workbook(path).sheets()[0]  # 获取第一个sheet表
    row = table.nrows  # 行数
    col = table.ncols  # 列数
    datamatrix = []  # 生成一个nrows行*ncols列的初始矩阵
    for i in range(col):  # 对列进行遍历
        cols = table.row_values(i)  # 把list转换为矩阵进行矩阵操作
        datamatrix.append(cols)  # 按列把数据存进矩阵中
    return np.array(datamatrix).astype(np.float32)


# Training

# @profile
def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()

    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))


#######################设置超参数##################################################
batch_size = 128  # 批的大小
learning_rate = 1e-2  # 学习率
num_epoches = 200  # 遍历训练集的次数
image_size = 32  # 数据增强 图形尺寸


quant_setting = [8,
                 8, 8, 8, 8,
                 8, 8, 8, 8, 8,
                 8, 8, 8, 8, 8,
                 8, 8, 8, 8, 8,
                 8]

quant_setting_str = 'quant_set_'

for i in range(21):
    quant_setting_str = quant_setting_str + str(quant_setting[i])

app_setting = [1,
               1, 1, 1, 1,
               1, 1, 1, 1, 1,
               1, 1, 1, 1, 1,
               1, 1, 1, 1, 1,
               1]

app_setting_str = 'app_set_'
for i in range(21):
    app_setting_str = app_setting_str + str(app_setting[i])

mul_lut_set = []

mul_lut_app_path = './data/Net_Mat2.txt'
mul_lut_app = import_Lut(mul_lut_app_path)
#mul_lut_app = import_excel_matrixmul_lut_app_path)
mul_lut_acc = import_Lut('./data/Mul8-signed_acc_mat.txt')

for i in range(21):
#    if i in {1,2,3,4,7,8,9,12,13,14,17,18,19}:
    mul_lut_set.append(mul_lut_app)
#    else:
#        mul_lut_set.append(mul_lut_acc)


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
print(mul_lut_app_path)
print("successfuly load the mul lut")
# 导入数据集
train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
test_transform = transforms.Compose([transforms.ToTensor()])

# augument
data_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='edge'),
    #transforms.Resize((50, 50)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(45),
    transforms.ColorJitter(brightness=0.5),
    transforms.ColorJitter(contrast=0.5),
    transforms.ColorJitter(hue=0.5),
    transforms.ColorJitter(saturation=0.5),
    transforms.RandomGrayscale(p=0.5),  # 随机灰度化
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])  # 增加数据变化

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=data_transform)
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

#model select
model = resnet18()
#device = torch.device("cpu")
print(model)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
model = model.to(device)
# 定义loss和optimizer
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=learning_rate)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
# optimizer = optim.SGD(model.classifier.parameters(), lr=learning_rate)
# 第一次全精度训练模型

for epoch in range(0, num_epoches):
    train(epoch)
    test(epoch)
    scheduler.step()
# torch.save(model, './cifar10_pth/VGG/VGG19/fullacc_train.pth')

torch.save(model, './cifar10_pth/resnet18/fullacc_train_new.pth')


model = torch.load('./cifar10_pth/resnet18/fullacc_train_new.pth')

pretext_model = model.state_dict()


##############################AlexNet############################################################################################
print("baseline accuary test")
test(0)
# model = AlexNet_quant(mul_lut, flag=0)
model = resnet18_quant(mul_lut=mul_lut_set, flag=0, w_bits=quant_setting)
model = model.to(device)
pth_path_quant = './cifar10_pth/resnet18/quant' + str(quant_setting_str) + '.pth'
#model = torch.load(pth_path_quant)
############################################################################################################################


############################################################################################################################
model_dict = model.state_dict()
state_dict = {k: v for k, v in pretext_model.items() if k in model_dict.keys()}  # load same name layer weiget
model_dict.update(state_dict)
model.load_state_dict(model_dict)

num_quant_retrain = 1  # 15
learning_rate = 1e-2  # 学习率
print("quant test befor train")
test(0)
for epoch in range(0, num_quant_retrain):
    train(epoch)
    test(epoch)
    scheduler.step()


torch.save(model, pth_path_quant)

device = torch.device("cpu")          
print(device)
print("run here ****************************************")

model = torch.load(pth_path_quant, map_location='cpu')        

#optimizer = optim.SGD(model.classifier.parameters(), lr=learning_rate)
pretext_model = model.state_dict()
print("another quant test")
test(0)
model = resnet18_quant(mul_lut=mul_lut_set, flag=1, w_bits=quant_setting)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

model_dict = model.state_dict()
state_dict = {k: v for k, v in pretext_model.items() if k in model_dict.keys()}  # load same name layer weiget
model_dict.update(state_dict)
model.load_state_dict(model_dict)

num_app_retrain = 15
test(0)
for epoch in range(0, num_app_retrain):
    train(epoch)
    test(epoch)
    scheduler.step()

torch.save(model, './cifar10_pth/resnet18/quant_bmf_train.pth')
