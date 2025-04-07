
import warnings
warnings.filterwarnings("ignore")

import argparse
import torch
import torch.nn as nn
from bitstring import Bits
import numpy as np
import torch.nn.functional as F
import os
import copy
import config
from utils import *

parser = argparse.ArgumentParser(description='TA-LBF (targeted attack with limited bit-flips)')

parser.add_argument('--gpu-id', '-gpu-id', default="0", type=str)

parser.add_argument('--attack-idx', '-attack_idx', default=9490, type=int)
parser.add_argument('--target-class', '-target_class', default=0, type=int)

parser.add_argument('--lam', '-lam', default=100, type=float)
parser.add_argument('--k', '-k', default=5, type=float)
parser.add_argument('--n-aux', '-n_aux', default=128, type=int)


parser.add_argument('--margin', '-margin', default=10, type=float)
parser.add_argument('--max-search-k', '-max_search_k', default=4, type=int)
parser.add_argument('--max-search-lam', '-max_search_lam', default=8, type=int)
parser.add_argument('--ext-max-iters', '-ext_max_iters', default=2000, type=int)
parser.add_argument('--inn-max-iters', '-inn_max_iters', default=5, type=int)
parser.add_argument('--initial-rho1', '-initial_rho1', default=0.0001, type=float)
parser.add_argument('--initial-rho2', '-initial_rho2', default=0.0001, type=float)
parser.add_argument('--initial-rho3', '-initial_rho3', default=0.00001, type=float)
parser.add_argument('--max-rho1', '-max_rho1', default=50, type=float)
parser.add_argument('--max-rho2', '-max_rho2', default=50, type=float)
parser.add_argument('--max-rho3', '-max_rho3', default=5, type=float)
parser.add_argument('--rho-fact', '-rho_fact', default=1.01, type=float)
parser.add_argument('--inn-lr', '-inn_lr', default=0.001, type=float)
parser.add_argument('--stop-threshold', '-stop_threshold', default=1e-4, type=float)
parser.add_argument('--projection-lp', '-projection_lp', default=2, type=int)

# 用于定义和初始化模型的权重、偏置和其他相关参数
class AugLag(nn.Module):
    def __init__(self, n_bits, w, b, step_size, init=False):
        super(AugLag, self).__init__()
        # 初始化权重、偏置等参数
        # 参数说明：n_bits表示位数，w表示权重，b表示偏置，step_size表示步长
        # 主要用于构造一个量化神经网络的线性层，其权重由多个二进制位构成。
        # n_bits 表示权重的二进制位数。
        # w、b 分别为原始的权重和偏置，step_size 用于缩放权重。
        # 将偏置转换为可学习参数。

        self.n_bits = n_bits
        self.b = nn.Parameter(torch.tensor(b).float(), requires_grad=True)

        # w_twos是一个三维张量，其尺寸为 [输出通道数, 输入通道数, 二进制位数]，用来存储每一位的值（通常是 0 或 1），经过后续的权重组合得到浮点数权重。
        self.w_twos = nn.Parameter(torch.zeros([w.shape[0], w.shape[1], self.n_bits]), requires_grad=True)
        self.step_size = step_size
        self.w = w

        # 构造一个基底向量，其中每个元素表示对应二进制位的权重值。
        # 注意这里将最高位设为负数（符号位），从而实现带符号的二进制表示。
        # self.base 用于在前向传播时将二进制权重转换为对应的浮点值。
        base = [2**i for i in range(self.n_bits-1, -1, -1)]
        base[0] = -base[0]
        self.base = nn.Parameter(torch.tensor([[base]]).float())

        # 如果 init 为 True，则调用 reset_w_twos 方法，将原始的权重 w 转换为对应的二进制表示，并初始化 w_twos。
        if init:
            self.reset_w_twos()

    def forward(self, x):

        # covert w_twos to float
        w = self.w_twos * self.base
        w = torch.sum(w, dim=2) * self.step_size

        # calculate output
        x = F.linear(x, w, self.b)
        # 这里利用二进制的每一位和相应的基底值（乘以 step_size）计算出实际的权重矩阵，然后利用 F.linear 计算线性变换。

        return x
    
    
    # 遍历权重矩阵的每个元素，将其转换为二进制表示（长度为 n_bits），并赋值到 w_twos 中
    def reset_w_twos(self):
        for i in range(self.w.shape[0]):
            for j in range(self.w.shape[1]):
                self.w_twos.data[i][j] += \
                    torch.tensor([int(b) for b in Bits(int=int(self.w[i][j]), length=self.n_bits).bin])

# 将输入 x 中的所有元素限制在 [0, 1] 范围内。
def project_box(x):
    xp = x
    xp[x>1]=1
    xp[x<0]=0

    return xp

# 将 x 投影到一个中心为 1/2 的 Lp 球中。
def project_shifted_Lp_ball(x, p):
    shift_vec = 1/2*np.ones(x.size)
    shift_x = x-shift_vec
    normp_shift = np.linalg.norm(shift_x, p)
    n = x.size
    xp = (n**(1/p)) * shift_x / (2*normp_shift) + shift_vec

    return xp

# 将输入 x 中的负数截断为 0，保证非负。
def project_positive(x):
    xp = np.clip(x, 0, None)
    return xp

# 计算损失函数，包括分类损失和增广拉格朗日乘子项
def loss_func(output, labels, s, t, lam, w, target_thr, source_thr,
              b_ori, k_bits, y1, y2, y3, z1, z2, z3, k, rho1, rho2, rho3):

    # 这里 s 表示原始类别（source），t 表示目标类别（target）。
    # output[-1] 表示模型的最后一层输出。
    # l1_1 保证原始类别的输出不超过某个阈值（source_thr），l1_2 保证目标类别的输出超过某个阈值（target_thr），从而实现对抗攻击的目标。
    l1_1 = torch.max(output[-1][s] - source_thr, torch.tensor(0.0).cuda())
    l1_2 = torch.max(target_thr - output[-1][t], torch.tensor(0.0).cuda())
    l1 = l1_1 + l1_2 # 分类目标项
    # l1 = output[-1][s] - output[-1][t]
    # l1 = torch.max(output[-1][s] - source_thr, 0)[0] + torch.max(target_thr - output[-1][t], 0)[0]

    # print(target_thr, output[-1][t])
    # print(source_thr, output[-1][s])
    l2 = F.cross_entropy(output[:-1], labels[:-1]) # 交叉熵项 用于保证在其他辅助数据上的分类性能不会大幅下降（保持模型的整体准确率）。
    # print(l1.item(), l2.item())

    y1, y2, y3, z1, z2, z3 = torch.tensor(y1).float().cuda(), torch.tensor(y2).float().cuda(), torch.tensor(y3).float().cuda(), \
                             torch.tensor(z1).float().cuda(), torch.tensor(z2).float().cuda(), torch.tensor(z3).float().cuda()

    b_ori = torch.tensor(b_ori).float().cuda()
    b = torch.cat((w[s].view(-1), w[t].view(-1)))

    # l3 项结合了拉格朗日乘子（z1, z2, z3）和变量差值，作用是对约束的松弛惩罚。
    l3 = z1@(b-y1) + z2@(b-y2) + z3*(torch.norm(b - b_ori) ** 2 - k + y3)
    # l4 项则是平方惩罚项，形式为 rho/2 范数平方，用于增强约束的满足程度。
    l4 = (rho1/2) * torch.norm(b - y1) ** 2 + (rho2/2) * torch.norm(b - y2) ** 2 \
         + (rho3/2) * (torch.norm(b - b_ori)**2 - k_bits + y3) ** 2

    return l1 + lam * l2 + l3 + l4

# 实施有限位翻转目标攻击的函数
def attack(auglag_ori, all_data, labels, labels_cuda, clean_output,
           target_idx, target_class, source_class, aux_idx,
           lam, k, args):
    # set parameters
    n_aux = args.n_aux
    lam = lam
    ext_max_iters = args.ext_max_iters
    inn_max_iters = args.inn_max_iters
    initial_rho1 = args.initial_rho1
    initial_rho2 = args.initial_rho2
    initial_rho3 = args.initial_rho3
    max_rho1 = args.max_rho1
    max_rho2 = args.max_rho2
    max_rho3 = args.max_rho3
    rho_fact = args.rho_fact
    k_bits = k
    inn_lr = args.inn_lr
    margin = args.margin
    stop_threshold = args.stop_threshold

    projection_lp = args.projection_lp

    all_idx = np.append(aux_idx, target_idx)

    # 阈值设定
    # 从干净输出中计算一个辅助阈值 sub_max，
    # 然后设定目标阈值（希望目标类别输出超过该阈值）
    # 和源类别阈值（希望源类别输出低于该阈值），
    # 加上一定的 margin 保证间隔。
    sub_max = clean_output[target_idx][[i for i in range(len(clean_output[-1])) if i != source_class]].max()
    target_thr = sub_max + margin
    source_thr = sub_max - margin

    # 初始化 AugLag 模型和变量
    # 复制原始模型
    auglag = copy.deepcopy(auglag_ori) # 模型

    # 提取出源类别和目标类别对应的二进制权重，拼接成向量 b_ori ，并将其作为初始参考
    b_ori_s = auglag.w_twos.data[source_class].view(-1).detach().cpu().numpy()
    b_ori_t = auglag.w_twos.data[target_class].view(-1).detach().cpu().numpy()
    b_ori = np.append(b_ori_s, b_ori_t)
    b_new = b_ori

    # 初始化辅助变量 y
    y1 = b_ori
    y2 = y1
    y3 = 0

    # 初始化拉格朗日乘子
    z1 = np.zeros_like(y1)
    z2 = np.zeros_like(y1)
    z3 = 0

    # 设置惩罚参数
    rho1 = initial_rho1
    rho2 = initial_rho2
    rho3 = initial_rho3

    stop_flag = False
    for ext_iter in range(ext_max_iters):
        # 每次 外层迭代 中首先更新利用投影函数更新辅助变量 y1, y2, y3
        # 确保它们满足各自的约束条件
        y1 = project_box(b_new + z1 / rho1)
        y2 = project_shifted_Lp_ball(b_new + z2 / rho2, projection_lp)
        y3 = project_positive(-np.linalg.norm(b_new - b_ori, ord=2) ** 2 + k_bits - z3 / rho3)

        for inn_iter in range(inn_max_iters):
            # 内层迭代使用反向传播计算梯度，并对目标和源类别对应的二进制权重进行更新，更新步长由 inn_lr 控制。
            input_var = torch.autograd.Variable(all_data[all_idx], volatile=True)
            target_var = torch.autograd.Variable(labels_cuda[all_idx].long(), volatile=True)

            # 计算当前输出及损失
            output = auglag(input_var)
            loss = loss_func(output, target_var, source_class, target_class, lam, auglag.w_twos,
                             target_thr, source_thr,
                             b_ori, k_bits, y1, y2, y3, z1, z2, z3, k_bits, rho1, rho2, rho3)

            loss.backward(retain_graph=True) # 保持计算图，以便多次迭代
            
            # 根据梯度更新目标类别和源类别对应的二进制权重
            auglag.w_twos.data[target_class] = auglag.w_twos.data[target_class] - \
                                               inn_lr * auglag.w_twos.grad.data[target_class]
            auglag.w_twos.data[source_class] = auglag.w_twos.data[source_class] - \
                                               inn_lr * auglag.w_twos.grad.data[source_class]
            auglag.w_twos.grad.zero_()

        b_new_s = auglag.w_twos.data[source_class].view(-1).detach().cpu().numpy()
        b_new_t = auglag.w_twos.data[target_class].view(-1).detach().cpu().numpy()
        b_new = np.append(b_new_s, b_new_t)

        if True in np.isnan(b_new):
            return -1

        # 更新拉格朗日乘子以强化约束惩罚
        z1 = z1 + rho1 * (b_new - y1)
        z2 = z2 + rho2 * (b_new - y2)
        z3 = z3 + rho3 * (np.linalg.norm(b_new - b_ori, ord=2) ** 2 - k_bits + y3)

        # 罚参数 rho1, rho2, rho3 也按 rho_fact 因子逐步增加，直到达到预设的最大值
        rho1 = min(rho_fact * rho1, max_rho1)
        rho2 = min(rho_fact * rho2, max_rho2)
        rho3 = min(rho_fact * rho3, max_rho3)

        temp1 = (np.linalg.norm(b_new - y1)) / max(np.linalg.norm(b_new), 2.2204e-16)
        temp2 = (np.linalg.norm(b_new - y2)) / max(np.linalg.norm(b_new), 2.2204e-16)

        # 每迭代 50 次打印一次信息
        if ext_iter % 50 == 0:
            print('iter: %d, stop_threshold: %.6f, loss: %.4f' % (ext_iter, max(temp1, temp2), loss.item()))

        # ||b_new - y|| / ||b_new|| 低于阈值且迭代次数超过 100 时停止外层迭代
        if max(temp1, temp2) <= stop_threshold and ext_iter > 100:
            print('END iter: %d, stop_threshold: %.6f, loss: %.4f' % (ext_iter, max(temp1, temp2), loss.item()))
            stop_flag = True
            break
    # 直接按照阈值为 0.5 来划分
    # 将连续化的权重（接近于 0 或 1）二值化，得到实际的 bit 翻转结果
    auglag.w_twos.data[auglag.w_twos.data > 0.5] = 1.0
    auglag.w_twos.data[auglag.w_twos.data < 0.5] = 0.0

    output = auglag(all_data)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.squeeze(1)
    # 干净样本的准确率
    pa_acc = len([i for i in range(len(output)) if labels[i] == pred[i] and i != target_idx and i not in aux_idx]) / \
                     (len(labels) - 1 - n_aux)
    # 翻转的 bit 数量
    n_bit = torch.norm(auglag_ori.w_twos.data.view(-1) - auglag.w_twos.data.view(-1), p=0).item()

    ret = {
        "pa_acc": pa_acc,
        "stop": stop_flag,
        "suc": target_class == pred[target_idx].item(), # 目标样本被预测成目标类别 则视为成功
        "n_bit": n_bit
    }
    return ret


def main():
    np.random.seed(512)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    print(args)

    # prepare the data
    print("Prepare data ... ")
    arch = "resnet20_quan"
    bit_length = 8

    weight, bias, step_size = load_model(arch, bit_length)
    all_data, labels = load_data(arch, bit_length)
    labels_cuda = labels.cuda()

    # 初始化 AugLag 模型
    # 使用初始化好的权重进行一次前向传播，得到干净模型的输出，用于后续设置阈值等
    auglag = AugLag(bit_length, weight, bias, step_size, init=True).cuda()

    clean_output = auglag(all_data)

    # 计算模型在所有样本上的预测准确率。
    _, pred = clean_output.cpu().topk(1, 1, True, True)
    clean_output = clean_output.detach().cpu().numpy()
    pred = pred.squeeze(1)
    acc_ori = len([i for i in range(len(pred)) if labels[i] == pred[i]]) / len(labels)

    # 选定攻击样本（attack_idx），确定其原始类别（source_class）和目标类别（target_class）。
    attack_idx = args.attack_idx
    source_class = int(labels[attack_idx])
    target_class = args.target_class

    # 随机选择一些辅助样本（n_aux）用于保持模型整体性能，防止攻击后模型泛化能力下降太多。
    aux_idx = np.random.choice([i for i in range(len(labels)) if i != attack_idx], args.n_aux, replace=False)

    print("Attack Start")
    res = attack(auglag, all_data, labels, labels_cuda, clean_output,
                 attack_idx, target_class, source_class, aux_idx,
                 args.lam, args.k, args)

    if res["suc"]:
        print("END Original_ACC:{0:.4f} PA_ACC:{1:.4f} Success:{2} N_flip:{3:.4f}".format(
            acc_ori*100, res["pa_acc"]*100, res["suc"], res["n_bit"]))
    else:
        print("Fail!")


if __name__ == '__main__':
    main()