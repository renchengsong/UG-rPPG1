'''
  Adapted from here: https://github.com/ZitongYu/PhysFormer/TorchLossComputer.py
  Modifed based on the HR-CNN here: https://github.com/radimspetlik/hr-cnn
'''
import math
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import pdb
import torch.nn as nn
from evaluation.post_process import calculate_hr, calculate_psd
from evaluation.evidential_regression import evidential_regression
from scipy.signal import butter, filtfilt


def normal_sampling(mean, label_k, std):
    return math.exp(-(label_k - mean) ** 2 / (2 * std ** 2)) / (math.sqrt(2 * math.pi) * std)


def kl_loss(inputs, labels):
    criterion = nn.KLDivLoss(reduce=False)
    outputs = torch.log(inputs)
    loss = criterion(outputs, labels)
    # loss = loss.sum()/loss.shape[0]
    loss = loss.sum()
    return loss


def bandpass_filter(signal, fs=30, lowcut=0.75, highcut=2.5, order=1):
    b, a = butter(order, [lowcut / (fs / 2), highcut / (fs / 2)], btype='bandpass')
    return filtfilt(b, a, np.double(signal))


class Neg_Pearson(nn.Module):  # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(Neg_Pearson, self).__init__()

    def forward(self, preds, labels):  # all variable operation
        loss = 0
        for i in range(preds.shape[0]):
            sum_x = torch.sum(preds[i])  # x
            sum_y = torch.sum(labels[i])  # y
            sum_xy = torch.sum(preds[i] * labels[i])  # xy
            sum_x2 = torch.sum(torch.pow(preds[i], 2))  # x^2
            sum_y2 = torch.sum(torch.pow(labels[i], 2))  # y^2
            N = preds.shape[1]
            pearson = (N * sum_xy - sum_x * sum_y) / (
                torch.sqrt((N * sum_x2 - torch.pow(sum_x, 2)) * (N * sum_y2 - torch.pow(sum_y, 2))))
            loss += 1 - pearson

        loss = loss / preds.shape[0]
        return loss


class Hybrid_Loss(nn.Module):
    def __init__(self):
        super(Hybrid_Loss, self).__init__()
        self.criterion_Pearson = Neg_Pearson()

    def forward(self, pred_ppg, labels, epoch, FS, diff_flag):
        # 对labels做滤波
        # labels_np = labels.detach().cpu().numpy().flatten()
        # labels_filtered = bandpass_filter(labels_np, fs=FS).copy()
        # labels_filtered = torch.tensor(labels_filtered, dtype=labels.dtype, device=labels.device).view_as(labels)
        loss_time = self.criterion_Pearson(pred_ppg.view(1, -1), labels.view(1, -1))
        loss_Fre, _ = TorchLossComputer.Frequency_loss(pred_ppg.squeeze(-1), labels.squeeze(-1), diff_flag=diff_flag, Fs=FS, std=3.0)
        # if torch.isnan(loss_time):
        #     loss_time = 0
        loss = 0.2 * loss_time + 1.0 * loss_Fre
        # loss = 1 * loss_time + 0.2 * loss_Fre

        # loss = loss_time
        return loss




class TorchLossComputer(object):
    @staticmethod
    def compute_complex_absolute_given_k(output, k, N):
        two_pi_n_over_N = Variable(2 * math.pi * torch.arange(0, N, dtype=torch.float), requires_grad=True) / N
        hanning = Variable(torch.from_numpy(np.hanning(N)).type(torch.FloatTensor), requires_grad=True).view(1, -1)

        k = k.type(torch.FloatTensor).cuda()
        two_pi_n_over_N = two_pi_n_over_N.cuda()
        hanning = hanning.cuda()

        output = output.view(1, -1) * hanning
        output = output.view(1, 1, -1).type(torch.cuda.FloatTensor)
        k = k.view(1, -1, 1)
        two_pi_n_over_N = two_pi_n_over_N.view(1, 1, -1)
        complex_absolute = torch.sum(output * torch.sin(k * two_pi_n_over_N), dim=-1) ** 2 \
                           + torch.sum(output * torch.cos(k * two_pi_n_over_N), dim=-1) ** 2

        return complex_absolute

    @staticmethod
    def complex_absolute(output, Fs, bpm_range=None):
        output = output.view(1, -1)

        N = output.size()[1]

        unit_per_hz = Fs / N
        feasible_bpm = bpm_range / 60.0
        k = feasible_bpm / unit_per_hz

        # only calculate feasible PSD range [0.7,4]Hz
        complex_absolute = TorchLossComputer.compute_complex_absolute_given_k(output, k, N)

        return (1.0 / complex_absolute.sum()) * complex_absolute  # Analogous Softmax operator

    @staticmethod
    def cross_entropy_power_spectrum_loss(inputs, target, Fs):
        inputs = inputs.view(1, -1)
        target = target.view(1, -1)
        bpm_range = torch.arange(40, 180, dtype=torch.float).cuda()
        # bpm_range = torch.arange(40, 260, dtype=torch.float).cuda()

        complex_absolute = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)

        whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0)
        whole_max_idx = whole_max_idx.type(torch.float)

        # pdb.set_trace()

        # return F.cross_entropy(complex_absolute, target.view((1)).type(torch.long)).view(1),  (target.item() - whole_max_idx.item()) ** 2
        return F.cross_entropy(complex_absolute, target.view((1)).type(torch.long)), torch.abs(
            target[0] - whole_max_idx)

    @staticmethod
    def cross_entropy_power_spectrum_focal_loss(inputs, target, Fs, gamma):
        inputs = inputs.view(1, -1)
        target = target.view(1, -1)
        bpm_range = torch.arange(40, 180, dtype=torch.float).cuda()
        # bpm_range = torch.arange(40, 260, dtype=torch.float).cuda()

        complex_absolute = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)

        whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0)
        whole_max_idx = whole_max_idx.type(torch.float)

        # pdb.set_trace()
        criterion = FocalLoss(gamma=gamma)

        # return F.cross_entropy(complex_absolute, target.view((1)).type(torch.long)).view(1),  (target.item() - whole_max_idx.item()) ** 2
        return criterion(complex_absolute, target.view((1)).type(torch.long)), torch.abs(target[0] - whole_max_idx)

    @staticmethod
    def cross_entropy_power_spectrum_forward_pred(inputs, Fs):
        inputs = inputs.view(1, -1)
        bpm_range = torch.arange(40, 190, dtype=torch.float).cuda()
        # bpm_range = torch.arange(40, 180, dtype=torch.float).cuda()
        # bpm_range = torch.arange(40, 260, dtype=torch.float).cuda()

        complex_absolute = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)

        whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0)
        whole_max_idx = whole_max_idx.type(torch.float)

        return whole_max_idx

    @staticmethod
    def Frequency_loss(inputs, target, diff_flag, Fs, std):
        hr_pred, hr_gt = calculate_hr(inputs.detach().cpu(), target.detach().cpu(), diff_flag=diff_flag, fs=Fs)
        inputs = inputs.view(1, -1)
        target = target.view(1, -1)
        bpm_range = torch.arange(45, 150, dtype=torch.float).to(torch.device('cuda'))
        ca = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)
        sa = ca / torch.sum(ca)

        target_distribution = [normal_sampling(int(hr_gt), i, std) for i in range(45, 150)]
        target_distribution = [i if i > 1e-15 else 1e-15 for i in target_distribution]
        target_distribution = torch.Tensor(target_distribution).to(torch.device('cuda'))

        hr_gt = torch.tensor(hr_gt - 45).view(1).type(torch.long).to(torch.device('cuda'))
        return F.cross_entropy(ca, hr_gt), kl_loss(sa, target_distribution)

    @staticmethod
    def HR_loss(inputs, target, diff_flag, Fs, std):
        psd_pred, psd_gt = calculate_psd(inputs.detach().cpu(), target.detach().cpu(), diff_flag=diff_flag, fs=Fs)
        pred_distribution = [normal_sampling(np.argmax(psd_pred), i, std) for i in range(psd_pred.size)]
        pred_distribution = [i if i > 1e-15 else 1e-15 for i in pred_distribution]
        pred_distribution = torch.Tensor(pred_distribution).to(torch.device('cuda'))
        target_distribution = [normal_sampling(np.argmax(psd_gt), i, std) for i in range(psd_gt.size)]
        target_distribution = [i if i > 1e-15 else 1e-15 for i in target_distribution]
        target_distribution = torch.Tensor(target_distribution).to(torch.device('cuda'))
        return kl_loss(pred_distribution, target_distribution)



# class EvidentialRegression(nn.Module):
#     def __init__(self, coeff=1):
#         super(EvidentialRegression, self).__init__()
#         self.coeff = coeff
#
#     def forward(self, y_true, uncertainty_params):
#         """
#         y_true: [batch_size, output_units]
#         uncertainty_params: [batch_size, output_units, 4]
#         """
#         # 拆分不确定性参数
#         gamma, v, alpha, beta = torch.split(uncertainty_params, 1, dim=-1)
#         gamma = gamma.squeeze(-1)
#         v = v.squeeze(-1)
#         alpha = alpha.squeeze(-1)
#         beta = beta.squeeze(-1)
#         dist_params = (gamma, v, alpha, beta)
#         return evidential_regression(dist_params, y_true, lamb=self.coeff)

####################shuzhibaohu####################
    # def forward(self, y_true, uncertainty_params):
    #     gamma, v, alpha, beta = torch.split(uncertainty_params, 1, dim=-1)
    #     gamma = gamma.squeeze(-1)
    #     v = v.squeeze(-1).clamp(min=1e-2)
    #     alpha = alpha.squeeze(-1).clamp(min=1.1)
    #     beta = beta.squeeze(-1).clamp(min=1e-2)
    #     dist_params = (gamma, v, alpha, beta)
    #     return evidential_regression(dist_params, y_true, lamb=self.coeff)


#################coeff=aleatoric不确定性#################################
# class EvidentialRegression(nn.Module):
#     def __init__(self, coeff=1.0):
#         super(EvidentialRegression, self).__init__()
#         self.coeff = coeff
#
#     def forward(self, y_true, uncertainty_params):
#         """
#         y_true: [batch_size, output_units]
#         uncertainty_params: [batch_size, output_units, 4]
#         """
#         # 拆分不确定性参数
#         gamma, v, alpha, beta = torch.split(uncertainty_params, 1, dim=-1)
#         gamma = gamma.squeeze(-1)
#         gamma = (gamma - gamma.mean(dim=-1, keepdim=True)) / (gamma.std(dim=-1, keepdim=True) + 1e-6)
#         v = v.squeeze(-1)
#         alpha = alpha.squeeze(-1)
#         beta = beta.squeeze(-1)
#         # 计算aleatoric不确定性（数据不确定性）
#         aleatoric = torch.sqrt(beta / (alpha - 1))
#
#         # safe_alpha = torch.clamp(alpha - 1, min=1e-2)
#         # safe_beta = torch.clamp(beta, min=1e-2)
#         # aleatoric = torch.sqrt(safe_beta / safe_alpha)
#
#         batch_aleatoric_mean = aleatoric.mean().detach()  # 取均值，防止梯度回传
#         dist_params = (gamma, v, alpha, beta)
#         return evidential_regression(dist_params, y_true, lamb=batch_aleatoric_mean)


# class EvidentialRegression(nn.Module):
#     def __init__(self, coeff=1.0, mae_weight=1):
#         super(EvidentialRegression, self).__init__()
#         self.coeff = coeff
#         self.mae_weight = mae_weight
#
#     def forward(self, y_true, uncertainty_params, fs=30):
#         # 对y_true做滤波
#         # y_true_np = y_true.detach().cpu().numpy().flatten()
#         # y_true_filtered = bandpass_filter(y_true_np, fs=fs).copy()
#         # y_true_filtered = torch.tensor(y_true_filtered, dtype=y_true.dtype, device=y_true.device).view_as(y_true)
#         # 拆分不确定性参数
#         gamma, v, alpha, beta = torch.split(uncertainty_params, 1, dim=-1)
#         gamma = gamma.squeeze(-1)
#         # print("gamma:", gamma.shape)
#         # gamma = (gamma - gamma.mean(dim=-1, keepdim=True)) / (gamma.std(dim=-1, keepdim=True) + 1e-6)
#         v = v.squeeze(-1)
#         alpha = alpha.squeeze(-1)
#         beta = beta.squeeze(-1)
#         dist_params = (gamma, v, alpha, beta)
#         evidential_loss = evidential_regression(dist_params, y_true, lamb=0.0001)
#         # mae_loss = torch.mean(torch.abs(gamma - y_true))
#         # total_loss = evidential_loss + self.mae_weight * mae_loss
#         return evidential_loss


class EvidentialRegression(nn.Module):
    def __init__(self, coeff=1.0, mae_weight=1):
        super().__init__()
        self.coeff = coeff
        self.mae_weight = mae_weight

    def forward(self, y_true, uncertainty_params, fs=30):
        # 1. 拆分
        gamma, v, alpha, beta = torch.split(uncertainty_params, 1, dim=-1)
        gamma = gamma.squeeze(-1)
        v     = v.squeeze(-1)
        alpha = alpha.squeeze(-1)
        beta  = beta.squeeze(-1)

        # 2. 数值保护（只做一次）
        eps = 1e-6
        v     = torch.clamp(v,    min=eps)          # v > 0
        alpha = torch.clamp(alpha, min=1.0 + eps)   # α > 1
        beta  = torch.clamp(beta,  min=eps)         # β > 0

        # 3. 计算损失
        dist_params = (gamma, v, alpha, beta)
        loss = evidential_regression(dist_params, y_true, lamb=1)
        # loss = evidential_regression(dist_params, y_true, lambda_reg1= 0.001, lambda_reg2= 0.001)
        return loss






