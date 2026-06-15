import torch
import torch.nn.functional as F


# --------------------------------------------------
# yuanlai
# # --------------------------------------------------
def nig_nll(gamma, v, alpha, beta, y):
    two_beta_lambda = 2 * beta * (1 + v)
    t1 = 0.5 * (torch.pi / v).log()
    t2 = alpha * two_beta_lambda.log()
    t3 = (alpha + 0.5) * (v * (y - gamma) ** 2 + two_beta_lambda).log()
    t4 = alpha.lgamma()
    t5 = (alpha + 0.5).lgamma()
    nll = t1 - t2 + t3 + t4 - t5
    return nll.mean()

def nig_reg(gamma, v, alpha, _beta, y):
    reg = (y - gamma).abs() * (2 * v + alpha)
    return reg.mean()

def evidential_regression(dist_params, y, lamb=1):
    return nig_nll(*dist_params, y) + lamb * nig_reg(*dist_params, y)



# -----------------beiligong---------------------------------
# 1. 负对数似然 NLL（论文公式(10)）
# --------------------------------------------------
# def nig_nll(gamma, v, alpha, beta, y):
#     two_beta_lambda = 2 * beta * (1 + v)
#     t1 = 0.5 * torch.log(torch.pi / v)
#     t2 = alpha * torch.log(two_beta_lambda)
#     t3 = (alpha + 0.5) * torch.log(v * (y - gamma).square() + two_beta_lambda)
#     t4 = torch.lgamma(alpha)
#     t5 = torch.lgamma(alpha + 0.5)
#     nll = t1 - t2 + t3 + t4 - t5
#     return nll.mean()

# --------------------------------------------------
# 2. 第一类正则 REG1（论文公式(11)）
# # --------------------------------------------------
# def nig_reg1(gamma, v, alpha, beta, y):
#     return (y - gamma).abs() * (2 * v + alpha)
#
# # --------------------------------------------------
# # 3. 第二类正则 REG2（论文公式(13)）
# # --------------------------------------------------
# def nig_reg2(gamma, v, alpha, beta, y):
#     return (y - gamma).square() * (alpha * v / (beta + 1e-8))
#
# # --------------------------------------------------
# # 4. 总 Evidential Loss（论文式(9)）
# # --------------------------------------------------
# def evidential_regression(dist_params, y, lambda_reg1: float = 0.001, lambda_reg2: float = 0.001):
#     """
#     dist_params: [gamma, v, alpha, beta] 四个张量
#     y: 真值
#     """
#     gamma, v, alpha, beta = dist_params
#     loss = nig_nll(gamma, v, alpha, beta, y)            # L_NLL
#     loss += lambda_reg1 * nig_reg1(gamma, v, alpha, beta, y).mean()  # L_REG1
#     loss += lambda_reg2 * nig_reg2(gamma, v, alpha, beta, y).mean()  # L_REG2
#     return loss



# Normal Inverse Gamma Negative Log-Likelihood
# from https://arxiv.org/abs/1910.02600:
# > we denote the loss, L^NLL_i as the negative logarithm of model
# > evidence ...
# def nig_nll(gamma, v, alpha, beta, y):
#     two_beta_lambda = 2 * beta * (1 + v)
#     t1 = 0.5 * (torch.pi / v).log()
#     t2 = alpha * two_beta_lambda.log()
#     t3 = (alpha + 0.5) * (v * (y - gamma) ** 2 + two_beta_lambda).log()
#     t4 = alpha.lgamma()
#     t5 = (alpha + 0.5).lgamma()
#     nll = t1 - t2 + t3 + t4 - t5
#     return nll.mean()


#
# # Normal Inverse Gamma regularization
# # from https://arxiv.org/abs/1910.02600:
# # > we formulate a novel evidence regularizer, L^R_i
# # > scaled on the error of the i-th prediction
# def nig_reg(gamma, v, alpha, _beta, y):
#     reg = (y - gamma).abs() * (2 * v + alpha)
#     return reg.mean()



# def evidential_regression(dist_params, y, lamb=1.0):
#     return nig_nll(*dist_params, y) + lamb * nig_reg(*dist_params, y)


#  Uncertainty Regularized Evidential Regression##################HUA
# def nig_nll(gamma, v, alpha, beta, y):
#     two_beta_lambda = 2 * beta * (1 + v)
#     t1 = 0.5 * (torch.pi / v).log()
#     t2 = alpha * two_beta_lambda.log()
#     t3 = (alpha + 0.5) * (v * (y - gamma) ** 2 + two_beta_lambda).log()
#     t4 = alpha.lgamma()
#     t5 = (alpha + 0.5).lgamma()
#     nll = t1 - t2 + t3 + t4 - t5
#     return nll.mean()
#
# def nig_reg(gamma, v, alpha, _beta, y):
#     reg = (y - gamma).abs() * (2 * v + alpha)
#     return reg.mean()
#
# def nig_uncertainty_reg(gamma, v, alpha, _beta, y):
#     uncertainty_reg = - (y - gamma).abs() * torch.log(torch.exp(alpha - 1) - 1)
#     return uncertainty_reg.mean()
#
# def evidential_regression(dist_params, y, lamb=1.0, lamb_1=1.0):
#     gamma, v, alpha, beta = dist_params
#     nll_loss = nig_nll(gamma, v, alpha, beta, y)
#     reg_loss = nig_reg(gamma, v, alpha, beta, y)
#     uncertainty_reg_loss = nig_uncertainty_reg(gamma, v, alpha, beta, y)
#     total_loss = nll_loss + lamb * reg_loss + lamb_1 * uncertainty_reg_loss
#     return total_loss



# def nig_nll(gamma, v, alpha, beta, y):
#     two_beta_lambda = 2 * beta * (1 + v)
#     t1 = 0.5 * (torch.pi / v).log()
#     t2 = alpha * two_beta_lambda.log()
#     t3 = (alpha + 0.5) * (v * (y - gamma) ** 2 + two_beta_lambda).log()
#     t4 = alpha.lgamma()
#     t5 = (alpha + 0.5).lgamma()
#     nll = t1 - t2 + t3 + t4 - t5
#     return nll.mean()
#
# def nig_reg(gamma, v, alpha, _beta, y):
#     reg = (y - gamma).abs() * (2 * v + alpha)
#     return reg.mean()
#
# def evidential_regression(dist_params, y, lamb=0.01):
#     return nig_nll(*dist_params, y) + lamb * nig_reg(*dist_params, y)



###########################NUR###################
# def nig_nur(gamma, v, alpha, beta, y):
#     nur = (y - gamma) ** 2 * (v * (alpha - 1)) / (beta * (v + 1))
#     return nur.mean()

# def evidential_regression(dist_params, y, lamb=1.0, eta=1.0):
#     gamma, v, alpha, beta = dist_params
#     nll_loss = nig_nll(gamma, v, alpha, beta, y)
#     emr_reg = nig_reg(gamma, v, alpha, beta, y)
#     nur_reg = nig_nur(gamma, v, alpha, beta, y)
#     total_loss = nll_loss + lamb * emr_reg + eta * nur_reg
#     return total_loss



# # Normal Inverse Gamma Negative Log-Likelihood
# def nig_nll(gamma, v, alpha, beta, y):
#     two_beta_lambda = 2 * beta * (1 + v)
#     t1 = 0.5 * (torch.pi / v).log()
#     t2 = alpha * two_beta_lambda.log()
#     t3 = (alpha + 0.5) * (v * (y - gamma) ** 2 + two_beta_lambda).log()
#     t4 = alpha.lgamma()
#     t5 = (alpha + 0.5).lgamma()
#     nll = t1 - t2 + t3 + t4 - t5
#     return nll.mean()
#
# # Normal Inverse Gamma regularization with normalized residual
# def nig_reg(gamma, v, alpha, beta, y, p=2.0):
#     w_st = torch.sqrt(beta * (1 + v) / (alpha * v))
#     normalized_residual = torch.abs((y - gamma) / w_st)
#     reg = normalized_residual ** p * (2 * v + alpha)
#     return reg.mean()
#
# # Evidential regression loss function with normalized residual
# def evidential_regression(dist_params, y, lamb, p=2.0):
#     gamma, v, alpha, beta = dist_params
#     nll_loss = nig_nll(gamma, v, alpha, beta, y)
#     reg_loss = nig_reg(gamma, v, alpha, beta, y, p=p)
#     return nll_loss + lamb * reg_loss