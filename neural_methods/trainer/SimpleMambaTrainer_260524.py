"""Trainer for BIMamba."""
import os
import numpy as np
import torch
import torch.optim as optim
import random
from tqdm import tqdm
from evaluation.post_process import calculate_hr
from evaluation.metrics import calculate_metrics
from neural_methods.model.SimpleMamba import SimpleMamba
from neural_methods.trainer.BaseTrainer import BaseTrainer
from neural_methods.loss.TorchLossComputer import Hybrid_Loss, EvidentialRegression
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import pearsonr, spearmanr


_EXPERIMENT_TAGS = {
    "PURE_PURE_PURE_SimpleMamba": "P_P",
    "UBFC_UBFC_UBFC_SimpleMamba": "U_U",
    "PURE_UBFC_SimpleMamba": "P_U",
    "UBFC_PURE_SimpleMamba": "U_P",
    "PURE_MMPD_SimpleMamba": "P_M",
    "UBFC_MMPD_SimpleMamba": "U_M",
}


def _experiment_tag(model_file_name):
    try:
        return _EXPERIMENT_TAGS[model_file_name]
    except KeyError:
        raise ValueError(
            f"Unknown MODEL_FILE_NAME '{model_file_name}'. "
            "Cannot resolve uncertainty output tag."
        ) from None


def evaluate_uncertainty_quality(
        uncertainty_params,
        labels,
        save_dir,
        confidence_level=0.95,
        num_bins=20):
    """
    评估不确定性质量：

    1. ECE
    2. MCE
    3. NLL
    4. Sharpness
    5. Calibration Curve

    Args:
        uncertainty_params:
            {subj: {seg: tensor(T,4)}}

        labels:
            {subj: {seg: tensor(T,1)}}

        save_dir:
            保存路径

        confidence_level:
            Sharpness使用的置信区间

        num_bins:
            calibration bins
    """


    all_gamma = []
    all_v = []
    all_alpha = []
    all_beta = []
    all_y = []

    # =====================================================
    # 收集所有预测
    # =====================================================
    for subj in uncertainty_params.keys():
        for seg in uncertainty_params[subj].keys():

            params = uncertainty_params[subj][seg].cpu()
            y_true = labels[subj][seg].cpu()

            gamma, v, alpha, beta = torch.split(params, 1, dim=-1)

            all_gamma.append(gamma)
            all_v.append(v)
            all_alpha.append(alpha)
            all_beta.append(beta)
            all_y.append(y_true)

    gamma = torch.cat(all_gamma).numpy().flatten()
    v = torch.cat(all_v).numpy().flatten()
    alpha = torch.cat(all_alpha).numpy().flatten()
    beta = torch.cat(all_beta).numpy().flatten()
    y_true = torch.cat(all_y).numpy().flatten()

    # =====================================================
    # Student-t predictive distribution
    # =====================================================

    dof = 2 * alpha

    predictive_var = beta * (1 + v) / (v * alpha)

    predictive_std = np.sqrt(predictive_var)

    # =====================================================
    # NLL
    # =====================================================

    nll = -np.mean(
        st.t.logpdf(
            y_true,
            df=dof,
            loc=gamma,
            scale=predictive_std
        )
    )

    # =====================================================
    # Calibration Curve
    # =====================================================

    confidences = np.linspace(0.01, 0.99, num_bins)

    observed_coverages = []

    for conf in confidences:

        t_quantile = st.t.ppf(
            (1 + conf) / 2,
            df=dof
        )

        lower = gamma - t_quantile * predictive_std
        upper = gamma + t_quantile * predictive_std

        covered = (
                (y_true >= lower)
                & (y_true <= upper)
        )

        observed_coverages.append(np.mean(covered))

    observed_coverages = np.array(observed_coverages)

    # =====================================================
    # ECE
    # =====================================================

    ece = np.mean(
        np.abs(confidences - observed_coverages)
    )

    # =====================================================
    # MCE
    # =====================================================

    mce = np.max(
        np.abs(confidences - observed_coverages)
    )

    # =====================================================
    # Sharpness
    # =====================================================

    t_quantile = st.t.ppf(
        (1 + confidence_level) / 2,
        df=dof
    )

    interval_width = (
            2 * t_quantile * predictive_std
    )

    sharpness = np.mean(interval_width)

    # =====================================================
    # 保存结果
    # =====================================================

    os.makedirs(save_dir, exist_ok=True)

    np.savez(
        os.path.join(
            save_dir,
            "uncertainty_quality_metrics.npz"
        ),
        confidences=confidences,
        observed_coverages=observed_coverages,
        ece=ece,
        mce=mce,
        nll=nll,
        sharpness=sharpness
    )

    # =====================================================
    # 绘图
    # =====================================================

    plt.figure(figsize=(6, 6))

    plt.plot(
        confidences,
        observed_coverages,
        linewidth=2,
        label='Observed Coverage'
    )

    plt.plot(
        [0, 1],
        [0, 1],
        '--',
        label='Ideal Calibration'
    )

    plt.xlabel('Expected Confidence')
    plt.ylabel('Observed Coverage')

    plt.title(
        f'Calibration Curve\n'
        f'ECE={ece:.4f} | '
        f'MCE={mce:.4f}'
    )

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            save_dir,
            'calibration_curve.png'
        ),
        dpi=300
    )

    plt.close()

    # =====================================================
    # 打印
    # =====================================================

    print("\n========== Uncertainty Quality ==========")

    print(f"ECE        : {ece:.6f}")
    print(f"MCE        : {mce:.6f}")
    print(f"NLL        : {nll:.6f}")
    print(f"Sharpness  : {sharpness:.6f}")

    return {
        'ECE': ece,
        'MCE': mce,
        'NLL': nll,
        'Sharpness': sharpness
    }

def calculate_uncertainty_metrics(uncertainty_params):

    all_aleatoric = []
    all_epistemic = []
    all_predictive = []

    for subj in uncertainty_params.keys():
        for seg in uncertainty_params[subj].keys():

            params = uncertainty_params[subj][seg]

            gamma, v, alpha, beta = torch.split(params, 1, dim=-1)

            alpha = torch.clamp(alpha, min=1.0 + 1e-6)
            # v = torch.clamp(v, min=1e-6)
            v = torch.clamp(v, min=1e-2)
            beta = torch.clamp(beta, min=1e-6)

            # Aleatoric uncertainty
            aleatoric = torch.sqrt(
                beta / (alpha - 1)
            )

            # Epistemic uncertainty
            epistemic = torch.sqrt(
                beta / (v * (alpha - 1))
            )

            # Predictive uncertainty
            predictive = torch.sqrt(
                beta * (1 + v) / (v * (alpha - 1))
            )

            all_aleatoric.append(aleatoric)
            all_epistemic.append(epistemic)
            all_predictive.append(predictive)

    all_aleatoric = torch.cat(all_aleatoric, dim=0)
    all_epistemic = torch.cat(all_epistemic, dim=0)
    all_predictive = torch.cat(all_predictive, dim=0)

    metrics = {

        'aleatoric_mean':
            all_aleatoric.mean().item(),

        'aleatoric_std':
            all_aleatoric.std().item(),

        'aleatoric_max':
            all_aleatoric.max().item(),

        'epistemic_mean':
            all_epistemic.mean().item(),

        'epistemic_std':
            all_epistemic.std().item(),

        'epistemic_max':
            all_epistemic.max().item(),

        'predictive_mean':
            all_predictive.mean().item(),

        'predictive_std':
            all_predictive.std().item(),

        'predictive_max':
            all_predictive.max().item(),
    }

    return metrics

def calculate_error_uncertainty_correlation(
        predictions,
        labels,
        uncertainty_params,
        save_dir=None):

    all_errors = []

    all_aleatoric = []
    all_epistemic = []
    all_predictive = []

    for subj in predictions.keys():
        for seg in predictions[subj].keys():

            pred = predictions[subj][seg].cpu().numpy().flatten()
            gt = labels[subj][seg].cpu().numpy().flatten()

            params = uncertainty_params[subj][seg]

            gamma, v, alpha, beta = torch.split(
                params, 1, dim=-1)

            alpha = torch.clamp(alpha, min=1.0 + 1e-6)
            # v = torch.clamp(v, min=1e-6)
            v = torch.clamp(v, min=1e-2)
            beta = torch.clamp(beta, min=1e-6)

            aleatoric = torch.sqrt(
                beta / (alpha - 1)
            ).cpu().numpy().flatten()

            epistemic = torch.sqrt(
                beta / (v * (alpha - 1))
            ).cpu().numpy().flatten()

            predictive = torch.sqrt(
                beta * (1 + v) / (v * (alpha - 1))
            ).cpu().numpy().flatten()

            error = np.abs(pred - gt)

            all_errors.extend(error)

            all_aleatoric.extend(aleatoric)
            all_epistemic.extend(epistemic)
            all_predictive.extend(predictive)

    all_errors = np.array(all_errors)

    all_aleatoric = np.array(all_aleatoric)
    all_epistemic = np.array(all_epistemic)
    all_predictive = np.array(all_predictive)

    # Pearson correlation
    alea_pearson, _ = pearsonr(
        all_errors,
        all_aleatoric
    )

    epis_pearson, _ = pearsonr(
        all_errors,
        all_epistemic
    )

    pred_pearson, _ = pearsonr(
        all_errors,
        all_predictive
    )

    # Spearman correlation
    alea_spearman, _ = spearmanr(
        all_errors,
        all_aleatoric
    )

    epis_spearman, _ = spearmanr(
        all_errors,
        all_epistemic
    )

    pred_spearman, _ = spearmanr(
        all_errors,
        all_predictive
    )

    print("\n=== Error-Uncertainty Correlation ===")

    print(f"Aleatoric Pearson : {alea_pearson:.4f}")
    print(f"Epistemic Pearson : {epis_pearson:.4f}")
    print(f"Predictive Pearson: {pred_pearson:.4f}")

    print(f"Aleatoric Spearman : {alea_spearman:.4f}")
    print(f"Epistemic Spearman : {epis_spearman:.4f}")
    print(f"Predictive Spearman: {pred_spearman:.4f}")

    # scatter plot
    if save_dir is not None:

        os.makedirs(save_dir, exist_ok=True)

        plt.figure(figsize=(6, 5))

        plt.scatter(
            all_errors,
            all_predictive,
            s=5,
            alpha=0.3
        )

        plt.xlabel('Prediction Error')
        plt.ylabel('Predictive Uncertainty')

        plt.title('Error-Uncertainty Correlation')

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                save_dir,
                'error_uncertainty_scatter.png'
            ),
            dpi=300
        )

        plt.close()

    return {
        'alea_pearson': alea_pearson,
        'epis_pearson': epis_pearson,
        'pred_pearson': pred_pearson,

        'alea_spearman': alea_spearman,
        'epis_spearman': epis_spearman,
        'pred_spearman': pred_spearman,
    }

def plot_uncertainty_histogram(
        uncertainty_params,
        save_dir,
        dataset_name='OOD'):

    all_predictive = []

    for subj in uncertainty_params.keys():
        for seg in uncertainty_params[subj].keys():

            params = uncertainty_params[subj][seg]

            gamma, v, alpha, beta = torch.split(
                params,
                1,
                dim=-1
            )

            alpha = torch.clamp(alpha, min=1.0 + 1e-6)
            # v = torch.clamp(v, min=1e-6)
            v = torch.clamp(v, min=1e-2)
            beta = torch.clamp(beta, min=1e-6)

            predictive = torch.sqrt(
                beta * (1 + v) /
                (v * (alpha - 1))
            )

            all_predictive.extend(
                predictive.cpu().numpy().flatten()
            )

    plt.figure(figsize=(6, 5))

    plt.hist(
        all_predictive,
        bins=100,
        density=True,
        alpha=0.7
    )

    plt.xlabel('Predictive Uncertainty')
    plt.ylabel('Density')

    plt.title(
        f'Predictive Uncertainty Histogram ({dataset_name})'
    )

    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)

    plt.savefig(
        os.path.join(
            save_dir,
            f'{dataset_name}_predictive_histogram.png'
        ),
        dpi=300
    )

    plt.close()

def save_ood_statistics(
        dataset_name,
        mae,
        uncertainty_metrics,
        save_dir):

    os.makedirs(save_dir, exist_ok=True)

    txt_path = os.path.join(
        save_dir,
        f'{dataset_name}_ood_statistics.txt'
    )

    with open(txt_path, 'w') as f:

        f.write(f'Dataset: {dataset_name}\n')
        f.write(f'MAE: {mae:.6f}\n\n')

        for k, v in uncertainty_metrics.items():
            f.write(f'{k}: {v:.6f}\n')

    print(f'OOD statistics saved to: {txt_path}')


def plot_calibration_curve(uncertainty_params, labels, save_dir):
    """
    绘制 Expected Confidence vs Observed Coverage 曲线（平滑版本）
    并保存曲线数据以便复现。
    """
    import scipy.stats as st

    all_gamma, all_v, all_alpha, all_beta, all_y = [], [], [], [], []

    # === 汇总所有预测参数与真实值 ===
    for subj in uncertainty_params.keys():
        for seg in uncertainty_params[subj].keys():
            params = uncertainty_params[subj][seg].cpu()
            y_true = labels[subj][seg].cpu()
            gamma, v, alpha, beta = torch.split(params, 1, dim=-1)
            all_gamma.append(gamma)
            all_v.append(v)
            all_alpha.append(alpha)
            all_beta.append(beta)
            all_y.append(y_true)

    gamma = torch.cat(all_gamma).numpy().flatten()
    v = torch.cat(all_v).numpy().flatten()
    alpha = torch.cat(all_alpha).numpy().flatten()
    beta = torch.cat(all_beta).numpy().flatten()
    y_true = torch.cat(all_y).numpy().flatten()

    # === 计算预测区间与覆盖率 ===
    scale = np.sqrt(beta * (1 + v) / (v * alpha))
    # confidences = np.linspace(0.05, 0.99, 50)  # 增加采样点，曲线更平滑
    confidences = np.linspace(0, 1, 300)  # 增加采样点，曲线更平滑
    observed_coverages = []

    for conf in confidences:
        dof = 2 * alpha
        t_quantile = st.t.ppf((1 + conf) / 2, df=dof)
        lower = gamma - t_quantile * scale
        upper = gamma + t_quantile * scale
        covered = (y_true >= lower) & (y_true <= upper)
        observed_coverages.append(np.mean(covered))

    observed_coverages = np.array(observed_coverages)
    ece = np.mean(np.abs(confidences - observed_coverages))

    # === 创建输出目录 ===
    os.makedirs(save_dir, exist_ok=True)

    # === 保存数据以便复现 ===
    data_path = os.path.join(save_dir, "uncertainty_calibration_data.npz")
    np.savez(
        data_path,
        expected_confidence=confidences,
        observed_coverage=observed_coverages,
        ece=np.array([ece])
    )
    print(f"\n💾 Calibration data saved to: {data_path}")

    # === 绘制图像 ===
    plt.figure(figsize=(6, 6))
    plt.plot(confidences, observed_coverages, '-', linewidth=2.2, color='tab:blue', label='Observed Coverage')
    plt.plot([0, 1], [0, 1], '--', color='gray', linewidth=1.5, label='Ideal Calibration')
    plt.xlabel('Expected Confidence', fontsize=12)
    plt.ylabel('Observed Coverage', fontsize=12)
    plt.title('Uncertainty Calibration Curve', fontsize=13)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'uncertainty_calibration_curve.png')
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"✅ Saved calibration curve to: {save_path}")
    print(f"Expected Calibration Error (ECE): {ece:.4f}")



class SimpleMambaTrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.chunk_len = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0
        self.diff_flag = 0

        # 新增：基于损失大小的动态权重方法参数（直接写在代码中）
        self.loss_balance_method = 'exponential_decay'  # 可选值: 'normalization', 'exponential_decay', 'uncertainty_only'
        self.exp_decay_scale = 1.0  # 指数衰减强度（仅用于exponential_decay）
        self.min_weight = 0.001  # 最小权重
        self.max_weight = 0.99  # 最大权重

        if config.TRAIN.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized":
            self.diff_flag = 1
        if config.TOOLBOX_MODE == "train_and_test":
            self.model = SimpleMamba().to(self.device)
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
            self.num_train_batches = len(data_loader["train"])
            self.criterion1 = Hybrid_Loss()
            self.criterion2 = EvidentialRegression()
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=config.TRAIN.LR, weight_decay=0)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS,
                steps_per_epoch=self.num_train_batches)
        elif config.TOOLBOX_MODE == "only_test":
            self.model = SimpleMamba().to(self.device)
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
        else:
            raise ValueError("EfficientPhys trainer initialized in incorrect toolbox mode!")

    def _compute_uncertainty_weight(self, uncertainty_params):
        """
        原有的不确定性权重计算方法（Kendall & Gal方法）
        """
        gamma, v, alpha, beta = torch.chunk(uncertainty_params, 4, dim=-1)
        alpha = torch.clamp(alpha, min=1.0 + 1e-4)
        # v = torch.clamp(v, min=1e-4)
        v = torch.clamp(v, min=1e-2)
        beta = torch.clamp(beta, min=1e-4)

        # total uncertainty: aleatoric + epistemic
        sigma2 = beta / (alpha - 1) + beta / ((alpha - 1) * v)

        # Kendall & Gal 权重: w = 1 / mean(sigma^2)
        w = 1.0 / (sigma2.mean().item() + 1e-6)
        return w

    def _compute_normalization_weight(self, loss1, loss2):
        """
        归一化方案：α = L_uncertainty / (L_uncertainty + L_regression)
        """
        total_loss = loss1.item() + loss2.item()
        if total_loss > 1e-8:
            alpha = loss2.item() / total_loss  # loss2是uncertainty loss
        else:
            alpha = 0.5  # 默认平衡权重

        # 限制权重范围
        alpha = np.clip(alpha, self.min_weight, self.max_weight)

        return alpha

    def _compute_exponential_decay_weight(self, loss1, loss2):
        """
        指数衰减方案：α = exp(-scale * L_uncertainty)
        """
        uncertainty_loss = loss2.item()

        # 指数衰减权重，scale控制衰减速度
        alpha = np.exp(-self.exp_decay_scale * uncertainty_loss)

        # 限制权重范围
        alpha = np.clip(alpha, self.min_weight, self.max_weight)

        return alpha

    def _compute_loss_based_weight(self, loss1, loss2, epoch, uncertainty_params=None):
        """
        统一的基于损失大小的权重计算接口
        """
        if self.loss_balance_method == 'normalization':
            alpha = self._compute_normalization_weight(loss1, loss2)

        elif self.loss_balance_method == 'exponential_decay':
            alpha = self._compute_exponential_decay_weight(loss1, loss2)

        elif self.loss_balance_method == 'uncertainty_only':
            # 保持原有的不确定性权重方法
            if uncertainty_params is not None:
                alpha = self._compute_uncertainty_weight(uncertainty_params)
            else:
                alpha = 0.5  # 默认权重
        else:
            alpha = 0.5  # 默认平衡权重

        # 应用warm-up（可选）
        warm_epoch = 5
        if epoch < warm_epoch:
            alpha = alpha * (epoch / warm_epoch)

        return alpha

    def train(self, data_loader):
        """Training routine for model"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        val_mae_list = []
        val_pearson_list = []

        # 训练与验证损失历史
        train_loss1_hist = []
        train_loss2_hist = []
        train_total_hist = []
        train_weight_hist = []  # 权重历史

        val_loss1_hist = []
        val_loss2_hist = []
        val_total_hist = []

        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            self.model.train()

            tbar = tqdm(data_loader["train"], ncols=80)
            epoch_loss1 = 0.0
            epoch_loss2 = 0.0
            epoch_total = 0.0
            epoch_weight = 0.0
            batch_count = 0

            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                data, labels = batch[0].float(), batch[1].float()
                N, D, C, H, W = data.shape

                if self.config.TRAIN.AUG:
                    data, labels = self.data_augmentation(data, labels)

                data = data.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                uncertainty_params = self.model(data)
                mu = uncertainty_params[..., 0]

                # 计算两个损失
                loss1 = 0.0
                for ib in range(N):
                    loss1 += self.criterion1(mu[ib], labels[ib], epoch, self.config.TRAIN.DATA.FS, self.diff_flag)
                loss1 = loss1 / N

                loss2 = self.criterion2(labels, uncertainty_params)

                # 基于损失大小计算动态权重
                alpha = self._compute_loss_based_weight(loss1, loss2, epoch, uncertainty_params)

                # 计算总损失：L = α * L_uncertainty + (1-α) * L_regression
                # loss = (1 - alpha) * loss1 + alpha * 0.01 * loss2 #UBFC
                loss = (1 - alpha) * loss1 + alpha * loss2  # PURE

                # 打印信息
                if idx % 50 == 0:
                    print(f"[Train] batch={idx:04d} | α={alpha:.4f} | loss1={loss1.item():.4f} | loss2={loss2.item():.4f} | total={loss.item():.4f}")

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                tbar.set_postfix(loss=loss.item(), α=alpha)

                epoch_loss1 += loss1.item()
                epoch_loss2 += loss2.item()
                epoch_total += loss.item()
                epoch_weight += alpha
                batch_count += 1

                if idx % 100 == 0:
                    v_vals = uncertainty_params[..., 1].detach().cpu().numpy()
                    alpha_vals = uncertainty_params[..., 2].detach().cpu().numpy()
                    beta_vals = uncertainty_params[..., 3].detach().cpu().numpy()
                    print(f"v mean: {v_vals.mean():.4f}, min: {v_vals.min():.4f}, max: {v_vals.max():.4f}; "
                          f"alpha mean: {alpha_vals.mean():.4f}, min: {alpha_vals.min():.4f}, max: {alpha_vals.max():.4f}; "
                          f"beta mean: {beta_vals.mean():.4f}, min: {beta_vals.min():.4f}, max: {beta_vals.max():.4f}")
                    print("mu min/max/mean:", mu.min().item(), mu.max().item(), mu.mean().item())
                    print("labels min/max/mean:", labels.min().item(), labels.max().item(), labels.mean().item())
                    print("loss1 (regression):", loss1.item())
                    print("loss2 (uncertainty):", loss2.item())
                    print("dynamic weight α:", alpha)

            # 记录每个epoch的统计信息
            train_loss1_hist.append(epoch_loss1 / batch_count)
            train_loss2_hist.append(epoch_loss2 / batch_count)
            train_total_hist.append(epoch_total / batch_count)
            train_weight_hist.append(epoch_weight / batch_count)

            self.save_model(epoch)

            # 验证阶段
            if not self.config.TEST.USE_LAST_EPOCH:
                valid_total, valid_mae, valid_pearson, valid_l1, valid_l2 = self.valid(data_loader, epoch)
                print(f'validation total-loss: {valid_total:.4f}, '
                      f'mae: {valid_mae:.4f}, pearson: {valid_pearson:.4f}')
                val_mae_list.append(valid_mae)
                val_pearson_list.append(valid_pearson)

                val_total_hist.append(valid_total)
                val_loss1_hist.append(valid_l1)
                val_loss2_hist.append(valid_l2)

        # 训练结束：画对比图（包含权重曲线）
        custom_fig_dir = './loss'
        os.makedirs(custom_fig_dir, exist_ok=True)

        def _plot(train, val, ylabel, title, fname):
            plt.figure()
            epochs = range(1, len(train) + 1)
            plt.plot(epochs, train, label='Train', color='tab:blue')
            if val:
                plt.plot(epochs, val, label='Val', color='tab:orange')
            plt.xlabel('Epoch')
            plt.ylabel(ylabel)
            plt.title(title)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(custom_fig_dir, fname))
            plt.close()

        _plot(train_loss1_hist, val_loss1_hist,
              'Loss1 Value', 'Regression Loss (Train vs Val)', 'loss1_curve.png')

        _plot(train_loss2_hist, val_loss2_hist,
              'Loss2 Value', 'Uncertainty Loss (Train vs Val)', 'loss2_curve.png')

        _plot(train_total_hist, val_total_hist,
              'Total Loss Value', 'Total Loss (Train vs Val)', 'total_loss_curve.png')

        # 新增：动态权重变化曲线
        _plot(train_weight_hist, None,
              'Weight α Value', 'Dynamic Weight Evolution', 'dynamic_weight_curve.png')

        # 选择最佳epoch
        if not self.config.TEST.USE_LAST_EPOCH:
            best_mae_epoch = int(np.argmin(val_mae_list))
            best_pearson_epoch = int(np.argmax(val_pearson_list))
            self.best_mae_epoch = best_mae_epoch
            self.best_pearson_epoch = best_pearson_epoch
            print(f"Best epoch by MAE: {best_mae_epoch}, "
                  f"min MAE: {val_mae_list[best_mae_epoch]}")
            print(f"Best epoch by Pearson: {best_pearson_epoch}, "
                  f"max Pearson: {val_pearson_list[best_pearson_epoch]}")

    def valid(self, data_loader, current_epoch):
        """Model evaluation on the validation dataset."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")
        print('')
        print("===Validating===")

        self.model.eval()
        all_preds, all_labels = [], []
        total_loss, total_l1, total_l2 = 0.0, 0.0, 0.0
        num_batches = 0

        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")
                data_valid, labels_valid = valid_batch[0].to(self.device), \
                                           valid_batch[1].to(self.device)
                N, D, C, H, W = data_valid.shape
                uncertainty_params = self.model(data_valid)
                mu_valid = uncertainty_params[..., 0]

                loss1 = 0.0
                for ib in range(N):
                    loss1 += self.criterion1(mu_valid[ib], labels_valid[ib],
                                             current_epoch,  # 修正：使用当前epoch而不是总epoch数
                                             self.config.VALID.DATA.FS, self.diff_flag)
                loss1 = loss1 / N
                loss2 = self.criterion2(labels_valid, uncertainty_params)

                # 验证阶段也使用相同的动态权重计算
                alpha = self._compute_loss_based_weight(loss1, loss2, current_epoch, uncertainty_params)

                # 计算总损失
                # loss = (1 - alpha) * loss1 + alpha * 0.01 * loss2
                loss = (1 - alpha) * loss1 + alpha * loss2  # PURE

                total_l1 += loss1.item()
                total_l2 += loss2.item()
                total_loss += loss.item()
                num_batches += 1

                vbar.set_postfix(loss=loss.item(), α=alpha)

                all_preds.append(mu_valid.cpu().numpy().flatten())
                all_labels.append(labels_valid.cpu().numpy().flatten())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        mae = np.mean(np.abs(all_preds - all_labels))
        if np.std(all_preds) > 1e-6 and np.std(all_labels) > 1e-6:
            pearson = np.corrcoef(all_preds, all_labels)[0, 1]
        else:
            pearson = 0.0

        return total_loss / num_batches, mae, pearson, \
               total_l1 / num_batches, total_l2 / num_batches


    def test(self, data_loader):
        """ Model evaluation on the testing dataset."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")

        print('')
        print("===Testing===")
        # 支持分别加载best_mae_epoch和best_pearson_epoch的模型权重
        test_epochs = []
        if hasattr(self, 'best_mae_epoch') and hasattr(self, 'best_pearson_epoch'):
            test_epochs = [self.best_mae_epoch, self.best_pearson_epoch]
        else:
            # 兼容老用法
            if self.config.TOOLBOX_MODE == "only_test":
                test_epochs = [None]  # 只用INFERENCE.MODEL_PATH
            elif self.config.TEST.USE_LAST_EPOCH:
                test_epochs = [self.max_epoch_num - 1]
            else:
                test_epochs = [self.best_epoch]

        for epoch in test_epochs:
            if epoch is not None:
                model_path = os.path.join(
                    self.model_dir, self.model_file_name + f'_Epoch{epoch}.pth')
                print(f"Testing with model from epoch {epoch} ...")
                self.model.load_state_dict(torch.load(model_path))
            else:
                if self.config.TOOLBOX_MODE == "only_test":
                    if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                        raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
                    self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
                    print("Testing uses pretrained model!")
                elif self.config.TEST.USE_LAST_EPOCH:
                    last_epoch_model_path = os.path.join(
                        self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                    print("Testing uses last epoch as non-pretrained model!")
                    print(last_epoch_model_path)
                    self.model.load_state_dict(torch.load(last_epoch_model_path))
                else:
                    best_model_path = os.path.join(
                        self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                    print("Testing uses best epoch selected using model selection as non-pretrained model!")
                    print(best_model_path)
                    self.model.load_state_dict(torch.load(best_model_path))

            self.model = self.model.to(self.config.DEVICE)
            self.model.eval()
            with torch.no_grad():
                predictions = dict()
                labels = dict()
                uncertainty_params = dict()  # 存储不确定性参数

                for _, test_batch in enumerate(data_loader['test']):
                    batch_size = test_batch[0].shape[0]
                    chunk_len = self.chunk_len
                    data_test, labels_test = test_batch[0].to(self.config.DEVICE), test_batch[1].to(self.config.DEVICE)

                    uncertainty_params_test = self.model(data_test)
                    # pred_ppg_test = (pred_ppg_test-torch.mean(pred_ppg_test, axis=-1).view(-1, 1))/torch.std(pred_ppg_test, axis=-1).view(-1, 1)    # normalize
                    mu_test = uncertainty_params_test [..., 0]  # [N, T]
                    # mu_norm_test = (mu_test - torch.mean(mu_test, axis=-1, keepdim=True)) / torch.std(mu_test, axis=-1, keepdim=True)

                    # mu_test = (mu_test - mu_test.mean(dim=-1, keepdim=True)) / (mu_test.std(dim=-1, keepdim=True) + 1e-6)

                    # 重构形状
                    labels_test = labels_test.view(-1, 1)
                    # pred_ppg_test = pred_ppg_test.view( -1 , 1)
                    mu_norm_test = mu_test.view(-1, 1)
                    # mu_norm_test = mu_norm_test.view(-1, 1)
                    # uncertainty_params_test = uncertainty_params_test.view(-1, 4)  # 展平不确定性参数
                    uncertainty_params_test = uncertainty_params_test.reshape(-1, 4)  # 展平不确定性参数


                    # 存储结果
                    for ib in range(batch_size):
                        subj_index = test_batch[2][ib]
                        sort_index = int(test_batch[3][ib])

                        if subj_index not in predictions.keys():
                            predictions[subj_index] = dict()
                            labels[subj_index] = dict()
                            uncertainty_params[subj_index] = dict()

                        # predictions[subj_index][sort_index] = pred_ppg_test[ib * chunk_len:(ib + 1) * chunk_len]
                        predictions[subj_index][sort_index] = mu_norm_test[ib * chunk_len:(ib + 1) * chunk_len]
                        labels[subj_index][sort_index] = labels_test[ib * chunk_len:(ib + 1) * chunk_len]
                        uncertainty_params[subj_index][sort_index] = uncertainty_params_test[ib * chunk_len:(ib + 1) * chunk_len]
                print(' ')
                calculate_metrics(predictions, labels, self.config)

                # 波形图论文用字号
                waveform_font = {
                    "label": 20,
                    "title": 20,
                    "legend": 15,
                    "tick": 14,
                }

                exp_name = self.model_file_name
                exp_tag = _experiment_tag(exp_name)
                waveform_dir = os.path.join("results", "waveforms", exp_name)
                custom_fig_dir = os.path.join("results", "uncertainty", "edl", f"edl_{exp_tag}")
                os.makedirs(waveform_dir, exist_ok=True)
                os.makedirs(custom_fig_dir, exist_ok=True)

                # 画出每个受试者的预测值和标签值的波形图
                for subj in predictions:
                    subj_preds = []
                    subj_labels = []

                    # 建议排序
                    for seg in sorted(predictions[subj].keys()):
                        pred = predictions[subj][seg].cpu().numpy().flatten()
                        label = labels[subj][seg].cpu().numpy().flatten()

                        subj_preds.append(pred)
                        subj_labels.append(label)

                    subj_preds = np.concatenate(subj_preds)
                    subj_labels = np.concatenate(subj_labels)

                    # =========================
                    # 可视化标准化（仅用于画图）
                    # =========================
                    pred_vis = (subj_preds - subj_preds.mean()) / (subj_preds.std() + 1e-6)
                    label_vis = (subj_labels - subj_labels.mean()) / (subj_labels.std() + 1e-6)

                    subj_str = str(subj)
                    if subj_str.lower().startswith("subject"):
                        subj_id = subj_str[7:] or subj_str
                    else:
                        subj_id = subj_str

                    fig, ax = plt.subplots(figsize=(16, 7))

                    # 使用标准化后的波形
                    ax.plot(pred_vis, label='Predicted', alpha=0.8, linewidth=1.5)
                    ax.plot(label_vis, label='Ground Truth', alpha=0.8, linewidth=1.5)

                    ax.set_xlabel('Sample Index', fontsize=waveform_font["label"])
                    ax.set_ylabel('Normalized Signal', fontsize=waveform_font["label"])
                    ax.set_title(f'Normalized Waveform - Subject {subj_id}', fontsize=waveform_font["title"], pad=18)
                    ax.legend(fontsize=waveform_font["legend"])
                    ax.tick_params(axis='both', which='major', labelsize=waveform_font["tick"])
                    ax.grid(True, alpha=0.3)
                    fig.tight_layout()

                    fig_path = os.path.join(waveform_dir, f'waveform_subj_{subj}.png')

                    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)

                    print(f"Waveform plot for subject {subj} saved to {fig_path}.")


                # # 画出每个受试者的预测值和标签值的波形图
                # for subj in predictions:
                #     subj_preds = []
                #     subj_labels = []
                #     for seg in predictions[subj]:
                #         pred = predictions[subj][seg].cpu().numpy().flatten()
                #         label = labels[subj][seg].cpu().numpy().flatten()
                #         subj_preds.append(pred)
                #         subj_labels.append(label)
                #     subj_preds = np.concatenate(subj_preds)
                #     subj_labels = np.concatenate(subj_labels)
                #
                #     fig, ax = plt.subplots(figsize=(16, 7))
                #     ax.plot(subj_preds, label='Predicted', alpha=0.8, linewidth=1.5)
                #     ax.plot(subj_labels, label='Ground Truth', alpha=0.8, linewidth=1.5)
                #     ax.set_xlabel('Sample Index', fontsize=waveform_font["label"])
                #     ax.set_ylabel('Signal Value', fontsize=waveform_font["label"])
                #     subj_str = str(subj)
                #     if subj_str.lower().startswith("subject"):
                #         subj_id = subj_str[7:] or subj_str
                #     else:
                #         subj_id = subj_str
                #     ax.set_title(f'Predicted vs Ground Truth Waveform - Subject {subj_id}', fontsize=waveform_font["title"], pad=18)
                #     ax.legend(fontsize=waveform_font["legend"])
                #     ax.tick_params(axis='both', which='major', labelsize=waveform_font["tick"])
                #     ax.grid(True, alpha=0.3)
                #     fig.tight_layout()
                #     fig_path = os.path.join(waveform_dir, f'waveform_subj_{subj}.png')
                #     fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                #     plt.close(fig)
                #     print(f"Waveform plot for subject {subj} saved to {fig_path}.")




                # 2. 计算不确定性指标（新增）
                # ===================================
                # uncertainty metrics
                # ===================================

                uncertainty_metrics = \
                    calculate_uncertainty_metrics(
                        uncertainty_params
                    )

                print("\n=== Uncertainty Metrics ===")

                for k, v in uncertainty_metrics.items():
                    print(f'{k}: {v:.6f}')

                # ===================================
                # calibration curve
                # ===================================

                plot_calibration_curve(
                    uncertainty_params,
                    labels,
                    save_dir=custom_fig_dir
                )

                # ===================================
                # Error-Uncertainty Correlation
                # ===================================

                corr_metrics = \
                    calculate_error_uncertainty_correlation(
                        predictions,
                        labels,
                        uncertainty_params,
                        save_dir=custom_fig_dir
                    )

                # ===================================
                # OOD histogram
                # ===================================

                plot_uncertainty_histogram(
                    uncertainty_params,
                    save_dir=custom_fig_dir,
                    dataset_name='OOD_Test'
                )

                # ===================================
                # MAE
                # ===================================

                all_preds = []
                all_labels = []

                for s in predictions.keys():
                    for k in predictions[s].keys():
                        all_preds.extend(
                            predictions[s][k]
                            .cpu()
                            .numpy()
                            .flatten()
                        )

                        all_labels.extend(
                            labels[s][k]
                            .cpu()
                            .numpy()
                            .flatten()
                        )

                all_preds = np.array(all_preds)
                all_labels = np.array(all_labels)

                mae = np.mean(
                    np.abs(all_preds - all_labels)
                )

                # ===================================
                # save OOD statistics
                # ===================================

                save_ood_statistics(
                    dataset_name='OOD_Test',
                    mae=mae,
                    uncertainty_metrics=uncertainty_metrics,
                    save_dir=custom_fig_dir
                )


                # === 绘制置信曲线 ===
                quality_metrics = evaluate_uncertainty_quality(uncertainty_params, labels, save_dir=custom_fig_dir)








    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)


    def data_augmentation(self,data,labels):
        N, D, C, H, W = data.shape
        data_aug = np.zeros((N, D, C, H, W))
        labels_aug = np.zeros((N, D))
        for idx in range(N):
            gt_hr_fft, _  = calculate_hr(labels[idx], labels[idx] , diff_flag = self.diff_flag , fs=self.config.VALID.DATA.FS)
            rand1 = random.random()
            rand2 = random.random()
            rand3 = random.randint(0, D//2-1)
            if rand1 < 0.5 :
                if gt_hr_fft > 90 :
                    for tt in range(rand3,rand3+D):
                        if tt%2 == 0:
                            data_aug[idx,tt-rand3,:,:,:] = data[idx,tt//2,:,:,:]
                            labels_aug[idx,tt-rand3] = labels[idx,tt//2]
                        else:
                            data_aug[idx,tt-rand3,:,:,:] = data[idx,tt//2,:,:,:]/2 + data[idx,tt//2+1,:,:,:]/2
                            labels_aug[idx,tt-rand3] = labels[idx,tt//2]/2 + labels[idx,tt//2+1]/2
                elif gt_hr_fft < 75 :
                    for tt in range(D):
                        if tt < D/2 :
                            data_aug[idx,tt,:,:,:] = data[idx,tt*2,:,:,:]
                            labels_aug[idx,tt] = labels[idx,tt*2]
                        else :
                            data_aug[idx,tt,:,:,:] = data_aug[idx,tt-D//2,:,:,:]
                            labels_aug[idx,tt] = labels_aug[idx,tt-D//2]
                else :
                    data_aug[idx] = data[idx]
                    labels_aug[idx] = labels[idx]
            else :
                data_aug[idx] = data[idx]
                labels_aug[idx] = labels[idx]
        data_aug = torch.tensor(data_aug).float()
        labels_aug = torch.tensor(labels_aug).float()
        if rand2 < 0.5:
            data_aug = torch.flip(data_aug, dims=[4])
        data = data_aug
        labels = labels_aug
        return data,labels

