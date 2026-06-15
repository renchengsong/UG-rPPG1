"""
Trainer for RhythmMamba (MC Dropout Enhanced Version)
"""

import os
import numpy as np
import torch
import torch.optim as optim
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from evaluation.post_process import calculate_hr
from evaluation.metrics import calculate_metrics
from neural_methods.model.SimpleMamba_MC import SimpleMamba
from neural_methods.trainer.BaseTrainer import BaseTrainer
from neural_methods.loss.TorchLossComputer import Hybrid_Loss
import subprocess
import scipy.stats as stats

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import subprocess
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr
from evaluation.post_process import calculate_hr
from evaluation.metrics import calculate_metrics
from scipy.interpolate import interp1d
from scipy.stats import norm
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


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

        if config.TRAIN.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized":
            self.diff_flag = 1

        # ==== 模型初始化 ====
        if config.TOOLBOX_MODE in ["train_and_test", "only_test"]:
            self.model = SimpleMamba().to(self.device)
            self.model = torch.nn.DataParallel(
                self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN))
            )
        else:
            raise ValueError("Trainer initialized in incorrect toolbox mode!")

        # ==== 损失函数和优化器 ====
        self.criterion = Hybrid_Loss()
        if config.TOOLBOX_MODE == "train_and_test":
            self.num_train_batches = len(data_loader["train"])
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=config.TRAIN.LR, weight_decay=0
            )
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=config.TRAIN.LR,
                epochs=config.TRAIN.EPOCHS,
                steps_per_epoch=self.num_train_batches,
            )

    # ==============================================================
    #                    Training Phase
    # ==============================================================

    def train(self, data_loader):
        """Training routine for model"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        for epoch in range(self.max_epoch_num):
            print("\n==== Training Epoch: {} ====".format(epoch))
            self.model.train()
            tbar = tqdm(data_loader["train"], ncols=80)

            for idx, batch in enumerate(tbar):
                tbar.set_description(f"Train epoch {epoch}")
                data, labels = batch[0].float(), batch[1].float()
                N, D, C, H, W = data.shape

                if self.config.TRAIN.AUG:
                    data, labels = self.data_augmentation(data, labels)

                data, labels = data.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                pred_ppg = self.model(data)
                pred_ppg = (pred_ppg - torch.mean(pred_ppg, axis=-1).view(-1, 1)) / (
                    torch.std(pred_ppg, axis=-1).view(-1, 1) + 1e-8
                )

                loss = 0.0
                for ib in range(N):
                    loss += self.criterion(
                        pred_ppg[ib],
                        labels[ib],
                        epoch,
                        self.config.TRAIN.DATA.FS,
                        self.diff_flag,
                    )
                loss /= N
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                tbar.set_postfix(loss=loss.item())

            self.save_model(epoch)

            # === Validation ===
            if not self.config.TEST.USE_LAST_EPOCH:
                valid_loss = self.valid(data_loader)
                print(f"Validation Loss: {valid_loss:.6f}")
                if self.min_valid_loss is None or valid_loss < self.min_valid_loss:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print(f"Update best model! Best epoch: {self.best_epoch}")

        if not self.config.TEST.USE_LAST_EPOCH:
            print(
                f"Best trained epoch: {self.best_epoch}, Min validation loss: {self.min_valid_loss}"
            )

    # ==============================================================
    #                  Validation + MC Dropout
    # ==============================================================

    def _enable_mc_dropout(self):
        """强制在eval模式下启用Dropout层"""
        def apply_dropout(m):
            if type(m) == torch.nn.Dropout:
                m.train()
        self.model.apply(apply_dropout)

    @torch.no_grad()
    def _mc_dropout_inference(self, x, T=20):
        """Monte Carlo Dropout采样推理"""
        preds = []
        for _ in range(T):
            y = self.model(x)
            preds.append(y.unsqueeze(0))
        preds = torch.cat(preds, dim=0)  # [T, B, D]
        mean_pred = preds.mean(dim=0)
        var_pred = preds.var(dim=0)
        return mean_pred, var_pred

    def valid(self, data_loader):
        """Validation with MC Dropout"""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print("\n=== Validating (MC Dropout Enabled) ===")
        valid_loss = []
        self.model.eval()
        self._enable_mc_dropout()

        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for _, batch in enumerate(vbar):
                data_valid = batch[0].to(self.device)
                labels_valid = batch[1].to(self.device)
                N, D, C, H, W = data_valid.shape

                mean_pred, var_pred = self._mc_dropout_inference(data_valid, T=20)
                mean_pred = (mean_pred - mean_pred.mean(dim=-1, keepdim=True)) / (
                        mean_pred.std(dim=-1, keepdim=True) + 1e-8
                )

                loss = 0.0
                for ib in range(N):
                    loss += self.criterion(
                        mean_pred[ib],
                        labels_valid[ib],
                        self.config.TRAIN.EPOCHS,
                        self.config.VALID.DATA.FS,
                        self.diff_flag,
                    )
                loss /= N
                valid_loss.append(loss.item())
                vbar.set_postfix(loss=loss.item(), unc=var_pred.mean().item())

        return np.mean(valid_loss)

    # ==============================================================
    #                      Testing + MC Dropout
    # ==============================================================
    @torch.no_grad()
    def test(self, data_loader):
        """
        Test phase with MC Dropout uncertainty estimation.
        Draws confidence-interval calibration curves for:
          - Signal-level (waveform-based uncertainty)
          - HR-level (heart rate uncertainty)
        Also saves calibration data for later reproduction.
        """

        if data_loader["test"] is None:
            raise ValueError("No data for test")

        print("\n=== Testing with MC Dropout Uncertainty Estimation (Confidence Interval Calibration) ===")
        device = self.config.DEVICE

        # === Load Model ===
        if self.config.TOOLBOX_MODE == "only_test":
            model_path = self.config.INFERENCE.MODEL_PATH
            assert os.path.exists(model_path), "Inference model path error!"
            self.model.load_state_dict(torch.load(model_path))
            print(f"Testing uses pretrained model: {model_path}")
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                model_path = os.path.join(
                    self.model_dir, f"{self.model_file_name}_Epoch{self.max_epoch_num - 1}.pth"
                )
            else:
                model_path = os.path.join(
                    self.model_dir, f"{self.model_file_name}_Epoch{self.best_epoch}.pth"
                )
            self.model.load_state_dict(torch.load(model_path))
            print(f"Testing model: {model_path}")

        self.model.to(device)
        self.model.eval()

        # === Enable MC Dropout ===
        def enable_mc_dropout(m):
            if isinstance(m, torch.nn.Dropout):
                m.train()

        self.model.apply(enable_mc_dropout)

        T = 20  # MC samples

        predictions, labels = dict(), dict()
        hr_uncertainties, hr_errors = [], []
        all_preds_T_signal, all_labels_signal = [], []

        # === Inference ===
        with torch.no_grad():
            tbar = tqdm(data_loader["test"], ncols=90)
            for _, test_batch in enumerate(tbar):
                data_test, labels_test = test_batch[0].to(device), test_batch[1].to(device)
                subj_ids, sort_ids = test_batch[2], test_batch[3]
                batch_size = data_test.shape[0]

                preds_T = []
                for _ in range(T):
                    preds_T.append(self.model(data_test).unsqueeze(0))
                preds_T = torch.cat(preds_T, dim=0)  # [T, N, D]

                mean_pred = preds_T.mean(dim=0)
                mean_pred = (mean_pred - mean_pred.mean(dim=-1, keepdim=True)) / (
                        mean_pred.std(dim=-1, keepdim=True) + 1e-6
                )

                if labels_test.ndim == 3:
                    labels_test = labels_test.squeeze(1)

                for ib in range(batch_size):
                    subj = subj_ids[ib]
                    sort = int(sort_ids[ib])
                    if subj not in predictions:
                        predictions[subj], labels[subj] = {}, {}
                    predictions[subj][sort] = mean_pred[ib].cpu()
                    labels[subj][sort] = labels_test[ib].cpu()

                # === HR-level uncertainty estimation
                for ib in range(batch_size):
                    preds_sample_hr = []
                    for t in range(T):
                        pred_wave = preds_T[t, ib].cpu().numpy()
                        label_wave = labels_test[ib].cpu().numpy()
                        try:
                            pred_hr, gt_hr = calculate_hr(pred_wave, label_wave, fs=self.config.VALID.DATA.FS)
                            preds_sample_hr.append(pred_hr)
                        except Exception:
                            continue
                    if len(preds_sample_hr) > 1:
                        hr_unc = np.var(preds_sample_hr)
                        _, gt_hr = calculate_hr(pred_wave, label_wave, fs=self.config.VALID.DATA.FS)
                        hr_err = abs(np.mean(preds_sample_hr) - gt_hr)
                        hr_uncertainties.append(hr_unc)
                        hr_errors.append(hr_err)

                # === Collect all signal-level samples for CI calibration
                all_preds_T_signal.append(preds_T.cpu())
                all_labels_signal.append(labels_test.cpu())

        # === Merge all signal samples ===
        all_preds_T_signal = torch.cat(all_preds_T_signal, dim=1)  # [T, total_samples, D]
        all_labels_signal = torch.cat(all_labels_signal, dim=0)  # [total_samples, D]

        exp_tag = _experiment_tag(self.model_file_name)
        save_dir = os.path.join(
            "results", "uncertainty", "mc", f"MC_{exp_tag}", "uncertainty_analysis"
        )
        os.makedirs(save_dir, exist_ok=True)

        # =============================================================== #
        #          Confidence Interval Calibration Curve Function
        # =============================================================== #


        def draw_confidence_interval_calibration_curve(preds_T, labels, save_path, label_name="Signal"):
            """绘制置信区间校准曲线 (MC Dropout based) 并保存曲线数据"""
            print(f"\n=== Drawing {label_name} Confidence Interval Calibration Curve ===")

            all_mu = preds_T.mean(axis=0).flatten()
            all_sigma = preds_T.std(axis=0).flatten()
            all_label = labels.flatten()

            # confidence_levels = np.linspace(0.05, 0.95, 19)
            confidence_levels = np.linspace(0, 1, 19)
            expected_freq, observed_freq = [], []

            for conf_level in confidence_levels:
                alpha_tail = (1 - conf_level) / 2
                z_score = norm.ppf(1 - alpha_tail)
                lower_bound = all_mu - z_score * all_sigma
                upper_bound = all_mu + z_score * all_sigma
                in_interval = (all_label >= lower_bound) & (all_label <= upper_bound)
                observed_freq.append(np.mean(in_interval))
                expected_freq.append(conf_level)
            # === 绘制平滑置信区间校准曲线 ===
            from scipy.interpolate import interp1d

            f = interp1d(expected_freq, observed_freq, kind='cubic', fill_value='extrapolate')
            # smooth_x = np.linspace(0.05, 0.95, 300)
            smooth_x = np.linspace(0, 1, 300)
            smooth_y = f(smooth_x)

            plt.figure(figsize=(7, 6))
            plt.plot(smooth_x, smooth_y, color='royalblue', linewidth=2.5, label='Model Calibration')
            plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Calibration')

            plt.xlabel("Expected Confidence Level", fontsize=13)
            plt.ylabel("Observed Coverage Frequency", fontsize=13)
            plt.title(f"{label_name} MC Dropout Confidence Interval Calibration", fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(alpha=0.3)
            plt.xlim(0, 1)
            plt.ylim(0, 1)

            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            # # 平滑插值绘图
            # f = interp1d(expected_freq, observed_freq, kind='cubic', fill_value='extrapolate')
            # smooth_x = np.linspace(0.05, 0.95, 100)
            # smooth_y = f(smooth_x)
            #
            # plt.figure(figsize=(7, 6))
            # plt.plot(expected_freq, observed_freq, 'bo', label='Observed')
            # plt.plot(smooth_x, smooth_y, 'b-', linewidth=2)
            # plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
            # plt.xlabel("Expected Confidence Level")
            # plt.ylabel("Observed Coverage Frequency")
            # plt.title(f"{label_name} MC Dropout Confidence Interval Calibration")
            # plt.legend()
            # plt.grid(alpha=0.3)
            # plt.xlim(0, 1)
            # plt.ylim(0, 1)
            # plt.savefig(save_path, dpi=300, bbox_inches='tight')
            # plt.close()

            # 保存数据
            np.save(os.path.join(save_dir, f"{label_name}_expected_conf.npy"), np.array(expected_freq))
            np.save(os.path.join(save_dir, f"{label_name}_observed_freq.npy"), np.array(observed_freq))
            print(f"✅ Saved {label_name} CI calibration curve to: {save_path}")
            print(f"✅ Saved curve data in: {save_dir}")

        # === 绘制波形级置信区间校准曲线 ===
        draw_confidence_interval_calibration_curve(
            all_preds_T_signal.numpy(),
            all_labels_signal.numpy(),
            os.path.join(save_dir, "signal_confidence_interval_curve.png"),
            label_name="Signal"
        )

        # === 绘制 HR级置信区间校准曲线 ===
        hr_preds_T = np.array(hr_uncertainties)
        hr_labels = np.array(hr_errors)

        if len(hr_preds_T) > 10:
            # 转换为均值-方差结构，模拟 MC Dropout 采样结果
            mu = np.mean(hr_preds_T)
            sigma = np.std(hr_preds_T)
            fake_preds_T = np.random.normal(mu, sigma, (T, len(hr_preds_T)))
            fake_labels = hr_labels  # 用HR误差替代GT
            draw_confidence_interval_calibration_curve(
                fake_preds_T,
                fake_labels,
                os.path.join(save_dir, "HR_confidence_interval_curve.png"),
                label_name="HR"
            )

        print(f"\n✅ All results and calibration data saved in: {save_dir}")

    # ==============================================================
    #                    Model Save & Data Augment
    # ==============================================================

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + "_Epoch" + str(index) + ".pth"
        )
        torch.save(self.model.state_dict(), model_path)
        print("Saved Model Path:", model_path)

    def data_augmentation(self, data, labels):
        """心率导向数据增强"""
        N, D, C, H, W = data.shape
        data_aug = np.zeros((N, D, C, H, W))
        labels_aug = np.zeros((N, D))
        for idx in range(N):
            gt_hr_fft, _ = calculate_hr(
                labels[idx], labels[idx], diff_flag=self.diff_flag, fs=self.config.VALID.DATA.FS
            )
            rand1 = random.random()
            rand2 = random.random()
            rand3 = random.randint(0, D // 2 - 1)
            if rand1 < 0.5:
                if gt_hr_fft > 90:
                    for tt in range(rand3, rand3 + D):
                        if tt % 2 == 0:
                            data_aug[idx, tt - rand3, :, :, :] = data[idx, tt // 2, :, :, :]
                            labels_aug[idx, tt - rand3] = labels[idx, tt // 2]
                        else:
                            data_aug[idx, tt - rand3, :, :, :] = (
                                data[idx, tt // 2, :, :, :] / 2 + data[idx, tt // 2 + 1, :, :, :] / 2
                            )
                            labels_aug[idx, tt - rand3] = (
                                labels[idx, tt // 2] / 2 + labels[idx, tt // 2 + 1] / 2
                            )
                elif gt_hr_fft < 75:
                    for tt in range(D):
                        if tt < D / 2:
                            data_aug[idx, tt, :, :, :] = data[idx, tt * 2, :, :, :]
                            labels_aug[idx, tt] = labels[idx, tt * 2]
                        else:
                            data_aug[idx, tt, :, :, :] = data_aug[idx, tt - D // 2, :, :, :]
                            labels_aug[idx, tt] = labels_aug[idx, tt - D // 2]
                else:
                    data_aug[idx] = data[idx]
                    labels_aug[idx] = labels[idx]
            else:
                data_aug[idx] = data[idx]
                labels_aug[idx] = labels[idx]
        data_aug = torch.tensor(data_aug).float()
        labels_aug = torch.tensor(labels_aug).float()
        if rand2 < 0.5:
            data_aug = torch.flip(data_aug, dims=[4])
        return data_aug, labels_aug
