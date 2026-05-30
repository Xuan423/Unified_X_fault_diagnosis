import torch
import torch.nn as nn
import pytorch_lightning as pl
from model.TSPN import Transparent_Signal_Processing_Network
# from config import args
# from config import signal_processing_modules,feature_extractor_modules
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchmetrics
from .utils import l1_reg,get_all_layers,wgn2,sim_reg,mixup
import numpy as np

def check_attr(args,attr = 'attention_norm'):
    if not hasattr(args, attr):
        setattr(args, attr, False)

class Basic_plmodel(pl.LightningModule):
    def __init__(self, network, args):
        super().__init__()
        self.network = network # placeholder
        self.args = args
        self.loss = nn.CrossEntropyLoss()
        self.acc_val = torchmetrics.Accuracy(task="multiclass", num_classes=args.num_classes)
        self.acc_train = torchmetrics.Accuracy(task="multiclass", num_classes=args.num_classes)
        self.acc_test = torchmetrics.Accuracy(task="multiclass", num_classes=args.num_classes)
        
        args_dict = vars(args)
        self.save_hyperparameters(args_dict,
                                  ignore=['network'])
    
        # print('### network:\n',self.network)
        
    def forward(self, x):
        
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        x = self._apply_snr_if_enabled(x, phase="train")
            
        y_hat = self(x)
        loss = self.loss(y_hat, y.long())
        
        check_attr(self.args,'mixup')
        if self.args.mixup:
            x_mix,y_mix = mixup(batch,alpha = self.args.mixup)
            y_hat_mix = self(x_mix)
            loss += self.loss(y_hat_mix, y_mix.long())

        acc = self.acc_train(y_hat, y.long())  # torch.argmax(y_hat,dim =1)
        
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)  # sync_dist = False lead to BUG
        self.log('train_acc', acc,  on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        
        
        if self.args.l1_norm > 0: # l1 regularization
            # for param in self.network.parameters():  # TODO 只norm linear的权重
            #     regularization_loss += l1_reg(param = param) 
                
            regularization_loss = self.update_regularization_loss()      
                          
            loss += self.args.l1_norm * regularization_loss          
            self.log('l1_loss_', regularization_loss,on_epoch=True,prog_bar =True,sync_dist=True)   
        
        check_attr(self.args,'attention_norm') 
        if self.args.attention_norm:
            attention_loss = self.update_attention_loss()
            loss += self.args.attention_norm * attention_loss
            self.log('attention_loss', attention_loss,on_epoch=True,prog_bar =True,sync_dist=True)
            
        check_attr(self.args,'Energy_loss')
        if self.args.Energy_loss:
            energy_loss = self.update_energy_loss()
            loss += self.args.Energy_loss * energy_loss
            self.log('energy_loss', energy_loss,on_epoch=True,prog_bar =True,sync_dist=True) 
        
        
        return loss

    def _apply_snr_if_enabled(self, x, phase: str):
        """
        Apply additive noise for the given phase.

        By default, noise is applied only in training. You can control behavior via:
        - args.snr_apply: 'train' (default) | 'train_val' | 'all'
        - args.snr_eval: bool (if True, apply in val/test as well)
        - args.snr_mode: 'per_sample' | 'per_batch' | 'per_sample_channel' (default auto-select for multi-channel inputs)
        - args.snr_train / args.snr_eval_db: override SNR (dB) for train vs eval phases
        """
        # Allow using different SNR for train vs eval if configured
        snr_value = None
        if phase == "train" and hasattr(self.args, "snr_train"):
            snr_value = getattr(self.args, "snr_train")
        elif phase in ("val", "validation", "test") and hasattr(self.args, "snr_eval_db"):
            snr_value = getattr(self.args, "snr_eval_db")

        snr_db = self._sample_snr_db(snr_value=snr_value)
        if snr_db is None:
            return x

        snr_eval = bool(getattr(self.args, "snr_eval", False))
        snr_apply = getattr(self.args, "snr_apply", None)
        if snr_apply is None:
            # Backward compatible behavior:
            # - default: apply noise only in training
            # - if snr_eval=True: apply in train/val/test
            snr_apply = "all" if snr_eval else "train"

        snr_apply = str(snr_apply).lower()
        phase = str(phase).lower()

        apply = False
        if snr_apply in ("none", "off", "false", "0"):
            apply = False
        elif snr_apply == "train":
            apply = phase == "train"
        elif snr_apply in ("val", "validation"):
            apply = phase in ("val", "validation")
        elif snr_apply == "test":
            apply = phase == "test"
        elif snr_apply in ("eval", "val_test"):
            apply = phase in ("val", "validation", "test")
        elif snr_apply == "train_val":
            apply = phase in ("train", "val", "validation")
        elif snr_apply == "train_test":
            apply = phase in ("train", "test")
        elif snr_apply in ("all", "train_val_test"):
            apply = True
        else:
            # Unknown value -> fallback to legacy behavior
            apply = True if snr_eval else (phase == "train")

        if not apply:
            return x

        snr_mode = getattr(self.args, "snr_mode", None)
        if snr_mode is None:
            # Default to per-sample-per-channel for multi-channel inputs [B, L, C],
            # which avoids channel energy imbalance causing unintended SNR.
            if x.ndim == 3 and x.shape[-1] > 1:
                snr_mode = "per_sample_channel"
            else:
                snr_mode = "per_sample"

        return wgn2(x, snr_db, mode=snr_mode)

    def _sample_snr_db(self, snr_value=None):
        """
        Return SNR in dB (float) or None if disabled.

        Supported formats:
        - snr: 10            -> fixed 10 dB
        - snr: [0, 10]       -> uniform int in [0, 10]
        - snr: [0.0, 10.0]   -> uniform float in [0.0, 10.0]

        Note: old behavior (random in [0, snr)) is removed because configs
        annotate snr as a fixed noise level (e.g. "10dB, 0dB").
        """
        if snr_value is None and not hasattr(self.args, "snr"):
            return None

        snr = self.args.snr if snr_value is None else snr_value
        if snr is None or snr is False:
            return None
        if isinstance(snr, (int, float)) and snr == 0:
            return None

        if isinstance(snr, (list, tuple)) and len(snr) == 2:
            low, high = snr[0], snr[1]
            if isinstance(low, float) or isinstance(high, float):
                return float(np.random.uniform(low, high))
            low_i, high_i = int(low), int(high)
            if low_i > high_i:
                low_i, high_i = high_i, low_i
            return float(np.random.randint(low_i, high_i + 1))

        return float(snr)

    def update_regularization_loss(self):
        regularization_loss = 0
        for i, (name,param) in enumerate(self.network.named_parameters()):
            if 'WF' not in name:
                regularization_loss += l1_reg(param = param)
        return regularization_loss
    
    def update_attention_loss(self):
        regularization_loss = 0

        for layer in self.network.signal_processing_layers:
            gate_value = layer.channel_attention.gate
            regularization_loss += sim_reg(tensor = gate_value)
            
        gate_value = self.network.feature_extractor_layers.FEAttention.gate
        regularization_loss += sim_reg(tensor = gate_value)
        
        return regularization_loss
    
    def update_energy_loss(self):
        energy_loss = 0
        tfr = self.network.TFR
        if self.args.CI_name == 'Kurtosis':
            power2 = tfr ** 2
            power4 = tfr ** 4
            res = power4.mean(dim=-1)/(power2.mean(dim=-1)**2 + 1e-12)
            return - res.mean()
        elif self.args.CI_name == 'entropy':
            res = (tfr**2 * torch.log(tfr**2 + 1e-12)).sum(dim=-1)
            return res.mean()

        return

    
    
    def validation_step(self, batch, batch_idx):
        # self.eval()
        x, y = batch
        x = self._apply_snr_if_enabled(x, phase="val")
        y_hat = self(x)
        val_loss = self.loss(y_hat, y.long())
        acc = self.acc_val(y_hat, y.long())
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        # return val_loss
    def test_step(self, batch, batch_idx):
        x, y = batch
        x = self._apply_snr_if_enabled(x, phase="test")
        y_hat = self(x)
        test_loss = self.loss(y_hat, y.long())
        self.log('test_loss', test_loss,  on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        acc = self.acc_test(y_hat, y.long())
        self.log('test_acc', acc,  on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        return {'test_acc': acc, 'test_loss': test_loss}
    
    def configure_optimizers(self):
        '''defines model optimizer'''
        optimizer = self.config_different_lr_optimizer()
        # optimizer = Adam(self.parameters(), lr=self.args.learning_rate, weight_decay = self.args.weight_decay)
        out = {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": ReduceLROnPlateau(optimizer),
            "monitor": self.args.monitor ,
            "frequency": self.args.patience//2

            },
        }
        return out

    def _get_optimizer_cls(self):
        opt_name = getattr(self.args, "optimizer", "adam")
        opt_name = str(opt_name).lower()
        if opt_name == "adamw":
            return AdamW
        return Adam

    def config_different_lr_optimizer(self):
        '''config different learning rate for different layers'''
        
        if not hasattr(self.args, 'learnable_parameter_learning_rate'):
            setattr(self.args, 'learnable_parameter_learning_rate', self.args.learning_rate)  # fix bug
            
        layers = get_all_layers(self.network, layers = [])
        # for i, module in enumerate(self.network.children()):
        #     # if not isinstance(module, nn.Sequential):
        #         layers += [l for l in module.children()] if isinstance(module, nn.ModuleList) else [module]
        parameters_conv = []        
        for layer in layers:
            if isinstance(layer, nn.Linear):
                # 假设我们只想为线性层的权重参数设置不同的学习率
                parameters_conv.append({'params': layer.weight, 'lr': self.args.learning_rate, 'weight_decay': self.args.weight_decay})
                # 如果你也想为偏置项设置学习率，可以像这样添加:
                # parameters_conv.append({'params': layer.bias, 'lr': self.args.learning_rate})
        
        # 确保网络中未被上述步骤指定的其它所有参数都有一个默认的学习率
        base_params = filter(lambda p: id(p) not in [id(param['params']) for param in parameters_conv], self.network.parameters())

        optimizer_cls = self._get_optimizer_cls()
        optimizer = optimizer_cls([
            {'params': base_params},
            *parameters_conv
        ], lr=self.args.learnable_parameter_learning_rate, weight_decay=self.args.weight_decay)
        
        return optimizer



