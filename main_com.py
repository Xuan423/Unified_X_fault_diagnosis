############# learning ############
from cgi import test
from logging import config
from pytorch_lightning import seed_everything

from sklearn.calibration import log
import torch
############# config##########
import argparse
from trainer.trainer_basic import Basic_plmodel
from trainer.trainer_set import trainer_set
from trainer.utils import load_best_model_checkpoint
# from configs.config import args
# from configs.config import signal_processing_modules,feature_extractor_modules
from configs.config import parse_arguments,config_network
import os
import wandb
# os.environ['CUDA_VISIBLE_DEVICES'] = '0' for test ##########################

import numpy as np
from model_collection.Resnet import ResNet, BasicBlock
from model_collection.Sincnet import Sincnet,Sinc_net_m
from model_collection.WKN import WKN,WKN_m
from model_collection.EELM import Dong_ELM
from model_collection.MWA_CNN import A_cSE,Huan_net
from model_collection.TFN.Models.TFN import TFN_Morlet
from model_collection.MCN.models import MCN_GFK, MultiChannel_MCN_GFK
from model_collection.MCN.models import MCN_WFK,MultiChannel_MCN_WFK
from model_collection.Convformer_NSE import convoformer_v1_small
from model_collection.bsstn_flex import BSSTNFlex
import pandas as pd
import multiprocessing
import re
torch.set_float32_matmul_precision('medium')
if __name__ == '__main__':
    # multiprocessing.freeze_support()
    # 创建解析器
    iteration = 5
    parser = argparse.ArgumentParser(description='comparison model')
    # 添加参数
    parser.add_argument('--config_dir', type=str, default='configs/b_bjut/config_bsstn_gen.yaml',help='The directory of the configuration file')

    # parser.add_argument('--config_dir', type=str, default='configs/a_010_SEU/config_MCN_basic.yaml',
    #                     help='The directory of the configuration file')
    meta_args = parser.parse_args()
    config_dir = meta_args.config_dir
    def _format_run_tag(target):
        if target is None:
            return "target_unknown"
        if isinstance(target, list):
            tag = "+".join(str(item) for item in target)
        else:
            tag = str(target)
        return "target" + re.sub(r"[\\/: ]+", "_", tag)

    def _iter_targets(args):
        target_list = getattr(args, "target_list", None)
        if target_list:
            source_list = getattr(args, "source_list", None)
            if source_list:
                if len(source_list) != len(target_list):
                    raise ValueError("source_list length must match target_list length.")
                for source, target in zip(source_list, target_list):
                    yield target, source
            else:
                for target in target_list:
                    yield target, getattr(args, "source", None)
        else:
            yield getattr(args, "target", None), getattr(args, "source", None)

    for it in range(iteration):
        _, base_args, _, _ = parse_arguments(
            config_dir,
            it,
            create_path=False,
            verbose=False,
        )
        for target, source in _iter_targets(base_args):
            run_tag = _format_run_tag(target)
            configs, args, path, name = parse_arguments(
                config_dir,
                it,
                run_tag=run_tag,
                target_override=target,
                source_override=source,
            )
            seed_everything(args.seed + it) # 17 args.seed
            # wandb.init(project=args.dataset_task, name=name)

            ff = np.arange(0, args.in_dim//2 + 1) / args.in_dim//2 + 1

            MODEL_DICT = {
                'Resnet': lambda args: ResNet(BasicBlock, [2, 2, 2, 2], in_channel=args.in_channels, num_class=args.num_classes),
                'WKN_m': lambda args: WKN_m(BasicBlock, [2, 2, 2, 2], in_channel=args.in_channels, num_class=args.num_classes),
                'Sinc_net_m': lambda args: Sinc_net_m(BasicBlock, [2, 2, 2, 2], in_channel=args.in_channels, num_class=args.num_classes),
                'Huan_net': lambda args: Huan_net(input_size=args.in_channels, num_class=args.num_classes),
                'TFN_Morlet': lambda args: TFN_Morlet(in_channels=args.in_channels, out_channels=args.num_classes),
                'MCN_GFK': lambda args: MultiChannel_MCN_GFK(ff=ff, in_channels=args.in_channels, num_MFKs=8, num_classes=args.num_classes),
                'Convoformer_NSE': lambda args: convoformer_v1_small(in_channel=args.in_channels, out_channel=args.num_classes),
                'BSSTN_Flex': lambda args: BSSTNFlex(num_classes=args.num_classes, max_sensors=args.in_channels),
            }

            # 初始化模型
            model_plain = MODEL_DICT[args.model](args)
            model_structure = print(model_plain)
            ############## model train ########## 

            model = Basic_plmodel(model_plain, args)
            trainer,train_dataloader, val_dataloader, test_dataloader = trainer_set(args,path)
            # train
            trainer.fit(model,train_dataloader, val_dataloader)
            model = load_best_model_checkpoint(model,trainer)
            result = trainer.test(model,test_dataloader)

            # 保存结果
            result_df = pd.DataFrame(result)
            result_df.to_csv(os.path.join(path, 'test_result.csv'), index=False)
            # wandb.finish()
