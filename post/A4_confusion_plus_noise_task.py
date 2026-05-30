import sys

from pyparsing import line
sys.path.append('./')

from post.A1_plot_config import configure_matplotlib
configure_matplotlib(style='ieee', font_lang='en')
import matplotlib as mpl
mpl.rcParams["text.usetex"] = False          # 关键：别调用 latex
mpl.rcParams["svg.fonttype"] = "none"        # SVG里保留文字为文字（便于后续编辑）
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = [
    "Times New Roman",     # 有就用
    "Times",               # 有些系统会有这个别名
    "TeX Gyre Termes",     # Linux常见Times替代
    "Nimbus Roman",        # Linux常见Times替代
    "Liberation Serif",    # 常见
    "DejaVu Serif",        # Matplotlib自带兜底（强烈建议保留）
]
mpl.rcParams["mathtext.fontset"] = "stix"    # 数学字体更接近论文风格

from torchmetrics import ConfusionMatrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch

import pandas as pd
import numpy as np
###################### heatmap_confusion ######################
from model.TSPN import Transparent_Signal_Processing_Network
from trainer.trainer_basic import Basic_plmodel
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

import torch
from pytorch_lightning import seed_everything
from configs.config import parse_arguments,config_network
import torch.nn.functional as F
import argparse

def heatmap_confusion(predictions, test_labels, args,
                      plot_dir='./plot', name='model', cmap='cool', save_type='.pdf'):
    
    pred_labels = predictions.argmax(dim=1) if len(predictions.shape) > 1 else predictions
    true_labels = test_labels.argmax(dim=1) if len(test_labels.shape) > 1 else test_labels

# 计算混淆矩阵
    conf_mat = ConfusionMatrix(task="multiclass",num_classes=args.num_classes).cuda()
    matrix = conf_mat(pred_labels, true_labels).cpu().numpy()
    # plot_dir 
    pd.DataFrame(matrix).to_csv(os.path.join(plot_dir, f'{name}_confusion_matrix.csv'), index=False)
    
    matrix_normalized = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix_normalized, cmap=cmap, annot=True, fmt=".0%", linewidths=.5,annot_kws={"size": 24})
    # plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{name}_confusion_matrix{save_type}'), transparent=True, dpi=512)
    plt.savefig(os.path.join(plot_dir, f'{name}_confusion_matrix.svg'), transparent=True, dpi=512)
    plt.show()
    plt.close()
    
    return matrix




###################### noise ######################

def add_noise(x, snr):
    """
    向数据添加高斯噪声。
    """
    # snr = 10 ** (snr / 10.0)
    # x_power = torch.sum(data ** 2) / data.numel()
    # noise_power = x_power / snr
    # noise = torch.randn_like(data) * torch.sqrt(noise_power)
    # return data + noise
    snr = 10**(snr/10.0)
    xpower = torch.sum(x**2)/(x.size(0)*x.size(1)*x.size(2))
    npower = xpower / snr
    return torch.randn(x.size()).cuda() * torch.sqrt(npower) + x

def add_noise_perdata(x, snr):
    snr_lin = 10 ** (snr / 10.0)

    # 每个样本功率: [B,1,1]
    xpower = x.pow(2).mean(dim=(1,2), keepdim=True)
    npower = xpower / snr_lin

    noise = torch.randn_like(x) * torch.sqrt(npower)
    return x + noise

def plot_accuracy_vs_snr(test_data, test_labels, model_dict, snr_levels, plot_dir='./plot'):

    # EAFN 用蓝，其余全部用不同灰度（非常像某些顶刊的风格）
    colors = ["#9E5648",
        "#4E606E", 
        "#6A8864", 
        "#A56B45", 
        "#B5BCC2", 
        "#B7C685", 
        "#BD8A99", 
        "#CDBB71"
    ]

    # colors = [
    #     "#4477AA",  # blue
    #     "#EE6677",  # pink-red
    #     "#228833",  # green
    #     "#CCBB44",  # mustard
    #     "#66CCEE",  # cyan
    #     "#AA3377",  # purple
    #     "#BBBBBB",  # light gray
    #     "#000000",  # black
    # ]
    # 
    # colors = ["#CD3B42","#34183E","#4D779B", "#585D5E", "#606060", "#82093B", "#C45C69", "#FFC04D"]




# "#4E606E", "#6A8864", "#9E5648", "#A56B45", "#B5BCC2", "#B7C685", "#BD8A99", "#CDBB71"
# 如果你觉得黄色太浅：把 "#FFD92F" 换成 "#E6AB02" 或 "#666666"



    # marker 尽量选清晰、区分度高的（避免星号太花）
    markers = [
        "o",  # EAFN
        "s",  # M1
        "^",  # M2
        "v",  # M3
        "D",  # M4
        "P",  # M5
        "X",  # M6
        "x",  # M7
    ]

    # 线型固定，保证区分度；EAFN 用实线，其余分散
    line_styles = [
        "-",        # EAFN
        "--",       # M1
        "-.",       # M2
        ":",        # M3
        (0,(5,2)),  # M4
        (0,(3,1)),  # M5
        (0,(1,1)),  # M6
        (0,(7,2,1,2)) # M7
    ]
    
    # lenth = len(model_dict)
    plt.figure(figsize=(10, 6))
    for name, model in model_dict.items():
        file_name = f'{name}_accuracy_vs_snr'
        accuracies = record_noise_accuracy(test_data, test_labels, model, snr_levels, plot_dir, file_name)
        
        
        index = list(model_dict.keys()).index(name)
        
        plt.plot(snr_levels, accuracies,
                 label=name, linewidth=2.8 if name=='EAFN' else 1.8,
                 linestyle=line_styles[index],
                 c = colors[index],
                 marker=markers[index])
        
        plt.xlabel('SNR (dB)')
        plt.ylabel('Accuracy')
        # plt.title('Accuracy vs. SNR')
    plt.legend(loc='best')
    plt.grid(True)
    # plt.savefig(os.path.join(plot_dir, file_name) + '.pdf', dpi=512)
    plt.savefig(os.path.join(plot_dir, file_name) + '.pdf')
    # plt.savefig(os.path.join(plot_dir, file_name) + '.png', dpi=512)
    # plt.show()
    # plt.close()

def record_noise_accuracy(test_data, test_labels, model, snr_levels, plot_dir, file_name):
    accuracies = []
    test_data = torch.tensor(test_data).cuda()
    test_labels = torch.tensor(test_labels).cuda()
    model = model.cuda()
    for snr in snr_levels:
        noisy_data = add_noise_perdata(test_data, snr)
        with torch.no_grad():
            preds = model(noisy_data).argmax(dim=1)
        acc = (preds == test_labels).float().mean().item()
        accuracies.append(acc)
    pd.DataFrame(accuracies, columns=['Accuracy']).to_csv(os.path.join(plot_dir, f'{file_name}_noise_accuracy_list.csv'), index=False)
    model = model.cpu()
    torch.cuda.empty_cache()
    return accuracies

def record_noise_attention(test_data, test_labels, model, snr_levels, plot_dir, file_name):
    accuracies = []
    test_data = torch.tensor(test_data).cuda()
    test_labels = torch.tensor(test_labels).cuda()
    model = model.cuda()
    for snr in snr_levels:
        noisy_data = add_noise(test_data, snr)
        with torch.no_grad():
            preds = model(noisy_data).argmax(dim=1)
            signal_attention = model.signal
                        
        acc = (preds == test_labels).float().mean().item()
        accuracies.append(acc)
    pd.DataFrame(accuracies, columns=['Accuracy']).to_csv(os.path.join(plot_dir, f'{file_name}_noise_accuracy_list.csv'), index=False)
    model = model.cpu()
    torch.cuda.empty_cache()
    return accuracies

def parse_attention(noisy_data,model):
    
    model = model.cuda()
    model(torch.tensor(noisy_data).float().cuda())

    model = model.network
    SP_attentions = []
    for layer in model.signal_processing_layers:
        SP_attentions.append(layer.channel_attention.gate)
    FE_attention = model.feature_extractor_layers.FEAttention.gate
    return SP_attentions,FE_attention

def plot_DIRG():
    # # 设置噪声级别范围
    snr_levels = np.arange(-10, 20, 2)  # 示例：从-5dB到15dB

    # # 进行噪声实验并绘制准确率与SNR的关系图
    test_signal = np.load("/home/lab617/xuanli/dataset/PHMbench_DIRG_020_4096/data_DIRG_020_300Hz_1000N.npy")
    test_data = torch.from_numpy(test_signal).cuda().float()
    print(test_data.shape)
    test_label = np.load("/home/lab617/xuanli/dataset/PHMbench_DIRG_020_4096/label_DIRG_020_300Hz_1000N.npy")
    true_label = torch.from_numpy(test_label).cuda().long()

    parser = argparse.ArgumentParser(description='TSPN')
    parser.add_argument('--config_dir', type=str, default='configs/a_020_DIRG/config_TSPN_basic.yaml',help='The directory of the configuration file')
    # 适用于jupyter
    meta_args = parser.parse_known_args()[0]
    config_dir = meta_args.config_dir
    configs,args,path,name = parse_arguments(config_dir, 0)
    signal_processing_modules, feature_extractor_modules = config_network(configs,args)
    MODEL_DICT = {'TSPN': lambda args: Transparent_Signal_Processing_Network(signal_processing_modules, feature_extractor_modules,args)}
    model_plain = MODEL_DICT[args.model](args)
    model = Basic_plmodel(model_plain, args)
    # 300_1000
    state_dict = torch.load("save/task_DIRG_020_basic/model_TSPN/model_TSPNtime05-13-14-58_datasetDIRG_020_basic_it0_targetDIRG_020_300Hz_1000N/model-epoch=191-val_loss=0.0006-val_acc=1.0000.ckpt")
    # 200_1000
    # state_dict = torch.load("save/task_DIRG_020_basic/model_TSPN/model_TSPNtime05-12-58-20_datasetDIRG_020_basic_it0_targetDIRG_020_200Hz_1000N/model-epoch=96-val_loss=0.0009-val_acc=1.0000.ckpt")
    model.load_state_dict(state_dict['state_dict'])
    model_dict = {'EAFN': model}

    ff = np.arange(0, args.in_dim//2 + 1) / args.in_dim//2 + 1
    COM_MODEL_DICT = {'M1': lambda args: ResNet(BasicBlock, [2, 2, 2, 2], in_channel=args.in_channels, num_class=args.num_classes),
                      'M2': lambda args: Sinc_net_m(BasicBlock, [2, 2, 2, 2], in_channel=args.in_channels, num_class=args.num_classes),
                      'M3': lambda args: Huan_net(input_size=args.in_channels, num_class=args.num_classes),
                      'M4': lambda args: MultiChannel_MCN_GFK(ff=ff, in_channels=args.in_channels, num_MFKs=8, num_classes=args.num_classes),
                      'M5': lambda args: BSSTNFlex(num_classes=args.num_classes, max_sensors=args.in_channels),
                      'M6': lambda args: convoformer_v1_small(in_channel=args.in_channels, out_channel=args.num_classes),
                      'M7': lambda args: TFN_Morlet(in_channels=args.in_channels, out_channels=args.num_classes)}

    com_config_list = ['config_Resnet.yaml','config_Sincnet.yaml','config_MWA_CNN.yaml','config_MCN.yaml','config_bsstn.yaml', 'config_Convoformer_NSE.yaml', 'config_TFN.yaml']
    # # 300_1000
    com_state_dict_list = ['save/task_DIRG_020_basic/model_Resnet/model_Resnettime05-12-53-55_datasetDIRG_020_basic_it0_targetDIRG_020_300Hz_1000N/model-epoch=156-val_loss=0.0176-val_acc=1.0000.ckpt',
                           'save/task_DIRG_020_basic/model_Sinc_net_m/model_Sinc_net_mtime05-12-51-30_datasetDIRG_020_basic_it0_targetDIRG_020_300Hz_1000N/model-epoch=172-val_loss=0.0104-val_acc=1.0000.ckpt',
                           'save/task_DIRG_020_basic/model_Huan_net/model_Huan_nettime05-12-49-37_datasetDIRG_020_basic_it0_targetDIRG_020_300Hz_1000N/model-epoch=182-val_loss=0.0027-val_acc=1.0000.ckpt',
                           'save/task_DIRG_020_basic/model_MCN_GFK/model_MCN_GFKtime05-12-47-37_datasetDIRG_020_basic_it0_targetDIRG_020_300Hz_1000N/model-epoch=112-val_loss=0.0000-val_acc=1.0000.ckpt',
                           'save/task_DIRG_020_basic/model_BSSTN_Flex/model_BSSTN_Flextime05-12-44-58_datasetDIRG_020_basic_it0_targetDIRG_020_300Hz_1000N/model-epoch=73-val_loss=0.0001-val_acc=1.0000.ckpt',
                           'save/task_DIRG_020_basic/model_Convoformer_NSE/model_Convoformer_NSEtime05-12-46-25_datasetDIRG_020_basic_it0_targetDIRG_020_300Hz_1000N/model-epoch=169-val_loss=0.0024-val_acc=1.0000.ckpt',
                           'save/task_DIRG_020_basic/model_TFN_Morlet/model_TFN_Morlettime05-12-41-56_datasetDIRG_020_basic_it0_targetDIRG_020_300Hz_1000N/model-epoch=150-val_loss=0.0021-val_acc=1.0000.ckpt']
    # 200_1000
    # com_state_dict_list = ['save/task_DIRG_020_basic/model_Resnet/model_Resnettime05-12-59-28_datasetDIRG_020_basic_it0_targetDIRG_020_200Hz_1000N/model-epoch=154-val_loss=0.0067-val_acc=1.0000.ckpt',
    #                        'save/task_DIRG_020_basic/model_Sinc_net_m/model_Sinc_net_mtime05-13-00-50_datasetDIRG_020_basic_it0_targetDIRG_020_200Hz_1000N/model-epoch=07-val_loss=0.0186-val_acc=1.0000.ckpt',
    #                        'save/task_DIRG_020_basic/model_Huan_net/model_Huan_nettime05-13-02-03_datasetDIRG_020_basic_it0_targetDIRG_020_200Hz_1000N/model-epoch=181-val_loss=0.0049-val_acc=1.0000.ckpt',
    #                        'save/task_DIRG_020_basic/model_MCN_GFK/model_MCN_GFKtime05-13-02-40_datasetDIRG_020_basic_it0_targetDIRG_020_200Hz_1000N/model-epoch=10-val_loss=0.0027-val_acc=1.0000.ckpt',
    #                        'save/task_DIRG_020_basic/model_BSSTN_Flex/model_BSSTN_Flextime05-13-03-07_datasetDIRG_020_basic_it0_targetDIRG_020_200Hz_1000N/model-epoch=120-val_loss=0.0008-val_acc=1.0000.ckpt',
    #                        'save/task_DIRG_020_basic/model_Convoformer_NSE/model_Convoformer_NSEtime05-13-04-07_datasetDIRG_020_basic_it0_targetDIRG_020_200Hz_1000N/model-epoch=164-val_loss=0.0089-val_acc=1.0000.ckpt',
    #                        'save/task_DIRG_020_basic/model_TFN_Morlet/model_TFN_Morlettime05-13-04-50_datasetDIRG_020_basic_it0_targetDIRG_020_200Hz_1000N/model-epoch=113-val_loss=0.0075-val_acc=1.0000.ckpt']
    for i, com_config in enumerate(com_config_list):
        com_config_dir = f'configs/a_020_DIRG/{com_config}'
        parser = argparse.ArgumentParser(description='comparison model')        
        configs, args, path, name = parse_arguments(
            com_config_dir,
            0,
            run_tag='DIRG_020_300Hz_1000N',
            target_override='DIRG_020_300Hz_1000N',
            source_override=None,
        )

        com_model_plain = COM_MODEL_DICT[list(COM_MODEL_DICT.keys())[i]](args)
        com_model = Basic_plmodel(com_model_plain, args)
        state_dict = torch.load(com_state_dict_list[i])
        com_model.load_state_dict(state_dict['state_dict'])
        model_dict[list(COM_MODEL_DICT.keys())[i]] = com_model
    plot_accuracy_vs_snr(test_data, true_label, model_dict, snr_levels, plot_dir='./post/')

def plot_SEU():
    snr_levels = np.arange(-10, 20, 2)  # 示例：从-5dB到15dB
    # # 进行噪声实验并绘制准确率与SNR的关系图
    test_signal = np.load("/home/lab617/xuanli/dataset/SEU_bearing/SEU_bearing_30Hz_2_data.npy")
    test_data = torch.from_numpy(test_signal).cuda().float()
    print(test_data.shape)
    test_label = np.load("/home/lab617/xuanli/dataset/SEU_bearing/SEU_bearing_30Hz_2_label.npy")
    true_label = torch.from_numpy(test_label).cuda().long()

    parser = argparse.ArgumentParser(description='TSPN')
    parser.add_argument('--config_dir', type=str, default='configs/a_010_SEU/config_basic.yaml',help='The directory of the configuration file')
    # 适用于jupyter
    meta_args = parser.parse_known_args()[0]
    config_dir = meta_args.config_dir
    configs,args,path,name = parse_arguments(config_dir, 0)
    signal_processing_modules, feature_extractor_modules = config_network(configs,args)
    MODEL_DICT = {'TSPN': lambda args: Transparent_Signal_Processing_Network(signal_processing_modules, feature_extractor_modules,args)}
    model_plain = MODEL_DICT[args.model](args)
    model = Basic_plmodel(model_plain, args)
    # 加噪声
    state_dict = torch.load("save/task_SEU_010_Basic/model_TSPN/model_TSPNtime05-15-34-35_datasetSEU_010_Basic_it0_targetSEU_bearing_30Hz_2/model-epoch=154-val_loss=0.0001-val_acc=1.0000.ckpt")
    # state_dict = torch.load("save/task_SEU_010_Basic/model_TSPN/model_TSPNtime05-15-40-39_datasetSEU_010_Basic_it0_targetSEU_bearing_30Hz_2/model-epoch=180-val_loss=0.0003-val_acc=1.0000.ckpt")

    model.load_state_dict(state_dict['state_dict'])
    model_dict = {'EAFN': model}

    ff = np.arange(0, args.in_dim//2 + 1) / args.in_dim//2 + 1
    COM_MODEL_DICT = {'M1': lambda args: ResNet(BasicBlock, [2, 2, 2, 2], in_channel=args.in_channels, num_class=args.num_classes),
                      'M2': lambda args: Sinc_net_m(BasicBlock, [2, 2, 2, 2], in_channel=args.in_channels, num_class=args.num_classes),
                      'M3': lambda args: Huan_net(input_size=args.in_channels, num_class=args.num_classes),
                      'M4': lambda args: MultiChannel_MCN_GFK(ff=ff, in_channels=args.in_channels, num_MFKs=32, num_classes=args.num_classes),
                      'M5': lambda args: BSSTNFlex(num_classes=args.num_classes, max_sensors=args.in_channels),
                      'M6': lambda args: convoformer_v1_small(in_channel=args.in_channels, out_channel=args.num_classes),
                      'M7': lambda args: TFN_Morlet(in_channels=args.in_channels, out_channels=args.num_classes)}
    com_config_list = ['config_Resnet_basic.yaml',
                       'config_Sincnet.yaml',
                       'config_MWA_CNN_basic.yaml',
                       'config_MCN_basic.yaml',
                       'config_BSSTN_basic.yaml',
                       'config_Convoformer_NSE_basic.yaml',
                       'config_TFN_basic.yaml']
    # 30 no noise
    # com_state_dict_list = ['save/task_SEU_010_Basic/model_Resnet/model_Resnettime05-14-38-30_datasetSEU_010_Basic_it0_targetSEU_bearing_30Hz_2/model-epoch=122-val_loss=0.1049-val_acc=1.0000.ckpt',
    #                        'save/task_SEU_010_Basic/model_Sinc_net_m/model_Sinc_net_mtime05-14-42-27_datasetSEU_010_Basic_it0_targetSEU_bearing_30Hz_2/model-epoch=190-val_loss=0.0719-val_acc=0.9920.ckpt',
    #                        'save/task_SEU_010_Basic/model_Huan_net/model_Huan_nettime05-14-45-44_datasetSEU_010_Basic_it0_targetSEU_bearing_30Hz_2/model-epoch=14-val_loss=0.0684-val_acc=1.0000.ckpt',
    #                        'save/task_SEU_010_Basic/model_MCN_GFK/model_MCN_GFKtime05-15-29-06_datasetSEU_010_Basic_it0_targetSEU_bearing_30Hz_2/model-epoch=178-val_loss=0.1608-val_acc=1.0000.ckpt',
    #                        'save/task_SEU_010_Basic/model_BSSTN_Flex/model_BSSTN_Flextime05-14-54-18_datasetSEU_010_Basic_it0_targetSEU_bearing_30Hz_2/model-epoch=158-val_loss=0.0002-val_acc=1.0000.ckpt',
    #                        'save/task_SEU_010_Basic/model_Convoformer_NSE/model_Convoformer_NSEtime05-14-56-31_datasetSEU_010_Basic_it0_targetSEU_bearing_30Hz_2/model-epoch=167-val_loss=0.0003-val_acc=1.0000.ckpt',
    #                        'save/task_SEU_010_Basic/model_TFN_Morlet/model_TFN_Morlettime05-14-34-53_datasetSEU_010_Basic_it0_targetSEU_bearing_30Hz_2/model-epoch=121-val_loss=0.0000-val_acc=1.0000.ckpt']
    
    com_state_dict_list = ['save/task_SEU_010_Basic/model_Resnet/model_Resnettime05-14-38-30_datasetSEU_010_Basic_it0_targetSEU_bearing_30Hz_2/model-epoch=122-val_loss=0.1049-val_acc=1.0000.ckpt',
                           'save/task_SEU_010_Basic/model_Sinc_net_m/model_Sinc_net_mtime05-14-42-27_datasetSEU_010_Basic_it0_targetSEU_bearing_30Hz_2/model-epoch=190-val_loss=0.0719-val_acc=0.9920.ckpt',
                           'save/task_SEU_010_Basic/model_Huan_net/model_Huan_nettime05-14-45-44_datasetSEU_010_Basic_it0_targetSEU_bearing_30Hz_2/model-epoch=14-val_loss=0.0684-val_acc=1.0000.ckpt',
                           'save/task_SEU_010_Basic/model_MCN_GFK/model_MCN_GFKtime05-15-29-06_datasetSEU_010_Basic_it0_targetSEU_bearing_30Hz_2/model-epoch=178-val_loss=0.1608-val_acc=1.0000.ckpt',
                           'save/task_SEU_010_Basic/model_BSSTN_Flex/model_BSSTN_Flextime05-15-58-41_datasetSEU_010_Basic_it0_targetSEU_bearing_30Hz_2/model-epoch=184-val_loss=0.0004-val_acc=1.0000.ckpt',
                           'save/task_SEU_010_Basic/model_Convoformer_NSE/model_Convoformer_NSEtime05-14-56-31_datasetSEU_010_Basic_it0_targetSEU_bearing_30Hz_2/model-epoch=167-val_loss=0.0003-val_acc=1.0000.ckpt',
                           'save/task_SEU_010_Basic/model_TFN_Morlet/model_TFN_Morlettime05-14-34-53_datasetSEU_010_Basic_it0_targetSEU_bearing_30Hz_2/model-epoch=121-val_loss=0.0000-val_acc=1.0000.ckpt']
    for i, com_config in enumerate(com_config_list):
        com_config_dir = f'configs/a_010_SEU/{com_config}'
        parser = argparse.ArgumentParser(description='comparison model')        
        configs, args, path, name = parse_arguments(
            com_config_dir,
            0,
            run_tag='SEU_bearing_30Hz_2',
            target_override='SEU_bearing_30Hz_2',
            source_override=None,
        )

        com_model_plain = COM_MODEL_DICT[list(COM_MODEL_DICT.keys())[i]](args)
        com_model = Basic_plmodel(com_model_plain, args)
        state_dict = torch.load(com_state_dict_list[i])
        com_model.load_state_dict(state_dict['state_dict'])
        model_dict[list(COM_MODEL_DICT.keys())[i]] = com_model
    plot_accuracy_vs_snr(test_data, true_label, model_dict, snr_levels, plot_dir='./post/')

if __name__ == '__main__':
    plot_DIRG()

