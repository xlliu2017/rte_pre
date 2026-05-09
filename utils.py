import torch
import torch.nn as nn
import gpustat
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.font_manager as fm


def get_least_used_gpu():
    stats = gpustat.GPUStatCollection.new_query()
    ids = map(lambda gpu: int(gpu.entry['index']), stats)
    ratios = map(lambda gpu: float(gpu.entry['memory.used'])/float(gpu.entry['memory.total']), stats)
    bestGPU = min(zip(ids, ratios), key=lambda x: x[1])[0]
    return bestGPU

def check_cuda_mem(device):
    # check memory usage
    print('>>> memory usage: %.3f GB' % (torch.cuda.memory_allocated(device) / 1024**3))
    print('>>> memory cached: %.3f GB' % (torch.cuda.memory_reserved(device) / 1024**3))

def printkeyval(key, value):
    # print key with extra spaces to align values
    print(key + ' ' * (25 - len(key)), value)

def greenprint(print_str):
    print('\033[92m%s\033[0m' % print_str)
    
def redprint(print_str):
    print('\033[91m%s\033[0m' % print_str)

def yellowprint(print_str):
    print('\033[93m%s\033[0m' % print_str)
    
def blueprint(print_str):
    print('\033[94m%s\033[0m' % print_str)

def save_loss_history(dir_path, loss_list, fn):
    # check directory exist
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    with open(dir_path+'%s.txt'%fn, 'w') as f:
        for loss in loss_list:
            f.write('%.5e\n' % loss)
    # save history figure
    plt.semilogy(loss_list)
    plt.savefig(dir_path+'%s.png'%fn)
    plt.close()
    print("loss list saved!")
    
def save_model(dir_path, model, fn):
    """
    Save a PyTorch model's state_dict to a specified directory with filename prefix `fn`.

    Args:
        dir_path (str): Directory to save the model.
        model (torch.nn.Module): The trained PyTorch model.
        fn (str): Filename prefix.
    """
    # Check if directory exists
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    
    # Save model parameters
    save_path = os.path.join(dir_path, f'{fn}.pt')
    torch.save(model.state_dict(), save_path)
    print("model saved!")
    


# timer
import time

class Timer():
    def __init__(self, device='cpu'):
        self.device = device
        if device == 'cpu':
            self.start_time = time.time()
        else:
            self.start_time = torch.cuda.Event(enable_timing=True)
            self.end_time = torch.cuda.Event(enable_timing=True)
    def reset(self):
        device = self.device
        if device == 'cpu':
            self.start_time = time.time()
        else:
            self.start_time.record()
    def elapsed(self, type='ms'):
        elasped_time = 0
        device = self.device
        if device == 'cpu':
            elasped_time = time.time() - self.start_time
        else:
            self.end_time.record()
            torch.cuda.synchronize()
            elasped_time = self.start_time.elapsed_time(self.end_time)
        if type == 'ms':
            return elasped_time
        elif type == 's':
            return elasped_time / 1000
        elif type == 'min':
            return elasped_time / 1000 / 60
        else:
            return elasped_time


def check_chinese_font():
    # 下载思源黑体（首次运行需要下载）
    font_url = "https://github.com/adobe-fonts/source-han-sans/raw/release/OTF/SimplifiedChinese/SourceHanSansSC-Regular.otf"
    font_path = "SourceHanSansSC-Regular.otf"
    # 如果字体文件不存在，下载
    if not os.path.exists(font_path):
        import urllib.request
        print("正在下载中文字体...")
        urllib.request.urlretrieve(font_url, font_path)
        print("下载完成！")
    # 添加字体
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Source Han Sans SC'
    plt.rcParams['axes.unicode_minus'] = False