'''
@Desc:   获取计算机硬件设备信息
@Author: Dysin
@Date:   2025/9/26
'''

import torch
import platform
import psutil

def get_cpu_info():
    # 获取 CPU 信息
    cpu_info = platform.processor()  # 获取 CPU 名称
    cpu_count = psutil.cpu_count(logical=False)  # 获取物理 CPU 核心数

    # 获取内存信息
    memory_info = psutil.virtual_memory()
    total_memory = memory_info.total / (1024**3)  # 转换为 GB
    available_memory = memory_info.available / (1024**3)

    print(f"CPU 信息: {cpu_info}")
    print(f"物理核心数: {cpu_count}")
    print(f"总内存: {total_memory:.2f} GB")
    print(f"可用内存: {available_memory:.2f} GB")

def get_gpu_info():
    # 检查是否有可用的 GPU
    if torch.cuda.is_available():
        print(f"CUDA 设备数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("没有检测到 GPU。")

if __name__ == '__main__':
    get_cpu_info()
    get_gpu_info()
