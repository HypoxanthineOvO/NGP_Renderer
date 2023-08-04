# NGP_Renderer
## Introduction
一个简单的 Instant-NGP 渲染仿真器。

### 实现这个仿真器的主要目的：
1. 督促自己了解和掌握 NGP 的实现细节
2. 适当锻炼代码能力
3. 为 ISCAS 的实验搭建基础框架

### 仿真器的功能
- 读取 `.msgpack` 并正确加载到 tiny-cuda-nn 的网络中（暂时只支持 nerf_synthetic）
  - 是的，连 FOX 都不能跑，问就是 Ray Marching 部分还米有实现好
  - 但这部分并不简单，事实上大部分 NGP 的复现都抛弃了 `.msgpack` ，比如 `torch-ngp`
- 渲染出图片
  - 速度快的感人，这就是 Naive Python Cycle 的速度（

## 后续开发计划？
- 在这个版本里对 Ray Marching 做仿真
- 应该会有一版以 C++ 为基础的项目实现，然后开始设计 C++ 项目的 Python 接口

## 文件结构：
- `main.py` 是目前的主程序
- `main_naive_ngp.py` 是用朴素的 ray marching 加上多像素并行实现的版本，速度更快一点