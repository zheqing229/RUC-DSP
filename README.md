# RUC-DSP

---

# 声音检索与分类项目

## 项目简介

本项目是RUC高瓴人工智能学院开设的数字信号处理课程的期末作业。我们小组实现了两项任务：

1. **声音检索任务**：实现 FFT、STFT、MFCC 等信号处理算法，使用最后一个 fold 作为查询声音，前 4 个 fold 作为候选数据库。通过计算相似度，判断 Top10、Top20 中找到相同类别声音的精度。
2. **声音分类任务**：自由选择神经网络模型，对声音进行分类，利用前 4 个 fold 进行训练，最后 1 个 fold 进行测试。并将分类模型应用于检索任务，对比有无机器学习的效果。

## 文件结构

项目目录如下：

```
project/
│
├── ast_retrieval/  
│   ├── data.py          # 数据加载和预处理
│   ├── main.py          # 声音分类任务入口
│   ├── model.py         # 分类模型定义
│   ├── train.py         # 分类模型训练与测试
│
├── dsp/  
│   ├── fft.py           # FFT 实现
│   ├── mfcc.py          # MFCC 特征提取
│   ├── stft.py          # STFT 实现
│
├── retrieval/  
│   ├── dataload.py      # 数据加载
│   ├── evalute.py       # 检索任务评估
│   ├── main.py          # 声音检索任务入口
│
├── pretrained_models/   # 预训练模型存储目录
│
└── README.md            # 项目说明文件
```

## 环境依赖

运行项目需要安装以下依赖：

- Python 3.8+
- NumPy
- Librosa
- PyTorch
- tqdm
- timm
- SciPy

可以使用以下命令安装依赖：

```bash
pip install -r requirements.txt
```

### requirements.txt 文件内容：

```plaintext
numpy
librosa
torch
tqdm
timm
scipy
```

## 使用方法

1. **安装依赖**  
   确保安装好所有依赖库，可使用 `pip install -r requirements.txt`。

2. **运行声音分类任务**  
   使用以下命令运行声音分类任务：
   ```bash
   python ast_retrieval/main.py
   ```

3. **运行声音检索任务**  
   使用以下命令运行声音检索任务：
   ```bash
   python retrieval/main.py
   ```

4. **评估结果**  
   检索任务和分类任务的评估结果将保存在控制台输出，您可以对比 Top10、Top20 的准确率和有无机器学习模型的效果。

## 项目亮点

- 自主实现 FFT、STFT、MFCC 算法，强化信号处理基础。
- 使用深度学习模型（ AST ）完成声音分类，并对检索任务效果进行量化对比。
- 灵活的模块化代码结构，便于扩展和优化。

---
