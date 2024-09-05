
<div align="center">

[English](./README.md) | 简体中文

<h1>UniMERNet: 一个用于真实世界数学表达式识别的通用网络</h1>

[[ 论文 ]](https://arxiv.org/abs/2404.15254) [[ 网站 ]](https://github.com/opendatalab/UniMERNet/tree/main) [[ 数据集 (OpenDataLab)]](https://opendatalab.com/OpenDataLab/UniMER-Dataset) [[ 数据集 (Hugging Face) ]](https://huggingface.co/datasets/wanderkid/UniMER_Dataset)

[[模型 🤗(Hugging Face)]](https://huggingface.co/wanderkid/unimernet_base)
[[模型 <img src="./asset/images/modelscope_logo.png" width="20px">(ModelScope)]](https://www.modelscope.cn/models/wanderkid/unimernet_base)

</div>

欢迎来到 UniMERNet 的官方仓库，这是一个将数学表达式图像转换为 LaTeX 的解决方案，适用于各种真实世界场景。

## 新闻 🚀🚀🚀
**2024.09.06** 🎉🎉  UniMERNet 算法版本更新，新版本设计更小网络结构，速度更快，精度基本保持不变，具体见最新版本论文[UniMERNet](xxx)。  
**2024.07.21** 🎉🎉  基于 [PDF-Extract-Kit](https://github.com/opendatalab/PDF-Extract-Kit) MFD 模型，添加了数学公式检测 (MFD) 教程。  
**2024.06.06** 🎉🎉  开源了 UniMER 数据集的评估代码。  
**2024.05.06** 🎉🎉  开源了 UniMER 数据集，包括用于模型训练的 UniMER-1M 和用于 MER 评估的 UniMER-Test。  
**2024.05.06** 🎉🎉  添加了 Streamlit 公式识别演示，并提供了本地部署应用程序。  
**2024.04.24** 🎉🎉  论文现在可以在 [ArXiv](https://arxiv.org/abs/2404.15254) 上查看。  
**2024.04.24** 🎉🎉  发布了推理代码和检查点。  

## 演示视频
https://github.com/opendatalab/UniMERNet/assets/69186975/ac54c6b9-442c-48b0-95f9-a4a3fce8780b

https://github.com/opendatalab/UniMERNet/assets/69186975/09b71c55-c58a-4792-afc1-d5774880ccf8

## 快速开始

### 克隆仓库并下载模型
```bash
git clone https://github.com/opendatalab/UniMERNet.git
```

```bash
cd UniMERNet/models
# 单独下载模型和分词器或使用 git-lfs
git lfs install
git clone https://huggingface.co/wanderkid/unimernet_base  # 1.3GB  
git clone https://huggingface.co/wanderkid/unimernet_small # 773MB  
git clone https://huggingface.co/wanderkid/unimernet_tiny  # 441MB  

# 你也可以从 ModelScope 下载模型
git clone https://www.modelscope.cn/wanderkid/unimernet_base.git
git clone https://www.modelscope.cn/wanderkid/unimernet_small.git
git clone https://www.modelscope.cn/wanderkid/unimernet_tiny.git
```

### 安装

> 新建一个干净的conda环境

``` bash 
conda create -n unimernet python=3.10

conda activate unimernet
```

> 安装方式1：直接 pip install安装，适合一般用户
```bash
pip install --upgrade unimernet

pip install "unimernet[full]"
```

> 安装方式2：本地安装，适合开发者
```bash
pip install -e ."[full]"
```


### 运行 UniMERNet

1. **Streamlit 应用程序**：使用我们的基于 Streamlit 的 GUI 进行交互和用户友好的体验。此应用程序允许实时公式识别和渲染。

    ```bash
    unimernet_gui
    ```
    确保你已安装最新版本的 UniMERNet (`pip install --upgrade unimernet & pip install "unimernet[full]"`) 以使用 Streamlit GUI 应用程序。

2. **命令行演示**：从图像中预测 LaTeX 代码。

    ```bash
    python demo.py
    ```

3. **Jupyter Notebook 演示**：从图像中识别和渲染公式。

    ```bash
    jupyter-lab ./demo.ipynb
    ```

## 与 SOTA 方法的性能比较（BLEU）。

> UniMERNet 在识别真实世界数学表达式方面显著优于主流模型，在简单打印表达式（SPE）、复杂打印表达式（CPE）、屏幕截图表达式（SCE）和手写表达式（HWE）方面表现出色，如 BLEU 分数评估所示。

![BLEU](./asset/papers/fig1_bleu.jpg)

## 不同方法的可视化结果。

> UniMERNet 在挑战性样本的视觉识别方面表现出色，优于其他方法。

![Visualization](https://github.com/opendatalab/VIGC/assets/69186975/6edcac69-5082-43a2-8095-5681b7a707b9)

## UniMER 数据集
### 介绍
UniMER 数据集是一个专门收集的集合，旨在推进数学表达式识别（MER）领域。它包括全面的 UniMER-1M 训练集，包含超过一百万个实例，代表了多样且复杂的数学表达式，以及精心设计的 UniMER 测试集，用于基准测试 MER 模型在真实世界场景中的表现。数据集详情如下：

**UniMER-1M 训练集：**
  - 总样本数：1,061,791 对 LaTeX-图像对
  - 组成：简洁和复杂的扩展公式表达的平衡混合
  - 目的：训练鲁棒、高精度的 MER 模型，提高识别精度和泛化能力

**UniMER 测试集：**
  - 总样本数：23,757 个，分为四种表达式类型：
    - 简单打印表达式（SPE）：6,762 个样本
    - 复杂打印表达式（CPE）：5,921 个样本
    - 屏幕截图表达式（SCE）：4,742 个样本
    - 手写表达式（HWE）：6,332 个样本
  - 目的：在各种真实世界条件下对 MER 模型进行全面评估

### 数据集下载
你可以从 [OpenDataLab](https://opendatalab.com/OpenDataLab/UniMER-Dataset)（推荐中国用户）或 [HuggingFace](https://huggingface.co/datasets/wanderkid/UniMER_Dataset) 下载数据集。

### 下载 UniMER-Test 数据集

下载 UniMER-1M 数据集并将其解压到以下目录：
```bash
./data/UniMER-1M
```

下载 UniMER-Test 数据集并将其解压到以下目录：
```bash
./data/UniMER-Test
```

## 训练

要训练 UniMERNet 模型，请按照以下步骤操作：

1. **指定训练数据集路径**：打开 `configs/train` 文件夹并设置你的训练数据集路径。

2. **运行训练脚本**：执行以下命令以开始训练过程。

    ```bash
    bash script/train.sh
    ```

### 注意：
- 确保 `configs/train` 文件夹中指定的数据集路径是正确且可访问的。
- 监控训练过程中的任何错误或问题。

## 测试

要测试 UniMERNet 模型，请按照以下步骤操作：

1. **指定测试数据集路径**：打开 `configs/val` 文件夹并设置你的测试数据集路径。

2. **运行测试脚本**：执行以下命令以开始测试过程。

    ```bash
    bash script/test.sh
    ```

### 注意：
- 确保 `configs/val` 文件夹中指定的数据集路径是正确且可访问的。
- `test.py` 脚本将使用指定的测试数据集进行评估。记得将 `test.py` 中的测试集路径更改为你的实际路径。
- 查看测试结果以获取性能指标和潜在问题。

## 数学公式检测教程

公式识别的前提是检测 PDF 或网页截图中公式所在的区域。[PDF-Extract-Kit](https://github.com/opendatalab/PDF-Extract-Kit) 包含了一个强大的公式检测模型。如果你希望自行进行公式检测和识别，可以参考 [公式检测教程](./MFD/README.md) 以获取有关部署和使用公式检测模型的指南。

## 待办事项
 [✅] 发布 UniMERNet 的推理代码和模型。  
 [✅] 发布 UniMER-1M 和 UniMER-Test。  
 [✅] 开源 Streamlit 公式识别 GUI 应用程序。  
 [✅] 发布 UniMERNet 的训练代码。  


## 引用
如果你在研究中发现我们的模型/代码/论文有用，欢迎给我们项目点个 ⭐ 并引用我们的工作 📝，谢谢 :)
```bibtex
@misc{wang2024unimernet,
      title={UniMERNet: A Universal Network for Real-World Mathematical Expression Recognition}, 
      author={Bin Wang and Zhuangcheng Gu and Chao Xu and Bo Zhang and Botian Shi and Conghui He},
      year={2024},
      eprint={2404.15254},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## 致谢
- [VIGC](https://github.com/opendatalab/VIGC)。模型框架依赖于 VIGC。
- [Texify](https://github.com/VikParuchuri/texify)。一个主流的 MER 算法，UniMERNet 的数据处理参考了 Texify。
- [Latex-OCR](https://github.com/lukas-blecher/LaTeX-OCR)。另一个主流的 MER 算法。
- [Donut](https://huggingface.co/naver-clova-ix/donut-base)。UniMERNet 的 Transformer 编码器-解码器参考了 Donut。
- [Nougat](https://github.com/facebookresearch/nougat)。分词器使用了 Nougat。

## 联系我们
如果你有任何问题、意见或建议，请随时通过 wangbin@pjlab.org.cn 联系我们。

## 许可证
[Apache 许可证 2.0](LICENSE)