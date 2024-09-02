[English](README.md) | 简体中文 
<div align="center">
<h1>UniMERNet: A Universal Network for Real-World Mathematical Expression Recognition</h1>


[[ 论文 ]](https://arxiv.org/abs/2404.15254) [[ 网站  ]](https://github.com/opendatalab/UniMERNet/tree/main) [[ 数据集 (OpenDataLab)]](https://opendatalab.com/OpenDataLab/UniMER-Dataset) [[ 数据集 (Hugging Face) ]](https://huggingface.co/datasets/wanderkid/UniMER_Dataset)
[[模型 (Hugging Face)]](https://huggingface.co/wanderkid/unimernet)

</div>

欢迎来到 UniMERNet 的官方仓库，这是一个将数学表达式图像转换为 LaTeX 的解决方案，适用于各种现实场景。

## 新闻 🚀🚀🚀
**2024.05.06** 🎉🎉  开源 UniMER 数据集，包括用于模型训练的 UniMER-1M 和用于 MER 评估的 UniMER-Test。  
**2024.05.06** 🎉🎉  添加了 Streamlit 公式识别演示并提供本地部署应用。 
**2024.04.24** 🎉🎉  论文现已在 ArXiv 上发布。 [ArXiv](https://arxiv.org/abs/2404.15254).  
**2024.04.24** 🎉🎉  推理代码和检查点已发布。


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
git clone https://huggingface.co/wanderkid/unimernet
```

### 安装

``` bash 
conda create -n unimernet python=3.10

conda activate unimernet

pip install --upgrade unimernet
```

### 运行 UniMERNet

1. **Streamlit 应用**: 为了获得互动和用户友好的体验，请使用基于 Streamlit 的 GUI。该应用允许实时的公式识别和渲染。

    ```bash
    unimernet_gui
    ```
    确保安装了最新版本的 UniMERNet (`pip install --upgrade unimernet`) 以使用 Streamlit GUI 应用。

2. **命令行演示**: 从图像预测 LaTeX 代码。

    ```bash
    python demo.py
    ```

3. **Jupyter Notebook 演示**: 从图像识别和渲染公式。

    ```bash
    jupyter-lab ./demo.ipynb
    ```


## 性能比较 (BLEU) 与 SOTA 方法

> UniMERNet 在识别现实世界的数学表达式方面显著优于主流模型，在简单印刷表达式 (SPE)、复杂印刷表达式 (CPE)、屏幕捕获表达式 (SCE) 和手写表达式 (HWE) 等方面表现出色，如 BLEU 分数评估所示。


![BLEU](./asset/papers/fig1_bleu.jpg)



## 不同方法的可视化结果。

> UniMERNet 在具有挑战性的样本的视觉识别中表现优异，优于其他方法。

![Visualization](https://github.com/opendatalab/VIGC/assets/69186975/6edcac69-5082-43a2-8095-5681b7a707b9)

## UniMER 数据集
### 介绍
UniMER 数据集是一个专门为推进数学表达式识别 (MER) 领域而策划的集合。它包含全面的 UniMER-1M 训练集，具有超过一百万个代表多样且复杂数学表达式的实例，以及精心设计的 UniMER 测试集，用于基准测试 MER 模型在现实场景中的表现。数据集详情如下：

**UniMER-1M Training Set:**
  - 总样本数：1,061,791 个 LaTeX-图像对
  - 组成：简明和复杂的扩展公式表达的平衡混合
  - 目标：训练出具有高精度的 MER 模型，增强识别精度和泛化能力

**UniMER 测试集:**
  - 总样本数：23,757 个，分为四种表达类型：
    - 简单印刷表达式 (SPE)：6,762 个样本
    - 复杂印刷表达式 (CPE)：5,921 个样本
    - 屏幕捕获表达式 (SCE)：4,742 个样本
    - 手写表达式 (HWE)：6,332 个样本
  - 目的：提供全面评估 MER 模型在现实条件下的表现

### 数据集下载
您可以从 [OpenDataLab](https://opendatalab.com/OpenDataLab/UniMER-Dataset) (推荐给中国的用户) 或 [HuggingFace](https://huggingface.co/datasets/wanderkid/UniMER_Dataset)下载数据集。


## 训练

要训练 UniMERNet 模型，请按照以下步骤操作：

1. **指定训练数据集路径**: 打开 `configs/train` 文件夹并设置您的训练数据集路径。

2. **运行训练脚本**: 执行以下命令开始训练过程。

    ```bash
    bash script/train.sh
    ```

### 注意:
- 确保在 `configs/train` 文件夹中指定的数据集路径正确且可访问。
- 监控训练过程中的错误或问题。

## 测试

要测试 UniMERNet 模型，请按照以下步骤操作：

1. **指定测试数据集路径**: 打开 `configs/val` 文件夹并设置您的测试数据集路径。

2. **运行测试脚本**: 执行以下命令开始测试过程。

    ```bash
    bash script/test.sh
    ```

### 注意:
- 确保在 `configs/val` 文件夹中指定的数据集路径正确且可访问。
- `test.py` 脚本将使用指定的测试数据集进行评估。请记得将 `test.py` 中的测试集路径更改为您的实际路径。
- 查看测试结果以获取性能指标和任何潜在问题。
## TODO

- [✅] 发布 UniMERNet 的推理代码和检查点。
- [✅] 发布 UniMER-1M 和 UniMER-Test。
- [✅] 开源 Streamlit 公式识别 GUI 应用程序。
- [✅] 发布 UniMERNet 的训练代码。

## 引用
如果您在研究中发现我们的模型/代码/论文有用，请考虑给我们一个星⭐并引用我们的工作📝，谢谢 :
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
- [VIGC](https://github.com/opendatalab/VIGC). 模型框架依赖于 VIGC。
- [Texify](https://github.com/VikParuchuri/texify).一种主流的 MER 算法，UniMERNet 数据处理参考了 Texify。
- [Latex-OCR](https://github.com/lukas-blecher/LaTeX-OCR). 另一种主流的 MER 算法。
- [Donut](https://huggingface.co/naver-clova-ix/donut-base). UniMERNet 的 Transformer 编码器-解码器参考了 Donut。
- [Nougat](https://github.com/facebookresearch/nougat). 分词器使用了 Nougat。

## 联系我们
如果您有任何问题、意见或建议，请随时通过 wangbin@pjlab.org.cn 联系我们。

## 许可证
[Apache License 2.0](LICENSE)
