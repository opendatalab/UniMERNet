<div align="center">
<h1>UniMERNet: A Universal Network for Real-World Mathematical Expression Recognition</h1>


[[ Paper ]](https://arxiv.org/abs/2404.15254) [[ Website ]](https://github.com/opendatalab/UniMERNet/tree/main) [[ Dataset (OpenDataLab)]](https://opendatalab.com/OpenDataLab/UniMER-Dataset) [[ Dataset (Hugging Face) ]](https://huggingface.co/datasets/wanderkid/UniMER_Dataset)
[[Models (Hugging Face)]](https://huggingface.co/wanderkid/unimernet)

</div>

Welcome to the official repository of UniMERNet, a solution that converts images of mathematical expressions into LaTeX, suitable for a wide range of real-world scenarios.

## News 🚀🚀🚀
**2024.05.06** 🎉🎉  Open-sourced UniMER dataset, including UniMER-1M for model training and UniMER-Test for MER evaluation.  
**2024.05.06** 🎉🎉  Added Streamlit formula recognition demo and provided local deployment App.  
**2024.04.24** 🎉🎉  Paper now available on [ArXiv](https://arxiv.org/abs/2404.15254).  
**2024.04.24** 🎉🎉  Inference code and checkpoints have been released. 


## Demo Video
https://github.com/opendatalab/UniMERNet/assets/69186975/ac54c6b9-442c-48b0-95f9-a4a3fce8780b


https://github.com/opendatalab/UniMERNet/assets/69186975/09b71c55-c58a-4792-afc1-d5774880ccf8

## Quick Start

### Clone the repo and download the model
```bash
git clone https://github.com/opendatalab/UniMERNet.git
```

```bash
cd UniMERNet/models
# Download the model and tokenizer individually or use git-lfs
git lfs install
git clone https://huggingface.co/wanderkid/unimernet
```

### Installation

``` bash 
conda create -n unimernet python=3.10

conda activate unimernet

pip install --upgrade unimernet
```

### Running UniMERNet

1. **Streamlit Application**: For an interactive and user-friendly experience, use our Streamlit-based GUI. This application allows real-time formula recognition and rendering.

    ```bash
    unimernet_gui
    ```
    Ensure you have the latest version of UniMERNet installed (`pip install --upgrade unimernet`) for the streamlit GUI application.

2. **Command-line Demo**: Predict LaTeX code from an image.

    ```bash
    python demo.py
    ```

3. **Jupyter Notebook Demo**: Recognize and render formula from an image.

    ```bash
    jupyter-lab ./demo.ipynb
    ```


## Performance Comparison (BLEU) with SOTA Methods.

> UniMERNet significantly outperforms mainstream models in recognizing real-world mathematical expressions, demonstrating superior performance across Simple Printed Expressions (SPE), Complex Printed Expressions (CPE), Screen-Captured Expressions (SCE), and Handwritten Expressions (HWE), as evidenced by the comparative BLEU Score evaluation.  


![BLEU](https://github.com/opendatalab/VIGC/assets/69186975/ec8eb3e2-4ccc-4152-b18c-e86b442e2dcc)



## Visualization Result with Different Methods.

> UniMERNet excels in visual recognition of challenging samples, outperforming other methods.  

![Visualization](https://github.com/opendatalab/VIGC/assets/69186975/6edcac69-5082-43a2-8095-5681b7a707b9)

## UniMER Dataset
### Introduction
The UniMER dataset is a specialized collection curated to advance the field of Mathematical Expression Recognition (MER). It encompasses the comprehensive UniMER-1M training set, featuring over one million instances that represent a diverse and intricate range of mathematical expressions, coupled with the UniMER Test Set, meticulously designed to benchmark MER models against real-world scenarios. The dataset details are as follows:

**UniMER-1M Training Set:**
  - Total Samples: 1,061,791 Latex-Image pairs
  - Composition: A balanced mix of concise and complex, extended formula expressions
  - Aim: To train robust, high-accuracy MER models, enhancing recognition precision and generalization

**UniMER Test Set:**
  - Total Samples: 23,757, categorized into four types of expressions:
    - Simple Printed Expressions (SPE): 6,762 samples
    - Complex Printed Expressions (CPE): 5,921 samples
    - Screen Capture Expressions (SCE): 4,742 samples
    - Handwritten Expressions (HWE): 6,332 samples
  - Purpose: To provide a thorough evaluation of MER models across a spectrum of real-world conditions

### Dataset Download
You can download the dataset from [OpenDataLab](https://opendatalab.com/OpenDataLab/UniMER-Dataset) (recommended for users in China) or [HuggingFace](https://huggingface.co/datasets/wanderkid/UniMER_Dataset).

## TODO

- [x] Release inference code and checkpoints of UniMERNet.
- [x] Release UniMER-1M and UniMER-Test.
- [x] Open-source the Streamlit formula recognition GUI application. 
- [ ] Release the training code for UniMERNet.

## Citation
If you find our models / code / papers useful in your research, please consider giving us a star ⭐ and citing our work 📝, thank you :)
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

## Acknowledgements
- [VIGC](https://github.com/opendatalab/VIGC). The model framework is dependent on VIGC.
- [Texify](https://github.com/VikParuchuri/texify). A mainstream MER algorithm, UniMERNet data processing refers to Texify.
- [Latex-OCR](https://github.com/lukas-blecher/LaTeX-OCR). Another mainstream MER algorithm.
- [Donut](https://huggingface.co/naver-clova-ix/donut-base). The UniMERNet's Transformer Encoder-Decoder are referenced from Donut.
- [Nougat](https://github.com/facebookresearch/nougat). The tokenizer uses Nougat.

## Contact Us
If you have any questions, comments, or suggestions, please do not hesitate to contact us at wangbin@pjlab.org.cn.

## License
[Apache License 2.0](LICENSE)