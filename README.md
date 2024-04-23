<div align="center">
<h1>UniMERNet: A Universal Network for Real-World Mathematical Expression Recognition</h1>

[![Paper](https://img.shields.io/badge/Paper-arxiv)]()
[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Community%20Space-blue)]()

</div>

Welcome to UniMERNet's official repository, providing a solution for converting images of mathematical expressions into LaTeX, suitable for diverse real-world scenarios

## News
**2024.4.24** 🎉🎉   Paper available on [ArXiv](https://arxiv.org/abs/xxx).

## TODO

- [ ] Release inference code and checkpoints of UniMERNet.
- [ ] Release UniMER-1M and UniMER-Test.
- [ ] Release the training code of UniMERNet.


## Performance Comparison (BLEU) with SOTA Methods.
> UniMERNet significantly outperforms mainstream models in recognizing real-world mathematical expressions, exhibiting superior performance across Simple Printed Expressions (SPE), Complex Printed Expressions (CPE), Screen-Captured Expressions (SCE), and Handwritten Expressions (HWE), as evidenced by the comparative BLEU Score evaluation.  
![BLEU](./asset/papers/fig1_BLEU.png)

## Visualization Result with Different Methods.
> UniMERNet excels in visual recognition of challenging samples, outshining other methods.  
![Visualization](./asset/papers/fig5_demo.png)

## Citation
If you find our models / code / papers useful in your research, please consider giving ⭐ and citations 📝, thx :)
```bibtex

```

## Acknowledgement
- [VIGC](https://github.com/opendatalab/VIGC). 
- [Texify](https://github.com/VikParuchuri/texify). 
- [Latex-OCR](https://github.com/lukas-blecher/LaTeX-OCR).
- [Donut](https://huggingface.co/naver-clova-ix/donut-base). 
- [Nougat](https://github.com/facebookresearch/nougat).

## Contact us
If you have any questions, comments or suggestions, please do not hesitate to contact us at wangbin@pjlab.org.cn.

## License
[Apache License 2.0](LICENSE)