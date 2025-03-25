<div align="center">

[English](./README.md) | [简体中文](./README-zh_CN.md) | 日本語

<h1>UniMERNet: 実世界の数式認識のためのユニバーサルネットワーク</h1>

[[ 論文 ]](https://arxiv.org/abs/2404.15254) [[ ウェブサイト ]](https://github.com/opendatalab/UniMERNet/tree/main) [[ データセット (OpenDataLab)]](https://opendatalab.com/OpenDataLab/UniMER-Dataset) [[ データセット (Hugging Face) ]](https://huggingface.co/datasets/wanderkid/UniMER_Dataset)

[[モデル 🤗(Hugging Face)]](https://huggingface.co/wanderkid/unimernet_base)
[[モデル <img src="./asset/images/modelscope_logo.png" width="20px">(ModelScope)]](https://www.modelscope.cn/models/wanderkid/unimernet_base)

🔥🔥 [CDM: 公平で正確な数式認識評価のための信頼できる指標](https://github.com/opendatalab/UniMERNet/tree/main/cdm)

</div>

UniMERNetの公式リポジトリへようこそ。これは、数式の画像をLaTeXに変換するソリューションであり、さまざまな実世界のシナリオに適しています。

## ニュース 🚀🚀🚀
**2025.03.25** 🎉🎉 <font color="red">新しい数式認識指標[CDM](https://arxiv.org/abs/2409.03643)に関する論文がCVPR 2025に採択されました。ぜひご利用ください。</font>  
**2024.09.06** 🎉🎉  UniMERNetの更新: 新バージョンはモデルが小さくなり、推論が高速化されました。トレーニングコードがオープンソース化されました。詳細は最新の論文[UniMERNet](https://arxiv.org/abs/2404.15254)をご覧ください。     
**2024.09.06** 🎉🎉  数式認識の新しい指標を導入: [CDM](https://github.com/opendatalab/UniMERNet/tree/main/cdm)。BLEU/EditDistanceと比較して、CDMはより直感的で正確な評価スコアを提供し、数式表現の多様性に影響されずに異なるモデルの公平な比較を可能にします。  
**2024.07.21** 🎉🎉  [PDF-Extract-Kit](https://github.com/opendatalab/PDF-Extract-Kit) MFDモデルに基づく数式検出（MFD）チュートリアルを追加しました。    
**2024.06.06** 🎉🎉  UniMERデータセットの評価コードをオープンソース化しました。  
**2024.05.06** 🎉🎉  UniMERデータセットをオープンソース化しました。これには、モデルトレーニング用のUniMER-1MとMER評価用のUniMER-Testが含まれます。  
**2024.05.06** 🎉🎉  Streamlit数式認識デモを追加し、ローカルデプロイメントアプリを提供しました。  
**2024.04.24** 🎉🎉  論文が[ArXiv](https://arxiv.org/abs/2404.15254)で公開されました。  
**2024.04.24** 🎉🎉  推論コードとチェックポイントがリリースされました。  

## デモビデオ
https://github.com/opendatalab/UniMERNet/assets/69186975/ac54c6b9-442c-48b0-95f9-a4a3fce8780b

https://github.com/opendatalab/UniMERNet/assets/69186975/09b71c55-c58a-4792-afc1-d5774880ccf8

## クイックスタート

### リポジトリをクローンし、モデルをダウンロード
```bash
git clone https://github.com/opendatalab/UniMERNet.git
```

```bash
cd UniMERNet/models
# モデルとトークナイザーを個別にダウンロードするか、git-lfsを使用
git lfs install
git clone https://huggingface.co/wanderkid/unimernet_base  # 1.3GB  
git clone https://huggingface.co/wanderkid/unimernet_small # 773MB  
git clone https://huggingface.co/wanderkid/unimernet_tiny  # 441MB  

# モデルをModelScopeからもダウンロードできます
git clone https://www.modelscope.cn/wanderkid/unimernet_base.git
git clone https://www.modelscope.cn/wanderkid/unimernet_small.git
git clone https://www.modelscope.cn/wanderkid/unimernet_tiny.git
```

### インストール

> クリーンなConda環境を作成

``` bash 
conda create -n unimernet python=3.10

conda activate unimernet
```

> インストール方法1：pip installで直接インストール（一般ユーザー向け）
```bash
pip install --upgrade unimernet

pip install "unimernet[full]"
```

> インストール方法2：ローカルインストール（開発者向け）
```bash
pip install -e ."[full]"
```


### UniMERNetの実行

1. **Streamlitアプリケーション**：インタラクティブでユーザーフレンドリーな体験のために、StreamlitベースのGUIを使用します。このアプリケーションでは、リアルタイムの数式認識とレンダリングが可能です。

    ```bash
    unimernet_gui
    ```
    最新バージョンのUniMERNetをインストールしていることを確認してください（`pip install --upgrade unimernet & pip install "unimernet[full]"`）Streamlit GUIアプリケーションを使用するために。

2. **コマンドラインデモ**：画像からLaTeXコードを予測します。

    ```bash
    python demo.py
    ```

3. **Jupyter Notebookデモ**：画像から数式を認識してレンダリングします。

    ```bash
    jupyter-lab ./demo.ipynb
    ```

## SOTAメソッドとのパフォーマンス比較（BLEU）。

> UniMERNetは、実世界の数式認識において主流のモデルを大幅に上回り、BLEUスコア評価によって示されるように、シンプルな印刷表現（SPE）、複雑な印刷表現（CPE）、スクリーンキャプチャ表現（SCE）、手書き表現（HWE）において優れた性能を示しています。

![BLEU](./asset/papers/fig1_bleu.jpg)

## SOTAメソッドとのパフォーマンス比較（CDM）。

> 数式の表現には多様性があるため、異なるモデルの比較にBLEU指標を使用することは公平ではありません。そのため、数式認識専用に設計されたCDMで評価を行いました。我々の方法はオープンソースモデルを大幅に上回り、商用ソフトウェアMathpixと同等の効果を示しました。CDM@ExpRateは完全に予測が正しい数式の割合を指し、詳細は[CDM](https://arxiv.org/pdf/2409.03643)論文を参照してください。

![CDM](./asset/papers/fig2_cdm.jpg)

## 異なるメソッドによる可視化結果。

> UniMERNetは、他のメソッドを上回る挑戦的なサンプルの視覚的認識において優れています。

![Visualization](https://github.com/opendatalab/VIGC/assets/69186975/6edcac69-5082-43a2-8095-5681b7a707b9)

## UniMERデータセット
### イントロダクション
UniMERデータセットは、数式認識（MER）分野を進展させるために特別に収集されたコレクションです。これには、100万以上のインスタンスを含む包括的なUniMER-1Mトレーニングセットと、実世界のシナリオに対するMERモデルのベンチマークテスト用に精巧に設計されたUniMERテストセットが含まれます。データセットの詳細は以下の通りです：

**UniMER-1Mトレーニングセット：**
  - 総サンプル数：1,061,791のLaTeX-画像ペア
  - 構成：簡潔で複雑な拡張数式表現のバランスの取れた混合
  - 目的：堅牢で高精度なMERモデルをトレーニングし、認識精度と一般化能力を向上させる

**UniMERテストセット：**
  - 総サンプル数：23,757、4種類の表現に分類：
    - シンプルな印刷表現（SPE）：6,762サンプル
    - 複雑な印刷表現（CPE）：5,921サンプル
    - スクリーンキャプチャ表現（SCE）：4,742サンプル
    - 手書き表現（HWE）：6,332サンプル
  - 目的：さまざまな実世界の条件下でMERモデルを徹底的に評価する

### データセットのダウンロード
データセットは[OpenDataLab](https://opendatalab.com/OpenDataLab/UniMER-Dataset)（中国ユーザー向け推奨）または[HuggingFace](https://huggingface.co/datasets/wanderkid/UniMER_Dataset)からダウンロードできます。

### UniMER-Testデータセットのダウンロード

UniMER-1Mデータセットをダウンロードし、以下のディレクトリに解凍します：
```bash
./data/UniMER-1M
```

UniMER-Testデータセットをダウンロードし、以下のディレクトリに解凍します：
```bash
./data/UniMER-Test
```

## トレーニング

UniMERNetモデルをトレーニングするには、以下の手順に従ってください：

1. **トレーニングデータセットパスの指定**：`configs/train`フォルダを開き、トレーニングデータセットのパスを設定します。

2. **トレーニングスクリプトの実行**：以下のコマンドを実行してトレーニングプロセスを開始します。

    ```bash
    bash script/train.sh
    ```

### 注意：
- `configs/train`フォルダに指定されたデータセットパスが正しくアクセス可能であることを確認してください。
- トレーニングプロセス中のエラーや問題を監視してください。

## テスト

UniMERNetモデルをテストするには、以下の手順に従ってください：

1. **テストデータセットパスの指定**：`configs/val`フォルダを開き、テストデータセットのパスを設定します。

2. **テストスクリプトの実行**：以下のコマンドを実行してテストプロセスを開始します。

    ```bash
    bash script/test.sh
    ```

### 注意：
- `configs/val`フォルダに指定されたデータセットパスが正しくアクセス可能であることを確認してください。
- `test.py`スクリプトは指定されたテストデータセットを使用して評価を行います。test.pyのテストセットパスを実際のパスに変更することを忘れないでください。
- テスト結果を確認して、パフォーマンス指標や潜在的な問題を確認してください。

## 数式検出チュートリアル

数式認識の前提条件は、PDFやウェブページのスクリーンショット内の数式が存在する領域を検出することです。[PDF-Extract-Kit](https://github.com/opendatalab/PDF-Extract-Kit)には、数式を検出するための強力なモデルが含まれています。数式の検出と認識の両方を自分で行いたい場合は、数式検出モデルのデプロイと使用に関するガイドラインについて[数式検出チュートリアル](./MFD/README.md)を参照してください。

## TODO
 [✅] UniMERNetの推論コードとモデルをリリース。  
 [✅] UniMER-1MとUniMER-Testをリリース。  
 [✅] Streamlit数式認識GUIアプリケーションをオープンソース化。  
 [✅] UniMERNetのトレーニングコードをリリース。  


## 引用
私たちのモデル/コード/論文が研究に役立つ場合は、スターを付けていただき、私たちの仕事を引用してください。ありがとうございます。
```bibtex
@misc{wang2024unimernetuniversalnetworkrealworld,
      title={UniMERNet: A Universal Network for Real-World Mathematical Expression Recognition}, 
      author={Bin Wang and Zhuangcheng Gu and Guang Liang and Chao Xu and Bo Zhang and Botian Shi and Conghui He},
      year={2024},
      eprint={2404.15254},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2404.15254}, 
}

@misc{wang2024cdmreliablemetricfair,
      title={CDM: A Reliable Metric for Fair and Accurate Formula Recognition Evaluation}, 
      author={Bin Wang and Fan Wu and Linke Ouyang and Zhuangcheng Gu and Rui Zhang and Renqiu Xia and Bo Zhang and Conghui He},
      year={2024},
      eprint={2409.03643},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.03643}, 
}
```

## 謝辞
- [VIGC](https://github.com/opendatalab/VIGC)。モデルフレームワークはVIGCに依存しています。
- [Texify](https://github.com/VikParuchuri/texify)。主流のMERアルゴリズムであり、UniMERNetのデータ処理はTexifyを参考にしています。
- [Latex-OCR](https://github.com/lukas-blecher/LaTeX-OCR)。もう一つの主流のMERアルゴリズムです。
- [Donut](https://huggingface.co/naver-clova-ix/donut-base)。UniMERNetのTransformerエンコーダー-デコーダーはDonutを参考にしています。
- [Nougat](https://github.com/facebookresearch/nougat)。トークナイザーはNougatを使用しています。

## お問い合わせ
質問、コメント、提案がある場合は、wangbin@pjlab.org.cnまでお気軽にお問い合わせください。

## ライセンス
[Apache License 2.0](LICENSE)
