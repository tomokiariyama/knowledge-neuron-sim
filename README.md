# 似た単語の知識ニューロンは似た形成過程を経る
- このリポジトリは、以下の論文の再現用リポジトリです：https://www.anlp.jp/proceedings/annual_meeting/2025/pdf_dir/A8-1.pdf
- このリポジトリに含まれる一部のコードは、次のリポジトリのコードから改変して作られています：https://github.com/EleutherAI/knowledge-neurons

## 論文掲載図の再現方法
このリポジトリは`uv`を使用して実行することを想定しています。
`uv`のインストール方法は以下のリンクを参照してください：https://docs.astral.sh/uv/getting-started/installation/

- 初めにこのリポジトリを `git clone` し、そのディレクトリに移動してください。
  - このリポジトリのコードは `python 3.10` で動作確認しています。
- 次に、以下のコマンドを実行してください：
  - `chmod +x scripts/setup.sh scripts/evaluate_pythia_410m.sh scripts/wasserstein_distance.sh scripts/group_neurons.sh experiment.sh`
- 最後に `./experiment.sh` コマンドを実行することで、下記に示す図を再現することができます。
  - `knowledge-neuron-sim/work/figures/*` 配下：
    - 図 3: "wasserstein_distance/wasserstein_distance_of_top_1000_neurons_log_scale.png"
    - 図 4: "wasserstein_distance/binary_wasserstein_distance_of_top_1000_neurons.png"
    - 図 5: "group_neurons/top_1000_covering_neurons.png"
  - 注）`./experiment.sh` の実行には長時間（〜十数時間）を要します。


## スクリプト
```yaml
setup.sh: uvをセットアップし、必要なライブラリをインストールします。
evaluate_pythia_410m.sh: pythia-410m内のニューロンについて、帰属値測定を行います。
  - evaluate_attribution_scores.py: 帰属値測定スクリプト
wasserstein_distance.sh: 各単語について、知識ニューロン形成過程間のワッサースタイン距離を測定し、グラフ化します。
  - wasserstein_distance.py: ワッサースタイン距離測定＆グラフ化スクリプト
group_neurons.sh: 各単語について形成される知識ニューロンの積集合を計算し、グラフ化します。
  - group_neurons.py: 知識ニューロン積集合計算＆グラフ化スクリプト
```
