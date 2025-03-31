import os
import argparse
from collections import defaultdict
import numpy as np
import pandas
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import japanize_matplotlib
import copy
from tqdm import tqdm
import sys
import json
import nltk
from scipy.stats import entropy
from PIL import Image
from typing import List


def extract_data_of_kn(results_path, training_steps, pivot_step):
    # extract pivot step's knowledge neurons
    pivot_data_path = os.path.join(
        results_path,
        f"multiberts-seed_0-step_{pivot_step}k",
        f"step{pivot_step}",
        "nt_4_at_0.2_mw_10",
        "suppress_activation_and_relevant_prompts.jsonl"
    )

    with open(pivot_data_path, 'r') as f:
        df = pd.read_json(f, orient='records', lines=True)

    pivot_df = df.drop_duplicates(subset="positive_concept")
    pivot_df = pivot_df[["positive_concept", "refined_neurons"]]
    pivot_data = pivot_df.to_dict(orient="records")  # {'positive_concept': concept, 'refined_neurons': [kn_indices]}

    pivot_dict = dict()
    for dic in pivot_data:
        sorted_data = sorted(dic['refined_neurons'], key=lambda x: (x[0], x[1]), reverse=True)  # kn のインデックスを並べ直す
        pivot_dict[dic["positive_concept"]] = sorted_data  # {concept: [kn_indices]}


    # extract training steps' knowledge neurons
    graph_df = pd.DataFrame(columns=["training_step", "kn_index", "num_overlap", "num_not_overlap"])

    for step in tqdm(training_steps):
        data_path = os.path.join(
            results_path,
            f"multiberts-seed_0-step_{step}k",
            f"step{step}",
            "nt_4_at_0.2_mw_10",
            "suppress_activation_and_relevant_prompts.jsonl"
        )

        with open(data_path, 'r') as f:
            df = pd.read_json(f, orient='records', lines=True)

        step_df = df.drop_duplicates(subset="positive_concept")
        step_df = step_df[["positive_concept", "refined_neurons"]]
        step_data = step_df.to_dict(orient="records")

        step_dict = dict()
        for dic in step_data:
            step_dict[dic["positive_concept"]] = dic["refined_neurons"]  # {concept: [kn_indices]}

        kn_overlaps_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        for concept, kns in pivot_dict.items():
            for kn in kns:
                if kn in step_dict[concept]:
                    kn_overlaps_dict[concept][str(kn)]["overlap"] += 1
                else:
                    kn_overlaps_dict[concept][str(kn)]["not_overlap"] += 1

        # dataframe に変換
        # c = 0
        for concept, overlap_dicts in tqdm(kn_overlaps_dict.items(), leave=False):
            for kn, overlap_dict in overlap_dicts.items():
                graph_df = graph_df.append(
                    {
                        "training_step": step,
                        "concept": concept,
                        "kn_index": kn,
                        "num_overlap": overlap_dict["overlap"],
                        "num_not_overlap": overlap_dict["not_overlap"]
                    },
                    ignore_index=True
                )
            # c += 1
            # if c == 3:
            #     break

    return graph_df, pivot_dict.keys()


# 全ての知識ニューロンをプロットするために使用するデータの抽出
def extract_data_of_all_checkpoints(results_path, training_steps: List[int], model_name: str):
    # # extract pivot step's knowledge neurons
    # pivot_data_path = os.path.join(
    #     results_path,
    #     f"multiberts-seed_0-step_{pivot_step}k",
    #     f"step{pivot_step}",
    #     "nt_4_at_0.2_mw_10",
    #     "suppress_activation_and_relevant_prompts.jsonl"
    # )
    #
    # with open(pivot_data_path, 'r') as f:
    #     df = pd.read_json(f, orient='records', lines=True)
    #
    # pivot_df = df.drop_duplicates(subset="positive_concept")
    # pivot_df = pivot_df[["positive_concept", "refined_neurons"]]
    # pivot_data = pivot_df.to_dict(orient="records")  # {'positive_concept': concept, 'refined_neurons': [kn_indices]}
    #
    # pivot_dict = dict()
    # for dic in pivot_data:
    #     sorted_data = sorted(dic['refined_neurons'], key=lambda x: (x[0], x[1]), reverse=True)  # kn のインデックスを並べ直す
    #     pivot_dict[dic["positive_concept"]] = sorted_data  # {concept: [kn_indices]}

    # dfの列を構成するリスト
    concepts = []
    neuron_indices = []
    avg_attrs = []
    checkpoints = []
    POSes = []

    for step in training_steps:
        if model_name == "mBERT":
            data_path = os.path.join(
                results_path,
                f"multiberts-seed_0-step_{step}k",
                f"step{step}",
                "nt_4_at_0.2_mw_10",
                "attribution_scores.jsonl"
            )
        elif model_name == "pythia":
            # data_path = os.path.join(
            #     results_path,
            #     f"step{step}",
            #     "nt_4_at_0.2_mw_10",
            #     "attribution_scores.jsonl"
            # )

            # 全てのニューロンの帰属値を使用する場合
            # data_path = os.path.join(
            #     results_path,
            #     f"step{step}",
            #     "nt_4_at_-10.0_mw_10",
            #     "attribution_scores.jsonl"
            # )
            data_path = os.path.join(
                results_path,
                f"step{step}",
                "nt_4_at_-10.0_mw_30",
                "attribution_scores.jsonl"
            )
        else:
            raise ValueError(f"Invalid model name: {model_name}")

        data_list = []
        with open(data_path, 'r') as f:
            for line in f:
                data_list.append(json.loads(line))

        # DataFrame の各列に対応するデータを抽出
        for data_dict in tqdm(data_list, total=len(data_list), desc=f"Extracting data at the checkpoint {step}k"):
            for concept, data in data_dict.items():
                pos = nltk.pos_tag([concept])
                for neuron_index, attr_info in data.items():
                    concepts.append(concept)
                    neuron_indices.append(neuron_index)
                    avg_attrs.append(sum(ig for ig in attr_info["IGs"]) / len(attr_info["IGs"]) * pow(10, 6))  # 各正例文における IG の平均値 * 10^6（値が非常に小さいため、一旦オーダーを大きくしている）
                    checkpoints.append(step)
                    POSes.append(pos[0][1])

    # dataframe に変換
    tmp_dict = {
        "concept": concepts,
        "neuron_index": neuron_indices,
        "avg_attr": avg_attrs,
        "checkpoint": checkpoints,
        "POS": POSes
    }
    graph_df = pd.DataFrame(tmp_dict)

    return graph_df, concepts


def plot_concept_figure(save_path, pivot_step, data, other_steps, concept="ALL"):
    fig, ax = plt.subplots(1, 1, figsize=(9, 15))
    fontsize = 12

    # sns.set_theme(style="whitegrid")
    # sns.set(font="IPAexGothic")
    # japanize_matplotlib.japanize()
    # cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)

    # 縦の罫線を引く
    ax.axvline(x=pivot_step, color="red", linestyle="-", linewidth=2.0)
    for training_step in other_steps:
        ax.axvline(x=training_step, color="gray", linestyle="-", linewidth=0.5)

    sns.scatterplot(
        data=data,
        x="training_step",
        y="kn_index",
        size="num_overlap",
        # palette=cmap,
        # sizes=(40, 400),
        sizes={0:0, 1:40},
        ax=ax,
    )

    ax.set_title(f"チェックポイント{pivot_step}kで見つかる'{concept}'ニューロンの安定性", fontsize=fontsize)
    ax.legend(loc=1, fontsize=fontsize)  # 凡例の位置調整
    # ax.set_xlabel("チェックポイントの学習ステップ数[×10^3]", fontsize=fontsize)
    # ax.set_ylabel("知識ニューロンのインデックス", fontsize=fontsize)

    figure = fig.get_figure()

    os.makedirs(os.path.join(save_path, f"pivot-{pivot_step}k", f"{concept[0]}"), exist_ok=True)
    save_path = os.path.join(save_path, f"pivot-{pivot_step}k", f"{concept[0]}", f"concept-{concept}.png")
    figure.savefig(save_path)
    print(f"figure is saved in {save_path}")
    plt.close()


def plot_all_concepts_figure(save_path, pivot_step, data, others_steps):
    fig, ax = plt.subplots(1, 1, figsize=(9, 15))
    fontsize = 12

    # sns.set_theme(style="whitegrid")
    # sns.set(font="IPAexGothic")
    # japanize_matplotlib.japanize()
    # cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)

    # 密度のプロットにしても良さそう
    sns.scatterplot(
        data=data,
        x="training_step",
        y="kn_index",
        size="num_overlap",
        # palette=cmap,
        # sizes=(40, 400),
        sizes={0:0, 1:40},
        ax=ax,
    )

    ax.set_title(f"チェックポイント{pivot_step}kで見つかる全ての知識ニューロンの安定性", fontsize=fontsize)
    ax.legend(loc=1, fontsize=fontsize)  # 凡例の位置調整
    # ax.set_xlabel("チェックポイントの学習ステップ数[×10^3]", fontsize=fontsize)
    # ax.set_ylabel("知識ニューロンのインデックス", fontsize=fontsize)
    # 縦の罫線を引く
    ax.axvline(x=pivot_step, color="red", linestyle="-", linewidth=2.0)
    for training_step in others_steps:
        ax.axvline(x=training_step, color="gray", linestyle="-", linewidth=0.5)

    figure = fig.get_figure()

    os.makedirs(os.path.join(save_path, f"pivot-{pivot_step}k"), exist_ok=True)
    save_path = os.path.join(save_path, f"pivot-{pivot_step}k", f"all_concept.png")
    figure.savefig(save_path)
    print(f"figure is saved in {save_path}")
    plt.close()


def sort_kn_indices(data_df: pandas.DataFrame, training_steps: List[int], sort_type: str):
    if sort_type == "IG_change_entropy":  # ニューロンインデックスを、IGのエントロピー順に並べ替える
        data_dict = defaultdict(list)
        # max_avg_IG = 0.0
        for step in training_steps:
            kn_indices = data_df['neuron_index'].unique().tolist()  # 重複を除いたニューロンインデックス
            step_df = data_df[data_df['checkpoint'] == step]  # あるチェックポイントのデータ

            for row in step_df.itertuples():
                data_dict[row.neuron_index].append(row.avg_attr)  # {"ニューロンインデックス"：[IGの合計値]}
                kn_indices.remove(row.neuron_index)  # 重複を除いたニューロンインデックスから削除
                # if row.avg_attr > max_avg_IG:
                #     max_avg_IG = row.avg_attr

            for kn_index in kn_indices:
                data_dict[kn_index].append(0)  # IGの値がない場合は0を追加

        # エントロピーを計算
        IG_change_entropy = []
        for row in data_df.itertuples():
            # IG_change_entropy.append(entropy(np.histogram(data_dict[row.neuron_index], range=(0, max_avg_IG))[0], base=2))
            IG_change_entropy.append(entropy(np.histogram(data_dict[row.neuron_index])[0], base=2))

        # エントロピーをデータフレームに追加
        data_df['IG_change_entropy'] = IG_change_entropy

        # エントロピーの大きい順にニューロンインデックスを並べ替え
        data_df = data_df.sort_values('IG_change_entropy', ascending=False)

        # 描画用にデータフレームを numpy 配列に変換
        neuron_index = "initialization"
        pixel_list = []

        IGs_list = []
        for row in data_df.itertuples():
            if neuron_index != row.neuron_index:
                if neuron_index != "initialization":
                    pixel_list.append(IGs_list)  # 全てのチェックポイントのIGをまとめ終わったニューロンのIGリストを追加
                IGs_list = [0.0] * len(training_steps)
                IGs_list[training_steps.index(row.checkpoint)] = row.avg_attr
                neuron_index = row.neuron_index
            else:
                IGs_list[training_steps.index(row.checkpoint)] = row.avg_attr
        pixel_list.append(IGs_list)  # 最後のニューロンのIGリストを追加

        data_np = np.array(pixel_list)
        data_np = ((data_np / np.max(data_np)) * 255).astype(np.uint8)

        return data_np

    elif sort_type == 'identified_as_kn_ascend' or 'identified_as_kn_descend':
        data_df = data_df.pivot(index="neuron_index", columns="checkpoint", values="avg_attr")
        data_df.fillna(0, inplace=True)

        data_np = data_df.to_numpy()
        data_np = ((data_np / np.max(data_np)) * 255).astype(np.uint8)

        # ニューロンが光ったステップを、三角行列に変換してソート
        def make_triangular(matrix):
            rows, cols = matrix.shape
            sorted_matrix = np.zeros_like(matrix)

            if sort_type == 'identified_as_kn_ascend':
                first_non_zero_positions = [np.argmax(row != 0) if np.any(row != 0) else cols for row in matrix]
                sorted_indices = np.argsort(first_non_zero_positions)
                for i, idx in enumerate(sorted_indices):
                    sorted_matrix[i] = matrix[idx]
            elif sort_type == 'identified_as_kn_descend':
                first_non_zero_positions = [cols - np.argmax(row[::-1] != 0) if np.any(row != 0) else 0 for row in matrix]
                sorted_indices = np.argsort(first_non_zero_positions)
                for i, idx in enumerate(sorted_indices, start=1):
                    sorted_matrix[sorted_matrix.shape[0] - i] = matrix[idx]
            else:
                raise ValueError(f"sort_type: {sort_type} is unsupported.")

            return sorted_matrix

        return make_triangular(data_np)

    else:
        raise ValueError(f"sort_type: {sort_type} is invalid.")


def plot_all_kn(save_path: str, data_df: pandas.DataFrame, training_steps: List[int], concept, sort_type: str):
    # 現在の概念に関するデータを取得
    data_df = data_df.get_group(concept)

    fig, ax = plt.subplots(1, 1, figsize=(90, 150))
    fontsize = 12

    # sns.set_theme(style="whitegrid")
    # sns.set(font="IPAexGothic")
    # japanize_matplotlib.japanize()
    # cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)

    # 縦の罫線を引く
    for training_step in training_steps:
        ax.axvline(x=training_step, color="gray", linestyle="-", linewidth=0.5)

    # ニューロンインデックスを指定されたソート順に並べ替えたnumpy配列を取得
    pixel_np = sort_kn_indices(data_df, training_steps, sort_type)
    assert np.max(pixel_np) <= 255, print(f"Warning: The maximum value of the data is over 255. The concept is {concept}.")
    # 1以上の値を全て0に、0を255に置換
    pixel_np[pixel_np >= 1] = 1
    pixel_np[pixel_np == 0] = 255
    pixel_np[pixel_np == 1] = 0

    im = Image.fromarray(pixel_np, mode="L")

    # os.makedirs(os.path.join(save_path, "raw_pixels", "plot_all_kn", f"{concept[0]}"), exist_ok=True)
    # raw_image_save_path = os.path.join(save_path, "raw_pixels", "plot_all_kn", f"{concept[0]}", f"concept-{concept}.png")
    os.makedirs(os.path.join(save_path, "raw_pixels", "monochrome", f"{concept[0]}"), exist_ok=True)
    raw_image_save_path = os.path.join(save_path, "raw_pixels", "monochrome", f"{concept[0]}",
                                       f"concept-{concept}.png")
    im.save(raw_image_save_path)
    print(f"figure is saved in {raw_image_save_path}")


def make_parser():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-rp", "--results_path",
        help="path for the result files.",
        required=True,
    )
    parser.add_argument(
        '-sp', '--save_path',
        help="figures will be saved under the designed path.",
        type=str,
        default='work/figure/tmp',
    )
    parser.add_argument(
        '-ps', '--pivot_step',
        help="the step of checkpoint to be used as pivot.",
        type=int,
        default='-1',
    )
    parser.add_argument(
        "-mn", "--model_name",
        help="Model name. Must be choiced from ['mBERT', 'pythia']",
        type=str,
        choices=['mBERT', 'pythia'],
    )
    parser.add_argument(
        "-st", "--sort_type",
        help="Sort type of neurons' indices. Must be choiced from ['IG_change_entropy', 'identified_as_kn_ascend', 'identified_as_kn_descend']",
        type=str,
        choices=['IG_change_entropy', 'identified_as_kn_ascend', 'identified_as_kn_descend'],
        default='IG_change_entropy',
    )

    return parser.parse_args()


def main():
    """
    USAGE:
    $ python kn_stability.py \
          -mn "mBERT" \
          -rp "work/result/10_negative_concepts_per_1_positive_concept/generics_kb_best/train" \
          -sp "work/figure/10_negative_concepts_per_1_positive_concept/generics_kb_best/train/kn_stability_with_plot_on_red_line" \
          -ps 2000 \
          -st "IG_change_entropy"
    $ python kn_stability.py \
          -mn "mBERT" \
          -rp "work/result/10_negative_concepts_per_1_positive_concept/kn_stability/generics_kb_best/train" \
          -sp "work/figure/10_negative_concepts_per_1_positive_concept/generics_kb_best/train/kn_stability/all_knowledge_neurons/resort_IG_entropy" \
          -st "IG_change_entropy"
    $ python kn_stability.py \
          -mn "pythia" \
          -rp "work/result/natural_questions/train/pythia-70m-deduped" \
          -sp "work/figure/natural_questions/train/pythia-70m-deduped/kn_stability" \
          -st "identified_as_kn_ascend"
    """
    args = make_parser()

    save_path = os.path.join(args.save_path, args.sort_type)
    os.makedirs(save_path, exist_ok=True)
    pivot_step = args.pivot_step
    model_name = args.model_name
    sort_type = args.sort_type

    if pivot_step != -1:
        if model_name == "mBERT":
            training_steps = [0, 20, 40, 60, 80, 100, 200, 300, 400, 500, 1000, 1500, 2000, ]
        elif model_name == "pythia":
            training_steps = [
                0, 512, 1000, 3000, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000,
                110000, 120000, 130000, 140000, 143000,
            ]
        else:
            raise NotImplementedError(f"model_name '{model_name}' is not supported.")

        # extract data
        kn_data_df, concepts = extract_data_of_kn(args.results_path, training_steps, pivot_step)

        # plot figure
        df_grouped_by_concept = kn_data_df.groupby("concept")

        other_steps = copy.deepcopy(training_steps)
        other_steps.remove(pivot_step)

        for concept in concepts:
            try:
                df = df_grouped_by_concept.get_group(concept)
                plot_concept_figure(args.save_path, pivot_step, df, other_steps, concept)
            except KeyError as e:
                print(f"{e}: concept '{concept}' はチェックポイント{pivot_step}kで見つかる知識ニューロンを持ちません。", file=sys.stderr)
                continue

        # 全ての概念の知識ニューロンをまとめてプロットする
        # plot_all_concepts_figure(args.save_path, pivot_step, kn_data_df, other_steps)
    else:
        if model_name == "mBERT":
            training_steps = [
                0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
                1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000,
            ]
        elif model_name == "pythia":
            training_steps = [
                0, 512, 1000, 3000, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000,
                110000, 120000, 130000, 140000, 143000,
            ]
        else:
            raise NotImplementedError(f"model_name '{model_name}' is not supported.")

        # extract data
        graph_df, concepts = extract_data_of_all_checkpoints(args.results_path, training_steps, model_name)

        # 全ての知識ニューロンをプロットする
        df_grouped_by_concept = graph_df.groupby("concept")

        # c = 0
        for concept in concepts:
            try:
                plot_all_kn(save_path, df_grouped_by_concept, training_steps, concept, sort_type)
            except KeyError as e:
                print(f"{e}: concept '{concept}' の知識ニューロンが見つかりません。", file=sys.stderr)
                continue
            # c += 1
            # if c > 0:
            #     break


if __name__ == '__main__':
    main()
