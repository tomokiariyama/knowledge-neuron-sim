import os
import random
import pickle
from collections import defaultdict
import itertools
from types import SimpleNamespace
import copy

import argparse
from tqdm import tqdm

import numpy as np
import torch

import pandas as pd
import polars as pl

import ot
from joblib import Parallel, delayed
from scipy.spatial.distance import squareform

import matplotlib as mpl
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import seaborn as sns

import utils.kn_stability as kn_stability


random.seed(42)
multiplier = 10


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
        '--intermediate_product_save_path',
        help="any intermediate products (e.g. pd.DataFrame) will be saved under this path.",
        type=str,
        default='work/tmp/intermediate_products/wasserstein_distance',
    )
    parser.add_argument(
        "-mn", "--model_name",
        help="Model name. Must be choice from ['mBERT', 'pythia']",
        type=str,
        choices=['mBERT', 'pythia'],
    )
    parser.add_argument(
        "--cbar_log_scale",
        help="Heatmap colorbar will be in log scale.",
        action="store_true",
    )
    parser.add_argument(
        "--binary_heatmap",
        help="Heatmap will be made with binary data.",
        action="store_true",
    )

    return parser.parse_args()


def set_experiment_variables(args):
    # 実験設定変数
    training_steps = [
        0, 512, 1000, 3000, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000,
        110000, 120000, 130000, 140000, 143000,
    ]

    # モデルパラメータ
    num_layer = 24
    d_mlp = 1024
    mlp_projection = 4  # 「mlpの内部で流れるベクトル次元 / mlpに入るベクトル次元」 のこと
    neurons_per_layer = d_mlp * mlp_projection
    neurons_per_model = neurons_per_layer * num_layer

    model_parameters = {
        "mlp_projection": mlp_projection,
        "num_layer": num_layer,
        "d_mlp": d_mlp,
        "neurons_per_layer": neurons_per_layer,
        "neurons_per_model": neurons_per_model,
    }

    if "similar_concepts" in args.results_path:
        selected_concepts = {
            "Animals": ["Cat", "Dog", "Fox", "Ant", "Bat", "Rat", "Hen", "Wolf", ],
            "Colors": ["Blue", "Red", "Green", "Black", "White", "Brown", "Gray", ],
            "Countries": ['Canada', 'Germany', 'Brazil', 'France', 'England', 'India', 'China', 'Mexico', 'Russia', 'Iran', 'Japan', 'Israel', 'Australia', 'Spain', ],
            "Languages": ['English', 'Spanish', 'French', 'Chinese', 'Japanese', 'German', 'Russian', 'Italian', 'Portuguese', ],
        }
    else:
        raise NotImplementedError(f"Wrong results path: '{args.results_path}'.")

    return model_parameters, training_steps, selected_concepts


def extract_list(concept_df, concept, k: int):
    concept_df["last_checkpoint_attr"] = np.array([row[-1] for row in concept_df["avg_attr"]])
    top_k_df = concept_df.nlargest(k, "last_checkpoint_attr")

    print(f"Extracted attr top {k} neurons for '{concept}'.")

    return SimpleNamespace(concept=concept, attr_list=top_k_df["avg_attr"].tolist())


def compute_emd2(list1, list2, gpu_index):
    assert len(list1) == len(list2), f"The length of 'list1' and 'list2' is not the same: 'list1' is '{len(list1)}' and 'list2' is '{len(list2)}'."
    n = len(list1)

    if not isinstance(list1, np.ndarray):
        list1 = np.array(list1)
    if not isinstance(list2, np.ndarray):
        list2 = np.array(list2)

    if torch.cuda.is_available():
        a, b = torch.ones((n,), device=torch.device(gpu_index)) / n, \
               torch.ones((n,), device=torch.device(gpu_index)) / n  # 一様重み
        M = ot.dist(torch.tensor(list1, device=torch.device(gpu_index)),
                    torch.tensor(list2, device=torch.device(gpu_index)))
    else:
        a, b = torch.ones((n,), device=torch.device('cpu')) / n, \
               torch.ones((n,), device=torch.device('cpu')) / n
        M = ot.dist(torch.tensor(list1, device=torch.device('cpu')),
                    torch.tensor(list2, device=torch.device('cpu')))

    return ot.emd2(a, b, M).cpu()


def extract_emd2(concept_pair, data_dict, gpu_index):
    list1 = data_dict[concept_pair[0]]
    list2 = data_dict[concept_pair[1]]
    emd2 = compute_emd2(list1, list2, gpu_index)
    return concept_pair, emd2


def parallel_emd2_computation(args, df, concept_pairs, model_parameters, training_steps, k: int):
    # 並列に計算する関数
    # 事前にgroupbyしておく（これにより、df内のニューロンインデックスの順序は崩れる）
    df_pl = pl.DataFrame(df)
    df_grouped_pl = df_pl.group_by(['concept', 'neuron_index']).agg(pl.col('avg_attr'))  # 2,052,096行
    df_grouped = df_grouped_pl.to_pandas()
    df_grouped = df_grouped.set_index(["concept", "neuron_index"])

    # 並列処理 with joblib
    concept_attr_dict = defaultdict(list)  # 既に計算済みの概念のattrリストを保存する辞書
    unique_concepts = list(set(concept for concept_pair in concept_pairs for concept in concept_pair))
    attr_results_list = Parallel(n_jobs=4) \
        (delayed(extract_list)(df_grouped.loc[f"{concept}"].copy(), concept, k)
         for concept in tqdm(unique_concepts, total=len(unique_concepts), desc=f"Extracting lists"))

    # 結果を辞書に格納
    for attr_result in attr_results_list:
        concept_attr_dict[attr_result.concept] = attr_result.attr_list

    # 検索して取り出したattrリストを保存
    os.makedirs(args.intermediate_product_save_path, exist_ok=True)
    pkl_name = f"attr_list_of_top_{k}_neurons.pkl"
    with open(os.path.join(args.intermediate_product_save_path, pkl_name), "wb") as f:
        pickle.dump(concept_attr_dict, f)
        print(f"Saved the attribution score list as {pkl_name} in '{args.intermediate_product_save_path}'.")


    # ワッサーシュタイン距離を並列に計算
    results_dict = defaultdict(float)
    num_gpus = torch.cuda.device_count()

    if num_gpus == 0:
        # 並列実行 with CPU
        emd2_results = Parallel(n_jobs=10) \
            (delayed(extract_emd2)(concept_pair, concept_attr_dict, -1)
             for concept_pair in tqdm(concept_pairs, total=len(concept_pairs), desc="Calculating emd2 on CPU"))
    else:
        # 並列実行 with GPU
        emd2_results = Parallel(n_jobs=10) \
            (delayed(extract_emd2)(concept_pair, concept_attr_dict, i % num_gpus)
             for i, concept_pair in tqdm(enumerate(concept_pairs), total=len(concept_pairs), desc="Calculating emd2 on GPU"))

    # 計算結果を収集
    for emd2_result in emd2_results:
        concept_pair, wasserstein_distance = emd2_result[0], emd2_result[1]
        results_dict[concept_pair] = wasserstein_distance
    # 計算したワッサーシュタイン距離を保存
    with open(os.path.join(args.intermediate_product_save_path, f"wasserstein_distance_of_top_{k}_neurons.pkl"), "wb") as f:
        pickle.dump(results_dict, f)
        print(
            f"Saved the Wasserstein distance as 'wasserstein_distance_of_top_{k}_neurons.pkl' in '{args.intermediate_product_save_path}'.")

    return results_dict, concept_attr_dict


def make_binary_df(wasserstein_df, selected_concepts):
    selected_concepts_flattened = list(itertools.chain.from_iterable(selected_concepts.values()))

    # バイナリデータを格納するdfを作成
    result_df = pd.DataFrame(0, index=selected_concepts_flattened, columns=selected_concepts_flattened)

    # selected_conceptsのキー・バリューを入れ替えて、各インデックスがどのグループかをわかるようにする
    group_dict = {concept: group for group, concepts in selected_concepts.items() for concept in concepts}

    # バイナリグラフの評価指標
    eval_idx = defaultdict(lambda: defaultdict(float))
    for row in selected_concepts_flattened:
        row_group = group_dict[row]  # 現在の行インデックスが属するグループ
        group_elements = copy.deepcopy(selected_concepts[row_group])  # 現在の行インデックスと同じグループの要素
        group_elements.remove(row)  # 自分自身をグループから除く

        # 同じグループ内の列の値を使って平均を計算
        group_avg = wasserstein_df.loc[row, group_elements].mean()

        for col in selected_concepts_flattened:
            if col in group_elements:
                result_df.loc[row, col] = 0.2
            else:
                if col == row:
                    result_df.loc[row, col] = 0
                else:
                    result_df.loc[row, col] = 1 if (wasserstein_df.loc[row, col] > group_avg) else 0.55

                eval_idx[row_group]["all_count"] += 1
                eval_idx[row_group]["black_count"] += int(wasserstein_df.loc[row, col] > group_avg)

    for group in selected_concepts.keys():
        eval_idx[group]["black_ratio"] = eval_idx[group]["black_count"] / eval_idx[group]["all_count"] * 100

    return result_df, eval_idx


def show_heatmap(args, emd2_results_dict, concept_attr_dict, selected_concepts, concept_pairs, k: int):
    os.makedirs(args.intermediate_product_save_path, exist_ok=True)
    df_path = os.path.join(args.intermediate_product_save_path, f"wasserstein_distance_of_top_{k}_neurons.csv")
    if os.path.isfile(df_path):
        wasserstein_distance_df = pd.read_csv(df_path, index_col=0)
    else:
        # 行名・列名が使用した概念名で、値が wasserstein 距離の dataframe を作成する
        wasserstein_distance_list = []
        for concept_pair in concept_pairs:
            wasserstein_distance_list.append(emd2_results_dict[concept_pair])

        wasserstein_distance_list = squareform(wasserstein_distance_list)  # 正方行列化
        selected_concepts_flattened = list(itertools.chain.from_iterable(selected_concepts.values()))  # 2次元リストを1次元リストに変換
        wasserstein_distance_df = pd.DataFrame(wasserstein_distance_list,
                                               index=selected_concepts_flattened, columns=selected_concepts_flattened)

        # dataframe の保存
        wasserstein_distance_df.to_csv(df_path)
        print(f"Saved the plot data of Wasserstein distance as 'wasserstein_distance_of_top_{k}_neurons.csv' in '{args.intermediate_product_save_path}'.")

    # フォントサイズ
    font_size = 25
    mpl.rcParams['font.size'] = font_size  # デフォルトのフォントサイズを変更

    os.makedirs(args.save_path, exist_ok=True)

    if args.cbar_log_scale:
        fig, ax = plt.subplots(1, 1, figsize=(22, 18))

        # 以下の対数スケールを使うには、dfの0を0.01などに置き換えてmiとし、dfの最大値をmaとする
        mi = 0.01
        ma = wasserstein_distance_df.max().max()
        wasserstein_distance_df = wasserstein_distance_df.replace(0.0, mi)
        graph = sns.heatmap(wasserstein_distance_df, cmap="YlGnBu", cbar=True, norm=LogNorm(vmin=mi, vmax=ma))
        # 目盛りのフォントサイズ
        ax.tick_params(axis='x', labelsize=font_size)
        ax.tick_params(axis='y', labelsize=font_size)

        figure_save_path = os.path.join(args.save_path, f"wasserstein_distance_of_top_{k}_neurons_log_scale.png")
    elif args.binary_heatmap:
        fig, ax = plt.subplots(1, 1, figsize=(18, 18))

        binary_df, eval_idx = make_binary_df(wasserstein_distance_df, selected_concepts)
        graph = sns.heatmap(binary_df, cmap="YlGnBu", cbar=False, annot=False, ax=ax)
        # 目盛りのフォントサイズ
        ax.tick_params(axis='x', labelsize=font_size)
        ax.tick_params(axis='y', labelsize=font_size)

        # 下部の余白を調整
        fig.subplots_adjust(bottom=0.3)
        txt = f"k={k}, Evaluation Index: {eval_idx}"
        # fig.text(0.5, 0.1, txt, wrap=True, horizontalalignment='center', fontsize=font_size)
        print(txt)

        figure_save_path = os.path.join(args.save_path, f"binary_wasserstein_distance_of_top_{k}_neurons.png")
    else:
        fig, ax = plt.subplots(1, 1, figsize=(22, 18))
        graph = sns.heatmap(wasserstein_distance_df, cmap="YlGnBu", cbar=True)

        # 目盛りのフォントサイズ
        ax.tick_params(axis='x', labelsize=font_size)
        ax.tick_params(axis='y', labelsize=font_size)

        figure_save_path = os.path.join(args.save_path, f"wasserstein_distance_of_top_{k}_neurons.png")

    plt.tight_layout()
    plt.savefig(figure_save_path)
    print(f"figure saved at '{figure_save_path}'.")
    plt.close()


def main():
    args = make_parser()
    model_parameters, training_steps, selected_concepts = set_experiment_variables(args)
    k = 1000

    selected_concepts_flattened = list(itertools.chain.from_iterable(selected_concepts.values()))

    # 概念ペア
    concept_pairs = []
    for concept in selected_concepts_flattened:
        for another_concept in selected_concepts_flattened[selected_concepts_flattened.index(concept) + 1:]:
            concept_pairs.append((concept, another_concept))

    extracting_data_flag = True
    # すでにワッサーシュタイン距離が計算されている場合は、その結果を読み込む
    file_path = os.path.join(args.intermediate_product_save_path, f"wasserstein_distance_of_top_{k}_neurons.csv")
    if os.path.isfile(file_path):
        print(f"Read the calculated data of wasserstein distance of top {k} neurons from '{file_path}'.")
        show_heatmap(args, dict(), dict(), selected_concepts, concept_pairs, k)
    else:
        if extracting_data_flag:
            # データの抽出
            all_data_df, concepts = kn_stability.extract_data_of_all_checkpoints(args.results_path, training_steps,
                                                                                 args.model_name)
            extracting_data_flag = False
        else:
            pass

        # 並列計算の実行
        emd2_results_dict, concept_attr_dict = parallel_emd2_computation(args, all_data_df, concept_pairs, model_parameters, training_steps, k)

        # 図の作成
        show_heatmap(args, emd2_results_dict, concept_attr_dict, selected_concepts, concept_pairs, k)


if __name__ == '__main__':
    main()
