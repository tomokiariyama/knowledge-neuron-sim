import os
import pickle
from collections import defaultdict
import itertools

import argparse
from tqdm import tqdm

import numpy as np

import pandas as pd
import polars as pl

from joblib import Parallel, delayed
from scipy.spatial.distance import squareform

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import utils.kn_stability as kn_stability
from wasserstein_distance import set_experiment_variables


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
        help="Model name. Must be choiced from ['mBERT', 'pythia']",
        type=str,
        choices=['mBERT', 'pythia'],
    )

    return parser.parse_args()


def extract_top_k(df, concept, k: int):
    df["avg_attr"] = np.array([x[0] for x in df["avg_attr"]])  # NumPy を使用してリストの中身を取り出す
    top_k_neuron_index_list = df.nlargest(k, 'avg_attr')["neuron_index"].values.tolist()
    print(f"Extracted top-{k} neurons for '{concept}'.")

    return concept, top_k_neuron_index_list


def count_covering_neurons(concept_pair, concept_top_k_neurons_dict):
    concept1, concept2 = concept_pair
    top_k_neurons1 = concept_top_k_neurons_dict[concept1]
    top_k_neurons2 = concept_top_k_neurons_dict[concept2]

    num_of_covering_neurons = len(set(top_k_neurons1) & set(top_k_neurons2))

    return concept_pair, num_of_covering_neurons


def extract_top_k_neurons(args, df, concept_pairs, k: int):
    # 事前にgroupbyしておく（これにより、df内のニューロンインデックスの順序は崩れる）
    df_pl = pl.DataFrame(df)
    df_grouped_pl = df_pl.group_by(['concept', 'neuron_index']).agg(pl.col('avg_attr'))
    df_grouped = df_grouped_pl.to_pandas().set_index(["concept"])

    # 概念ごとの top-k neurons リストを並列に検索し抽出
    file_path = os.path.join(args.intermediate_product_save_path, f"top_{k}_neuron_index_list.pkl")
    if os.path.isfile(file_path):
        with open(file_path, "rb") as f:
            concept_top_k_neurons_dict = pickle.load(f)
            print(f"Loaded the attribution score list from '{file_path}'.")
    else:
        concept_top_k_neurons_dict = defaultdict(list)  # 既に計算済みの概念の top-k neurons リストを保存する辞書

        # 並列処理 with joblib
        unique_concepts = list(set(concept for concept_pair in concept_pairs for concept in concept_pair))
        tmp_top_k_neurons_results = Parallel(n_jobs=8)\
            (delayed(extract_top_k)(df_grouped.loc[concept].copy(), concept, k)
             for concept in tqdm(unique_concepts, total=len(unique_concepts), desc=f"Extracting top-{k} neurons"))
        # 結果を辞書に格納
        for concept, top_k_neuron_index_list in tmp_top_k_neurons_results:
            concept_top_k_neurons_dict[concept] = top_k_neuron_index_list

        # 検索して取り出したattrリストを保存
        os.makedirs(args.intermediate_product_save_path, exist_ok=True)
        with open(os.path.join(args.intermediate_product_save_path, f"top_{k}_neuron_index_list.pkl"), "wb") as f:
            pickle.dump(concept_top_k_neurons_dict, f)
            print(f"Saved the attribution score list as 'top_{k}_neuron_index_list.pkl' in '{args.intermediate_product_save_path}'.")


    # 被っている上位ニューロンを調べる：並列実行
    covering_neurons_results = Parallel(n_jobs=8)\
        (delayed(count_covering_neurons)(concept_pair, concept_top_k_neurons_dict)
         for concept_pair in tqdm(concept_pairs, total=len(concept_pairs), desc="Counting covering neurons"))

    # 計算結果を収集
    results_dict = defaultdict(int)
    for covering_neurons_result in covering_neurons_results:
        concept_pair, num_of_covering_neurons = covering_neurons_result[0], covering_neurons_result[1]
        results_dict[concept_pair] = num_of_covering_neurons

    # 求めた被っている上位ニューロンを保存
    with open(os.path.join(args.intermediate_product_save_path, f"top_{k}_covering_neurons.pkl"), "wb") as f:
        pickle.dump(results_dict, f)
        print(f"Saved the number of covering neurons as 'top_{k}_covering_neurons.pkl' in '{args.intermediate_product_save_path}'.")

    return results_dict


def show_heatmap(args, top_k_neurons_dict, selected_concepts, concept_pairs, k: int):
    os.makedirs(args.intermediate_product_save_path, exist_ok=True)
    df_path = os.path.join(args.intermediate_product_save_path, f"top_{k}_covering_neurons.csv")
    if os.path.isfile(df_path):
        covering_neurons_df = pd.read_csv(df_path, index_col=0)
    else:
        # 行名・列名が使用した概念名で、値が covering_neurons の数である dataframe を作成する
        covering_neurons_list = []
        for concept_pair in concept_pairs:
            covering_neurons_list.append(top_k_neurons_dict[concept_pair])

        covering_neurons_list = squareform(covering_neurons_list)  # 正方行列化
        np.fill_diagonal(covering_neurons_list, k)  # 対角成分を k にする
        covering_neurons_df = pd.DataFrame(covering_neurons_list,
                                           index=selected_concepts, columns=selected_concepts)

        # dataframe の保存
        covering_neurons_df.to_csv(df_path)
        print(f"Saved the plot data of covering neurons as 'top_{k}_covering_neurons.csv' in '{args.intermediate_product_save_path}'.")

    # japanize_matplotlib.japanize()
    # フォントサイズ
    font_size = 25
    mpl.rcParams['font.size'] = font_size  # デフォルトのフォントサイズを変更

    # plt.figure(figsize=(24, 18))
    fig, ax = plt.subplots(1, 1, figsize=(22, 18))

    os.makedirs(args.save_path, exist_ok=True)
    graph = sns.heatmap(covering_neurons_df, cmap="YlOrBr", cbar=True, annot=False, fmt="d")
    figure_save_path = os.path.join(args.save_path, f"top_{k}_covering_neurons.png")

    # 目盛りのフォントサイズ
    ax.tick_params(axis='x', labelsize=font_size)
    ax.tick_params(axis='y', labelsize=font_size)
    plt.tight_layout()
    plt.savefig(figure_save_path)
    print(f"figure saved at '{figure_save_path}'.")
    plt.close()


def main():
    args = make_parser()
    model_parameters, _, selected_concepts_unflattened = set_experiment_variables(args)
    training_steps = [143000]
    k = 1000

    selected_concepts = list(itertools.chain.from_iterable(selected_concepts_unflattened.values()))

    # 概念ペア
    concept_pairs = []
    for concept in selected_concepts:
        for another_concept in selected_concepts[selected_concepts.index(concept) + 1:]:
            concept_pairs.append((concept, another_concept))

    # すでに top-k ニューロンが求められている場合は、その結果を読み込む
    file_path = os.path.join(args.intermediate_product_save_path, f"group_neurons_top-{k}.csv")
    if os.path.isfile(file_path):
        print(f"Read the calculated data of top-k neurons from '{file_path}'.")
        show_heatmap(args, dict(), selected_concepts, concept_pairs, k)

    # データの抽出
    all_data_df, concepts = kn_stability.extract_data_of_all_checkpoints(args.results_path, training_steps, args.model_name)

    top_k_neurons_dict = extract_top_k_neurons(args, all_data_df, concept_pairs, k)

    # 図の作成
    show_heatmap(args, top_k_neurons_dict, selected_concepts, concept_pairs, k)


if __name__ == '__main__':
    main()
