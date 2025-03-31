# -*- coding: utf-8 -*-
import os
import random
from collections import defaultdict
import json
import pathlib

import logzero
import argparse

import numpy as np
import torch

from knowledge_neurons import KnowledgeNeurons, initialize_model_and_tokenizer, model_type
from utils.data import extract_from_dataset


def make_parser():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--seed', help='the seed. default=42', type=int, default=42)
    parser.add_argument('-mn', '--model_name', help='the name of the model', type=str, default="bert-base-uncased")
    parser.add_argument('-ts', '--training_step',
                        help='(Required when you use pythia models.) the training step of the model',
                        type=str, default="1000",
                        )
    parser.add_argument('-dt', '--dataset_type', help="designate the dataset", default="original_similar_concepts")
    parser.add_argument('-ds', '--dataset_split', help="designate the split of dataset", default="train", type=str)
    parser.add_argument('-dp', '--dataset_path', help="the path for the GenericsKB file which manually downloaded",
                        type=str,
                        default="/home/acd13293hw/sftp_sync/master_research/concept-neurons_by_time/data/GenericsKB")
    parser.add_argument('-nt', '--number_of_templates',
                        help='the minimum number of templates which each entity have. default=4', type=int, default=4
                        )
    parser.add_argument('--local_rank', help="local rank for multigpu processing, default=0", type=int, default=0)
    parser.add_argument('-ln', '--logfile_name', help="designate the file name of log. default='run'", type=str,
                        default="run")
    parser.add_argument('-bs', '--batch_size', help="", type=int, default=20)
    parser.add_argument('--steps', help="number of steps in the integrated grad calculation", type=int, default=20)
    parser.add_argument('-at', '--adaptive_threshold', help="the threshold value", type=float, default=0.3)
    parser.add_argument('-sp', '--sharing_percentage', help="the threshold for the sharing percentage", type=float,
                        default=0.5)
    parser.add_argument('-mw', '--max_words', help="the maximum number of words which each template can have", type=int,
                        default=15)
    parser.add_argument('--save_path', help="results(contain used entities) will be saved under the designed path.",
                        type=str, default='')

    parser.add_argument('--pseudo_concept', help="適当な活性値を取得するために使用する概念", type=str, default="key")
    parser.add_argument('--pseudo_sentence_pythia', help="適当な活性値を取得するために使用する文（Pythia用）", type=str,
                        default="what small, often metal object is used to unlock doors, start vehicles, or operate locks, and can come in different shapes and sizes to fit specific locks and mechanisms ?"
                        )

    return parser.parse_args()


def make_log(args):
    log_directory = os.path.join("log", args.dataset_type)
    os.makedirs(log_directory, exist_ok=True)
    log_file_name = args.logfile_name + ".log"
    log_file_path = os.path.join(log_directory, log_file_name)
    if not os.path.isfile(log_file_path):
        log_file = pathlib.Path(log_file_path)
        log_file.touch()
    logger = logzero.setup_logger(
        logfile=log_file_path,
        disableStderrLogger=False
    )
    logger.info('--------start of script--------')
    logger.info('log will be saved in ' + log_file_path)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    logger.info('random seed is ' + str(args.seed))
    logger.info('model name is ' + args.model_name)

    return logger


def main():
    args = make_parser()

    logger = make_log(args)

    torch.cuda.set_device(args.local_rank)

    # first initialize some hyperparameters
    MODEL_NAME = args.model_name
    TRAINING_STEP = args.training_step

    # these are some hyperparameters for the integrated gradients step
    BATCH_SIZE = args.batch_size
    STEPS = args.steps  # number of steps in the integrated grad calculation
    ADAPTIVE_THRESHOLD = args.adaptive_threshold  # in the paper, they find the threshold value `t` by multiplying the max attribution score by some float - this is that float.
    P = args.sharing_percentage  # the threshold for the sharing percentage

    # setup model & tokenizer
    model, tokenizer = initialize_model_and_tokenizer(MODEL_NAME, TRAINING_STEP)

    # initialize the knowledge neuron wrapper with your model, tokenizer and a string expressing the type of your model ('gpt2' / 'gpt_neo' / 'bert')
    kn = KnowledgeNeurons(model, tokenizer, model_type=model_type(MODEL_NAME))

    # prepare dataset by the conditions
    matched_dataset = extract_from_dataset(args, tokenizer)

    logger.info('Number of entities covered this time: ' + str(len(matched_dataset.keys())))
    logger.info('')

    # 結果の保存先ディレクトリやファイル名を指定する
    if "/" in args.model_name:
        model_name = args.model_name.split("/")[1]
    else:
        model_name = args.model_name
    dataset_and_model = os.path.join(args.dataset_type, args.dataset_split, model_name, f"step{args.training_step}")
    # 実験のハイパラを連結した文字列
    experiment_settings = f"nt_{args.number_of_templates}_at_{args.adaptive_threshold}_mw_{args.max_words}"

    # Write out the entities used as the dataset in this condition to a file.
    if matched_dataset:
        ground_truths = []
        for concept in matched_dataset.keys():
            ground_truths.append(concept)
        dic = {"concept": ground_truths}

        if args.save_path:
            entities_path = os.path.join(args.save_path.rstrip("/"), "entities", dataset_and_model)
        else:
            entities_path = os.path.join("work", "entities", dataset_and_model)
        os.makedirs(entities_path, exist_ok=True)

        save_entities_path = os.path.join(entities_path, f"{experiment_settings}.txt")
        with open(save_entities_path, mode="w") as fi:
            json.dump(dic, fi, indent=4)

    # experiment
    total_sentences = 0
    total_refined_neurons = 0

    if args.save_path:
        result_path = os.path.join(args.save_path.rstrip("/"), dataset_and_model, experiment_settings)
    else:
        result_path = os.path.join("work", "results", dataset_and_model, experiment_settings)
    os.makedirs(result_path, exist_ok=True)
    attribution_scores = os.path.join(result_path, "attribution_scores.jsonl")

    with open(attribution_scores, mode="w", encoding='utf-8') as as_fi:
        for concept, sentences in matched_dataset.items():
            logger.info("Ground Truth: " + concept)
            logger.info('The number of related sentences: ' + str(len(sentences)))
            logger.info(f'Templates: {sentences}')
            logger.info("")

            total_sentences += len(sentences)

            # use the integrated gradients technique to find some refined neurons for your set of prompts
            refined_neurons_data = kn.get_refined_neurons(
                sentences,
                concept,
                p=P,
                batch_size=BATCH_SIZE,
                steps=STEPS,
                coarse_adaptive_threshold=ADAPTIVE_THRESHOLD,
                quiet=True,
            )

            refined_neurons = refined_neurons_data.refined_neurons

            logger.info(f'Getting attribution scores of all neurons for "{concept}" is done')
            total_refined_neurons += len(refined_neurons)

            # 各ニューロンの帰属値を記録する
            values = defaultdict(lambda: defaultdict(list))
            for idx, neuron_idx in enumerate(refined_neurons):
                values[f"{neuron_idx[0]}_{neuron_idx[1]}"]["IGs"] = \
                refined_neurons_data.activations_and_IG[f"{neuron_idx[0]}_{neuron_idx[1]}"]['attribution_scores']
                # values[f"{neuron_idx[0]}_{neuron_idx[1]}"]["kn_sentence_masks"] = \
                # refined_neurons_data.kn_sentence_masks[idx].astype(np.int32).tolist()  # どの正例文で知識ニューロンと判定されたかを表すマスク行列

            record = {concept: values}
            json.dump(record, as_fi, ensure_ascii=False)
            as_fi.write('\n')

        logger.debug('')

    logger.debug('script done!')
    logger.debug('')


if __name__ == '__main__':
    main()
