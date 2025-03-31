import os
import sys
import re
import requests

import json
from collections import defaultdict

from logzero import logger
from tqdm import tqdm

from transformers import BertTokenizer
from datasets import load_dataset

import unicodedata
# from textattack.augmentation import WordNetAugmenter
import nltk
from nltk.corpus import wordnet as wn

# import japanize_matplotlib


def extract_raw_dataset_from_jsonlines(file_path):
    """
    The function which returns the dictionary made from {ConceptNet, TREx, Google_RE or Squad} dataset and whose keys are 'subject, object, template'.
    """

    with open(file_path) as fi:
        dataset_list = []
        for line in tqdm(fi):
            d = defaultdict(str)

            case = json.loads(line)
            try:
                d["sub"] = case["sub_surface"]
            except KeyError:
                try:
                    d["sub"] = case["sub_label"]
                except KeyError:
                    try:
                        d["sub"] = case["sub"]
                    except KeyError:
                        logger.info(f"filepath: {file_path}")
                        logger.info(f"case: {case}")
                        logger.error("There is no key corresponding to 'subject' in this dataset.")
                        sys.exit(1)

            try:
                d["obj"] = case["obj_surface"]
            except KeyError:
                try:
                    d["obj"] = case["obj_label"]
                except KeyError:
                    try:
                        d["obj"] = case["obj"]
                    except KeyError:
                        logger.info(f"filepath: {file_path}")
                        logger.info(f"case: {case}")
                        logger.error("There is no key corresponding to 'object' in this dataset.")
                        sys.exit(1)

            try:
                for masked_sentence in case["masked_sentences"]:
                    if masked_sentence.count("[MASK]") == 1:
                        d["masked_sentence"] = masked_sentence
                # If we couldn't find the masked_sentence that has only one [MASK] token, skip that case.
                if not d["masked_sentence"]:
                    continue
            except KeyError:
                try:
                    d["masked_sentence"] = case["evidences"][0]["masked_sentence"]
                except KeyError:
                    logger.info(f"filepath: {file_path}")
                    logger.info(f"case: {case}")
                    logger.error("There is no 'masked_sentence' key in this dataset.")
                    sys.exit(1)

            dataset_list.append(d)

    return dataset_list


def extract_matched_dataset(dataset_list, entity_type, num_of_templates, max_words, is_remove_unk_concept):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    unk_id = tokenizer.convert_tokens_to_ids("[UNK]")

    if entity_type == "subject":
        d = defaultdict(set)  # key: concept(str), value: set consists of concept's templates

        # At first, replace "[MASK]" to obj, sub to "[MASK]" in the template.
        # Then, append templates as set type to the dictionary whose key is sub.
        # The reason for using the set type is that some templates are the same but have different obj, but if the template is the same, it is counted as one.
        for case in tqdm(dataset_list):
            try:
                r = re.compile(f'{case["sub"]}',
                               re.IGNORECASE)  # Replace sub to "[MASK]" without considering the case of the first letter of sub.
            except re.error:
                logger.warning(f'skipped a case which has sub_label: {case["sub"]}')
                continue

            # Since the original masked_sentence may not contain an exact match between the [MASK] token and the sub to be converted, this case is excluded.
            if not re.search(r, case["masked_sentence"]):
                logger.warning(
                    f'skipped a case which has masked_sentence with no sub_label, sub_label: {case["sub"]}, masked_sentence: {case["masked_sentence"]}')
                continue

            no_mask_sentence = case["masked_sentence"].replace('[MASK]', case["obj"])
            new_masked_sentence = re.sub(r, '[MASK]', no_mask_sentence,
                                         1)  # Make sure that only one mask token appears in a new_masked_sentence.
            new_masked_sentence = unicodedata.normalize("NFKD",
                                                        new_masked_sentence)  # Replace the Unicode's no-break-space and so on.

            # Restrict the number of maximum words in a template.
            if len(new_masked_sentence.split(" ")) <= max_words:
                d[case["sub"]].add(new_masked_sentence)
            else:
                continue

        # Exclude subject entities that do not meet the default number of templates.
        delete_entities = []
        if is_remove_unk_concept:
            for sub in d.keys():
                if len(d[sub]) < num_of_templates or tokenizer.convert_tokens_to_ids(sub) == unk_id:
                    delete_entities.append(sub)
        else:
            for sub in d.keys():
                if len(d[sub]) < num_of_templates:
                    delete_entities.append(sub)
        for delete_key in delete_entities:
            del d[delete_key]

        return d

    elif entity_type == "object":
        d = defaultdict(set)

        # Register a template for each object entity in the dictionary as a set type
        for case in tqdm(dataset_list):
            # Restrict the number of maximum words in a template.
            if len(case["masked_sentence"].split(" ")) <= max_words:
                d[case["obj"]].add(case["masked_sentence"])
            else:
                continue

        # Exclude object entities that do not meet the default number of templates.
        delete_entities = []
        if is_remove_unk_concept:
            for obj in d.keys():
                if len(d[obj]) < num_of_templates or tokenizer.convert_tokens_to_ids(obj) == unk_id:
                    delete_entities.append(obj)
        else:
            for obj in d.keys():
                if len(d[obj]) < num_of_templates:
                    delete_entities.append(obj)
        for delete_key in delete_entities:
            del d[delete_key]

        return d

    else:
        try:
            raise ValueError("entity type is somewhat wrong")
        except ValueError as e:
            print(e)
        sys.exit(1)


def initialize_natural_question_dataset(args, tokenizer):
    dataset = load_dataset(args.dataset_type)
    dataset = dataset[args.dataset_split]

    # # これまでの実験で使用した概念と被る概念だけを対象とする
    # previous_concepts = set()
    # with open("work/entities/generics_kb_best/multiberts-seed_0-step_2000k/nt_4_at_0.2_mw_10.txt", "r") as f:
    #     for line in f:
    #         previous_concepts.add(line.strip().strip(',').strip('"'))

    data_dict = defaultdict(list)
    for idx, data in enumerate(tqdm(dataset)):
        for short_answers_dict in data['annotations']['short_answers']:
            try:
                # GPTのトークナイザでも1単語になる概念のみを対象とする
                # if short_answers_dict["text"][0] in previous_concepts and len(tokenizer.encode(short_answers_dict["text"][0])) == 1:
                if len(tokenizer.encode(short_answers_dict["text"][0])) == 1:
                    concept = short_answers_dict["text"][0]
                    data_dict[concept].append(data['question']["text"] + " ?")
                    break
            except IndexError:
                continue

    # 正例文を Textattack によって増やす
    # def augment(text: str):
    #     augmenter = WordNetAugmenter(
    #         pct_words_to_swap=args.pct_words_to_swap,
    #         transformations_per_example=args.transformations_per_example,
    #     )
    #     return augmenter.augment(text)
    #
    # for concept, original_sentence in tqdm(data_dict.items()):
    #     data_dict[concept] += augment(original_sentence[0])

    # 抽出したデータセットをファイルに書きだす
    root_path = os.path.join("work", "datasets", args.dataset_type, args.dataset_split)
    os.makedirs(root_path, exist_ok=True)
    dataset_path = os.path.join(root_path, "dataset.jsonl")

    with open(dataset_path, mode="w", encoding='utf-8') as fi:
        json.dump(data_dict, fi, ensure_ascii=False)

    return data_dict


def judge_pos(concept: str):
    pos = nltk.pos_tag([concept])[0][1]

    if "NN" in pos:
        concept_tagged = concept + ".n.01"
    elif "VB" in pos:
        concept_tagged = concept + ".v.01"
    elif "JJ" in pos:
        concept_tagged = concept + ".a.01"
    elif "RB" in pos:
        concept_tagged = concept + ".r.01"
    else:
        concept_tagged = ""

    return concept_tagged


def extract_from_dataset(args, tokenizer):
    """
    データセットをロードし、その中から指定した条件を満たす概念とセンテンスを抽出する。

    dataset: {
            'source': Value(dtype='string', id=None),
              'term': Value(dtype='string', id=None),
              'quantifier_frequency': Value(dtype='string', id=None),
              'quantifier_number': Value(dtype='string', id=None),
              'generic_sentence': Value(dtype='string', id=None),
              'score': Value(dtype='float64', id=None)
            }
        -> dict{'term': [generic_sentences]}
        (= dict{'concept': [templates]})
    """

    ids = range(tokenizer.vocab_size)
    lm_vocab = set(tokenizer.convert_ids_to_tokens(ids))
    # lm_vocab = set(tokenizer.vocab.keys())

    dic = defaultdict(list)

    if "generics_kb" in args.dataset_type:
        dataset = load_dataset("generics_kb", args.dataset_type, data_dir=args.dataset_path)
        # more data => load_dataset("generics_kb","generics_kb", data_dir="<path/to/manual/data>")
        # others => load_dataset("generics_kb","generics_kb_simplewiki", data_dir="<path/to/manual/data>"),
        #           or load_dataset("generics_kb","generics_kb_waterloo", data_dir="<path/to/manual/data>")

        # デバッグのためのコード（小さいデータで動作確認）
        # dataset["train"] = dataset["train"][:300]
        # dataset["train"] = dataset["train"][:2347]
        # dataset["train"] = dataset["train"][:108347]
        # dataset["train"] = dataset["train"][args.dataset_start_index:args.dataset_end_index]

        # 言語モデルの語彙辞書に載っている"term"（＝概念）の"generic_sentence"（単語数はargs.max_words以下）を取ってくる
        for term, generic_sentence in zip(tqdm(dataset["train"]["term"]), dataset["train"]["generic_sentence"]):
            if term in lm_vocab and len(generic_sentence.split(" ")) <= args.max_words:
                dic[term].append(generic_sentence.lower())

        # 各テンプレートをマスク
        for concept, templates in tqdm(dic.items()):
            masked_templates = []
            for template in templates:
                tokenized_template = tokenizer.tokenize(template)
                masked_idx_list = [i for i, x in enumerate(tokenized_template) if x == concept]
                # テンプレートにconceptが一つも入っていない(ex) concept="carry", テンプレート="Sound carries well over water.")、または二つ以上入ってしまっている場合は取り除く
                if len(masked_idx_list) != 1:
                    # print(f"skipped: concept='{concept}', template='{template}'")
                    continue
                tokenized_template[masked_idx_list[0]] = "[MASK]"
                masked_templates.append(" ".join(tokenized_template))
            dic[concept] = masked_templates

        # 一概念あたりのテンプレート数が基準に満たない概念を削除
        delete_entities = []
        for concept in dic.keys():
            if len(dic[concept]) < args.number_of_templates:
                delete_entities.append(concept)
        for delete_key in delete_entities:
            del dic[delete_key]

        return dic
    elif args.dataset_type == "natural_questions":
        try:
            dataset_path = os.path.join("work", "datasets", args.dataset_type, args.dataset_split)
            fp = open(os.path.join(dataset_path, "dataset.jsonl"), "r")
            return json.load(fp)
        except FileNotFoundError:
            return initialize_natural_question_dataset(args, tokenizer)
    elif "original" in args.dataset_type:
        try:
            fp = open(args.dataset_path, "r")
            raw_dataset = json.load(fp)

            # 概念が1トークンの場合のみを抽出
            if args.dataset_type == "original_frequencies":
                dic = raw_dataset
            else:
                for concept, templates in raw_dataset.items():
                    if len(tokenizer.encode(concept)) == 1:
                        dic[concept] = templates

            # 品詞の場合、品詞判定結果を出力しておく
            if args.dataset_type == "original_poses":
                poses = nltk.pos_tag(dic.keys())
                print(f"概念の品詞判定結果: {poses}")

            return dic
        except FileNotFoundError as e:
            print(f"Given args 'dataset_path'(='{args.dataset_path}') is not exist.\n{e}")
            sys.exit(1)
    elif "net" in args.dataset_type:
        dataset_dict = defaultdict(list)
        concept_dict = defaultdict(list)
        base_patchscope_sentence = f"screen = screen; Sirius = Sirius; mountain = mountain; "
        if "conceptnet" in args.dataset_type:
            # 上位概念(hypernyms)の定義
            if "similar_concepts" in args.dataset_type:
                hypernyms = ["color", "music", "food", "atom", ]
            else:
                raise NotImplementedError

            # 下位概念(hyponyms)を取り出す
            for hypernym in hypernyms:
                obj_url = f'http://api.conceptnet.io/c/en/{hypernym}?rel=/r/HasA'
                obj = requests.get(obj_url)
                assert obj.status_code == 200, f"Conceptnet api status code: {obj.status_code}(!= 200)"

                raw_hyponyms = set(edge.get('end', {}).get('label', None) for edge in obj.json().get('edges', []))
                # 正規化処理
                regularized_hyponyms = set([
                    re.sub(r"^(a |an |the )", "", hyponym.strip(), flags=re.IGNORECASE) for hyponym in raw_hyponyms
                ])
                one_word_regularized_hyponyms = [hyponym for hyponym in regularized_hyponyms if len(hyponym.split(" ")) == 1]

                concept_dict[hypernym] = one_word_regularized_hyponyms

                # Wordnet を用いて、下位概念についての定義文を取得する
                for hyponym in one_word_regularized_hyponyms:
                    try:
                        hyponym_pos_tagged = judge_pos(hyponym)
                        question = wn.synset(hyponym_pos_tagged).definition()
                        data_sentence = "Answer the following question in one word: Q. What is " + question + "? A. "
                    except nltk.corpus.reader.wordnet.WordNetError as e:
                        print(f"The concept '{hyponym}' skipped; Error: {e}")
                        continue

                    dataset_dict[hyponym].append(data_sentence)

                    # patchscope sentence を追加
                    patchscope_sentence = base_patchscope_sentence + f"{hyponym} = "
                    dataset_dict[hyponym].append(patchscope_sentence)
        elif "wordnet" in args.dataset_type:
            # 上位概念(hypernyms)の定義
            if "similar_concepts" in args.dataset_type:
                hypernyms = ["beverage.n.01", "music.n.01", "furniture.n.01", "dog.n.01", ]
            else:
                raise NotImplementedError

            # 下位概念(hyponyms)を取り出す
            for hypernym in hypernyms:
                hyponyms = wn.synset(hypernym).hyponyms()

                # # アンダーバー"_"を含む概念は除外する
                # concept_dict[hypernym] = [
                #     hyponym.name().split(".")[0] for hyponym in hyponyms if len(hyponym.name().split(".")[0].split("_")) == 1
                # ]

                for hyponym in hyponyms:
                    hyponym_pos_tagged = hyponym.name()
                    hyponym = hyponym.name().split(".")[0]
                    if len(hyponym_pos_tagged.split(".")[0].split("_")) == 1:  # アンダーバー"_"を含む概念は除外する
                        concept_dict[hypernym.split(".")[0]].append(hyponym_pos_tagged.split(".")[0])
                    else:
                        continue

                    # Wordnet を用いて、下位概念についての定義文を取得する
                    try:
                        question = wn.synset(hyponym_pos_tagged).definition()
                        data_sentence = "Answer the following question in one word: Q. What is " + question + "? A. "
                    except nltk.corpus.reader.wordnet.WordNetError as e:
                        print(f"The concept '{hyponym}' skipped; Error: {e}")
                        continue

                    dataset_dict[hyponym].append(data_sentence)

                    # patchscope sentence を追加
                    patchscope_sentence = base_patchscope_sentence + f"{hyponym} = "
                    dataset_dict[hyponym].append(patchscope_sentence)

        return dataset_dict
    else:
        print(f"dataset type '{args.dataset_type}' is not supported")
        raise NotImplementedError
