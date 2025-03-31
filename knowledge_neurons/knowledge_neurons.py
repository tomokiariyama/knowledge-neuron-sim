# main knowledge neurons class
import torch
import torch.nn.functional as F
import torch.nn as nn
import einops
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from types import SimpleNamespace
from typing import List, Optional, Tuple, Callable, Any
import math
from functools import partial
from transformers import PreTrainedTokenizerBase
from .patch import *
from logzero import logger


class KnowledgeNeurons:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        model_type: str = "bert",
        device: str = None,
    ):
        self.model = model
        self.model_type = model_type
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.tokenizer = tokenizer

        self.baseline_activations = None

        self.autoregressive_model_names = ["gpt", "olmo"]

        if self.model_type == "bert":
            self.transformer_layers_attr = "bert.encoder.layer"
            self.input_ff_attr = "intermediate"
            self.output_ff_attr = "output.dense.weight"
            self.word_embeddings_attr = "bert.embeddings.word_embeddings.weight"
            self.unk_token = getattr(self.tokenizer, "unk_token_id", None)
        elif self.model_type == "mbert":
            self.transformer_layers_attr = "encoder.layer"
            self.input_ff_attr = "intermediate"
            self.output_ff_attr = "output.dense.weight"
            self.word_embeddings_attr = "embeddings.word_embeddings.weight"
            self.unk_token = getattr(self.tokenizer, "unk_token_id", None)
        elif self.model_type == "gpt_neo":
            self.transformer_layers_attr = "transformer.h"
            self.input_ff_attr = "mlp.c_fc"
            self.output_ff_attr = "mlp.c_proj.weight"
            self.word_embeddings_attr = "transformer.wpe"
        elif self.model_type == "gpt_neox":
            self.transformer_layers_attr = "gpt_neox.layers"
            self.input_ff_attr = "mlp.dense_h_to_4h"
            self.output_ff_attr = "mlp.dense_4h_to_h"
            self.word_embeddings_attr = "gpt_neox.embed_in"
        elif self.model_type == "olmo":
            self.transformer_layers_attr = "model.layers"
            self.input_ff_attr = "mlp.up_proj"
            self.output_ff_attr = "mlp.down_proj"
            self.word_embeddings_attr = "model.embed_tokens"
        else:
            raise NotImplementedError

    def _get_output_ff_layer(self, layer_idx):
        return get_ff_layer(
            self.model,
            layer_idx,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.output_ff_attr,
        )

    def _get_input_ff_layer(self, layer_idx):
        return get_ff_layer(
            self.model,
            layer_idx,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.input_ff_attr,
        )

    def _get_word_embeddings(self):
        return get_attributes(self.model, self.word_embeddings_attr)

    def _get_transformer_layers(self):
        return get_attributes(self.model, self.transformer_layers_attr)

    def _prepare_inputs(self, prompt, target=None, encoded_input=None):
        if encoded_input is None:
            encoded_input = self.tokenizer(prompt, max_length=128, return_tensors="pt").to(self.device)
        if self.model_type == "bert":
            mask_idx = torch.where(
                encoded_input["input_ids"][0] == self.tokenizer.mask_token_id
            )[0].item()
        else:
            # with autoregressive models we always want to target the last token
            mask_idx = -1
        if target is not None:
            if any(ar_model_name in self.model_type for ar_model_name in self.autoregressive_model_names):
                target = self.tokenizer.encode(target)
            else:
                target = self.tokenizer.convert_tokens_to_ids(target)

        return encoded_input, mask_idx, target

    def _generate(self, prompt, ground_truth):
        encoded_input, mask_idx, target_label = self._prepare_inputs(
            prompt, ground_truth
        )
        # for autoregressive models, we might want to generate > 1 token
        if any(ar_model_name in self.model_type for ar_model_name in self.autoregressive_model_names):
            n_sampling_steps = len(target_label)
        else:
            n_sampling_steps = 1  # TODO: we might want to use multiple mask tokens even with bert models

        all_gt_probs = []
        all_gt_ranks = []
        all_argmax_probs = []
        argmax_tokens = []
        argmax_completion_str = ""

        for i in range(n_sampling_steps):
            if i > 0:
                # retokenize new inputs
                encoded_input, mask_idx, target_label = self._prepare_inputs(
                    prompt, ground_truth
                )
            outputs = self.model(**encoded_input)
            probs = F.softmax(outputs.logits[:, mask_idx, :], dim=-1)
            target_idx = target_label[i]
            gt_prob = probs[:, target_idx].item()
            all_gt_probs.append(gt_prob)

            sorted_probs = probs.argsort(descending=True)  # vocab上で確率の大きい順にソートしたtensor
            gt_rank = torch.where(sorted_probs == target_idx)[1].item()  # サブワードが語彙上で何番目に出力確率が大きいか
            all_gt_ranks.append((self.tokenizer.decode(target_idx), gt_rank))

            # get info about argmax completion
            argmax_prob, argmax_id = [i.item() for i in probs.max(dim=-1)]
            argmax_tokens.append(argmax_id)
            argmax_str = self.tokenizer.decode([argmax_id])
            all_argmax_probs.append(argmax_prob)

            prompt += argmax_str
            argmax_completion_str += argmax_str

        gt_prob = math.prod(all_gt_probs) if len(all_gt_probs) > 1 else all_gt_probs[0]
        argmax_prob = (
            math.prod(all_argmax_probs)
            if len(all_argmax_probs) > 1
            else all_argmax_probs[0]
        )
        return gt_prob, all_gt_ranks, argmax_prob, argmax_completion_str, argmax_tokens

    def n_layers(self):
        return len(self._get_transformer_layers())

    def intermediate_size(self):
        if self.model_type == "bert":
            return self.model.config.intermediate_size
        else:
            return self.model.config.hidden_size * 4

    @staticmethod
    def scaled_input(activations: torch.Tensor, steps: int = 20, device: str = "cpu"):
        """
        Tiles activations along the batch dimension - gradually scaling them over
        `steps` steps from 0 to their original value over the batch dimensions.

        `activations`: torch.Tensor
        original activations
        `steps`: int
        number of steps to take
        """
        tiled_activations = einops.repeat(activations, "b d -> (r b) d", r=steps)
        out = (
            tiled_activations
            * torch.linspace(start=0, end=1, steps=steps).to(device)[:, None]
        )
        return out

    def get_baseline_with_activations(
        self, encoded_input: dict, layer_idx: int, mask_idx: int
    ):
        """
        Gets the baseline outputs and activations for the unmodified model at a given index.

        `encoded_input`: torch.Tensor
            the inputs to the model from self.tokenizer.encode_plus()
        `layer_idx`: int
            which transformer layer to access
        `mask_idx`: int
            the position at which to get the activations (TODO: rename? with autoregressive models there's no mask, so)
        """

        def get_activations(model, layer_idx, mask_idx):
            """
            This hook function should assign the intermediate activations at a given layer / mask idx
            to the 'self.baseline_activations' variable
            """

            def hook_fn(acts):
                self.baseline_activations = acts[:, mask_idx, :]

            return register_hook(
                model,
                layer_idx=layer_idx,
                f=hook_fn,
                transformer_layers_attr=self.transformer_layers_attr,
                ff_attrs=self.input_ff_attr,
            )

        handle = get_activations(self.model, layer_idx=layer_idx, mask_idx=mask_idx)
        baseline_outputs = self.model(**encoded_input)
        handle.remove()
        baseline_activations = self.baseline_activations
        self.baseline_activations = None
        return baseline_outputs, baseline_activations

    def get_scores(
        self,
        prompt: str,
        ground_truth: str,
        batch_size: int = 10,
        steps: int = 20,
        attribution_method: str = "integrated_grads",
        pbar: bool = True,
    ):
        """
        Gets the attribution scores for a given prompt and ground truth.
        `prompt`: str
            the prompt to get the attribution scores for
        `ground_truth`: str
            the ground truth / expected output
        `batch_size`: int
            batch size
        `steps`: int
            total number of steps (per token) for the integrated gradient calculations
        `attribution_method`: str
            the method to use for getting the scores. Choose from 'integrated_grads' or 'max_activations'.
        """

        scores = []
        activations = []
        encoded_input = self.tokenizer(prompt, max_length=128, return_tensors="pt").to(self.device)
        for layer_idx in tqdm(
            range(self.n_layers()),
            desc="Getting attribution scores for each layer...",
            disable=not pbar,
        ):
            layer_scores, neuron_activations = self.get_scores_for_layer(
                prompt,
                ground_truth,
                encoded_input=encoded_input,
                layer_idx=layer_idx,
                batch_size=batch_size,
                steps=steps,
                attribution_method=attribution_method,
            )
            scores.append(layer_scores)
            activations.append(neuron_activations)
        return torch.stack(scores), torch.stack(activations)

    def get_coarse_neurons(
        self,
        prompt: str,
        ground_truth: str,
        batch_size: int = 10,
        steps: int = 20,
        threshold: float = None,
        adaptive_threshold: float = None,
        percentile: float = None,
        attribution_method: str = "integrated_grads",
        pbar: bool = True,
    ) -> SimpleNamespace:
        # -> SimpleNamespace(List[List[int]], List[float], List[float]):
        """
        Finds the 'coarse' neurons for a given prompt and ground truth.
        The coarse neurons are the neurons that are most activated by a single prompt.
        We refine these by using multiple prompts that express the same 'fact'/relation in different ways.

        `prompt`: str
            the prompt to get the coarse neurons for
        `ground_truth`: str
            the ground truth / expected output
        `batch_size`: int
            batch size
        `steps`: int
            total number of steps (per token) for the integrated gradient calculations
        `threshold`: float
            `t` from the paper. If not None, then we only keep neurons with integrated grads above this threshold.
        `adaptive_threshold`: float
            Adaptively set `threshold` based on `maximum attribution score * adaptive_threshold` (in the paper, they set adaptive_threshold=0.3)
        `percentile`: float
            If not None, then we only keep neurons with integrated grads in this percentile of all integrated grads.
        `attribution_method`: str
            the method to use for getting the scores. Choose from 'integrated_grads' or 'max_activations'.
        """
        attribution_scores, activations = self.get_scores(
            prompt,
            ground_truth,
            batch_size=batch_size,
            steps=steps,
            pbar=pbar,
            attribution_method=attribution_method,
        )
        assert (
            sum(e is not None for e in [threshold, adaptive_threshold, percentile]) == 1
        ), f"Provide one and only one of threshold / adaptive_threshold / percentile"
        if adaptive_threshold is not None:
            threshold = attribution_scores.max().item() * adaptive_threshold
        if threshold is not None:
            # coarse_neurons のインデックスだけでなく、活性値とIGの値も返す
            masks = attribution_scores > threshold
            return SimpleNamespace(
                indices=torch.nonzero(masks).cpu().tolist(),
                activations=activations[masks].cpu().tolist(),
                attribution_scores=attribution_scores[masks].cpu().tolist(),
                max_attribution_score=attribution_scores.max().item(),
            )
        else:
            s = attribution_scores.flatten().detach().cpu().numpy()
            masks = attribution_scores > np.percentile(s, percentile)
            return SimpleNamespace(
                indices=torch.nonzero(masks).cpu().tolist(),
                activations=activations[masks].cpu().tolist(),
                attribution_scores=attribution_scores[masks].cpu().tolist(),
                max_attribution_score=attribution_scores.max().item(),
            )

    def get_refined_neurons(
        self,
        prompts: List[str],
        ground_truth: str,
        negative_examples: Optional[List[str]] = None,
        p: float = 0.5,
        batch_size: int = 10,
        steps: int = 20,
        coarse_adaptive_threshold: Optional[float] = 0.3,
        coarse_threshold: Optional[float] = None,
        coarse_percentile: Optional[float] = None,
        quiet=False,
    ) -> SimpleNamespace:
        # -> Tuple[List[List[Any]], defaultdict[Any, defaultdict[Any, list]]]:
        """
        Finds the 'refined' neurons for a given set of prompts and a ground truth / expected output.

        The input should be n different prompts, each expressing the same fact in different ways.
        For each prompt, we calculate the attribution scores of each intermediate neuron.
        We then set an attribution score threshold, and we keep the neurons that are above this threshold.
        Finally, considering the coarse neurons from all prompts, we set a sharing percentage threshold, p,
        and retain only neurons shared by more than p% of prompts.

        `prompts`: list of str
            the prompts to get the refined neurons for
        `ground_truth`: str
            the ground truth / expected output
        `negative_examples`: list of str
            Optionally provide a list of negative examples. Any neuron that appears in these examples will be excluded from the final results.
        `p`: float
            the threshold for the sharing percentage
        `batch_size`: int
            batch size
        `steps`: int
            total number of steps (per token) for the integrated gradient calculations
        `coarse_threshold`: float
            threshold for the coarse neurons
        `coarse_percentile`: float
            percentile for the coarse neurons
        """
        assert isinstance(
            prompts, list
        ), "Must provide a list of different prompts to get refined neurons"
        assert 0.0 <= p < 1.0, "p should be a float between 0 and 1"

        n_prompts = len(prompts)
        coarse_neurons = []
        coarse_neurons_activations_and_IG = defaultdict(lambda: defaultdict(list))
        max_attribution_scores = []
        for prompt in tqdm(
            prompts, desc="Getting coarse neurons for each prompt...", disable=quiet
        ):
            coarse_neurons_data = self.get_coarse_neurons(
                prompt,
                ground_truth,
                batch_size=batch_size,
                steps=steps,
                adaptive_threshold=coarse_adaptive_threshold,
                threshold=coarse_threshold,
                percentile=coarse_percentile,
                pbar=False,
            )
            coarse_neurons.append(
                coarse_neurons_data.indices
            )
            for coarse_neurons_in_this_step, activation, attribution_score in zip(coarse_neurons_data.indices, coarse_neurons_data.activations, coarse_neurons_data.attribution_scores):
                coarse_neurons_activations_and_IG[f"{coarse_neurons_in_this_step[0]}_{coarse_neurons_in_this_step[1]}"]["activations"].append(activation)
                coarse_neurons_activations_and_IG[f"{coarse_neurons_in_this_step[0]}_{coarse_neurons_in_this_step[1]}"]["attribution_scores"].append(attribution_score)
            max_attribution_scores.append(coarse_neurons_data.max_attribution_score)
        if negative_examples is not None:
            negative_neurons = []
            for negative_example in tqdm(
                negative_examples,
                desc="Getting coarse neurons for negative examples",
                disable=quiet,
            ):
                negative_neurons.append(
                    self.get_coarse_neurons(
                        negative_example,
                        ground_truth,
                        batch_size=batch_size,
                        steps=steps,
                        adaptive_threshold=coarse_adaptive_threshold,
                        threshold=coarse_threshold,
                        percentile=coarse_percentile,
                        pbar=False,
                    ).indices
                )
        if not quiet:
            total_coarse_neurons = sum([len(i) for i in coarse_neurons])
            logger.info(f"\n{total_coarse_neurons} coarse neurons found - refining")
        t = n_prompts * p
        refined_neurons = []

        # c = collections.Counter()
        # for neurons in coarse_neurons:
        #     for n in neurons:
        #         c[tuple(n)] += 1
        d = collections.defaultdict(list)
        for positive_sentence_index, neurons in enumerate(coarse_neurons):
            for n in neurons:
                d[tuple(n)].append(positive_sentence_index)

        # for neuron, count in c.items():
        #     if count >= t:
        #         refined_neurons.append(list(neuron))
        #     else:
        #         del coarse_neurons_activations_and_IG[f"{neuron[0]}_{neuron[1]}"]
        kn_sentence_masks = []
        for neuron, sentence_index_list in d.items():
            if len(sentence_index_list) >= t:
                refined_neurons.append(list(neuron))
                # どの正例文で知識ニューロンと判定されたか、マスク行列を作る
                kn_sentence_mask = np.zeros(len(coarse_neurons))
                for sentence_index in sentence_index_list:
                    kn_sentence_mask[sentence_index] = 1
                kn_sentence_masks.append(kn_sentence_mask)
            else:
                del coarse_neurons_activations_and_IG[f"{neuron[0]}_{neuron[1]}"]

        refined_neurons_activations_and_IG = coarse_neurons_activations_and_IG

        # filter out neurons that are in the negative examples
        if negative_examples is not None:
            for neuron in negative_neurons:
                if neuron in refined_neurons:
                    refined_neurons.remove(neuron)

        total_refined_neurons = len(refined_neurons)
        if not quiet:
            logger.info(f"{total_refined_neurons} neurons remaining after refining")
        return SimpleNamespace(
            refined_neurons=refined_neurons,
            activations_and_IG=refined_neurons_activations_and_IG,
            kn_sentence_masks=kn_sentence_masks,
            max_attribution_scores=max_attribution_scores,
        )

    def get_scores_for_layer(
        self,
        prompt: str,
        ground_truth: str,
        layer_idx: int,
        batch_size: int = 10,
        steps: int = 20,
        encoded_input: Optional[int] = None,
        attribution_method: str = "integrated_grads",
    ):
        """
        get the attribution scores for a given layer
        `prompt`: str
            the prompt to get the attribution scores for
        `ground_truth`: str
            the ground truth / expected output
        `layer_idx`: int
            the layer to get the scores for
        `batch_size`: int
            batch size
        `steps`: int
            total number of steps (per token) for the integrated gradient calculations
        `encoded_input`: int
            if not None, then use this encoded input instead of getting a new one
        `attribution_method`: str
            the method to use for getting the scores. Choose from 'integrated_grads' or 'max_activations'.
        """
        assert steps % batch_size == 0
        n_batches = steps // batch_size

        # First we take the unmodified model and use a hook to return the baseline intermediate activations at our chosen target layer
        encoded_input, mask_idx, target_label = self._prepare_inputs(
            prompt, ground_truth, encoded_input
        )

        # for autoregressive models, we might want to generate > 1 token
        if any(ar_model_name in self.model_type for ar_model_name in self.autoregressive_model_names):
            n_sampling_steps = len(target_label)
        else:
            n_sampling_steps = 1  # TODO: we might want to use multiple mask tokens even with bert models

        if attribution_method == "integrated_grads":
            integrated_grads = []
            neuron_activations = []

            for i in range(n_sampling_steps):
                #if i > 0 and self.model_type == "gpt":
                if i > 0 and "gpt" in self.model_type:
                    # retokenize new inputs
                    encoded_input, mask_idx, target_label = self._prepare_inputs(
                        prompt, ground_truth
                    )
                (
                    baseline_outputs,
                    baseline_activations,
                ) = self.get_baseline_with_activations(
                    encoded_input, layer_idx, mask_idx
                )
                if n_sampling_steps > 1:
                    argmax_next_token = (
                        baseline_outputs.logits[:, mask_idx, :].argmax(dim=-1).item()
                    )
                    next_token_str = self.tokenizer.decode(argmax_next_token)

                # Now we want to gradually change the intermediate activations of our layer from 0 -> their original value
                # and calculate the integrated gradient of the masked position at each step
                # we do this by repeating the input across the batch dimension, multiplying the first batch by 0, the second by 0.1, etc., until we reach 1
                # `scaled_weights`は、0から元の活性値までの20分割の値が入ったもの（リーマン近似用）
                scaled_weights = self.scaled_input(
                    baseline_activations, steps=steps, device=self.device
                )
                scaled_weights.requires_grad_(True)

                integrated_grads_this_step = []  # to store the integrated gradients

                for batch_weights in scaled_weights.chunk(n_batches):
                    # we want to replace the intermediate activations at some layer, at the mask position, with `batch_weights`
                    # first tile the inputs to the correct batch size
                    inputs = {
                        "input_ids": einops.repeat(
                            encoded_input["input_ids"], "b d -> (r b) d", r=batch_size
                        ),
                        "attention_mask": einops.repeat(
                            encoded_input["attention_mask"],
                            "b d -> (r b) d",
                            r=batch_size,
                        ),
                    }
                    if self.model_type == "bert":
                        inputs["token_type_ids"] = einops.repeat(
                            encoded_input["token_type_ids"],
                            "b d -> (r b) d",
                            r=batch_size,
                        )

                    # then patch the model to replace the activations with the scaled activations
                    patch_ff_layer(
                        self.model,
                        layer_idx=layer_idx,
                        mask_idx=mask_idx,
                        replacement_activations=batch_weights,
                        transformer_layers_attr=self.transformer_layers_attr,
                        ff_attrs=self.input_ff_attr,
                    )

                    # then forward through the model to get the logits
                    if "olmo" in self.model_type:
                        outputs = self.model(**inputs, max_new_tokens=10, do_sample=True, top_k=50, top_p=0.95)
                    else:
                        outputs = self.model(**inputs)

                    # then calculate the gradients for each step w/r/t the inputs
                    probs = F.softmax(outputs.logits[:, mask_idx, :], dim=-1)
                    if n_sampling_steps > 1:
                        target_idx = target_label[i]
                    else:
                        target_idx = target_label
                    grad = torch.autograd.grad(
                        torch.unbind(probs[:, target_idx]), batch_weights
                    )[0]
                    grad = grad.sum(dim=0)
                    integrated_grads_this_step.append(grad)

                    unpatch_ff_layer(
                        self.model,
                        layer_idx=layer_idx,
                        transformer_layers_attr=self.transformer_layers_attr,
                        ff_attrs=self.input_ff_attr,
                    )

                # then sum, and multiply by W-hat / m
                integrated_grads_this_step = torch.stack(
                    integrated_grads_this_step, dim=0
                ).sum(dim=0)
                integrated_grads_this_step *= baseline_activations.squeeze(0) / steps
                # integrated_grads_this_step *= abs(baseline_activations.squeeze(0)) / steps  # 帰属値の計算において、積分値に掛ける活性値の絶対値を取る場合

                integrated_grads.append(integrated_grads_this_step)

                # 各ニューロンの活性値の情報（baseline_activations.squeeze(0)）も保持する
                neuron_activations.append(baseline_activations.squeeze(0))
                # integrated_grads_this_step = torch.stack([integrated_grads_this_step, baseline_activations.squeeze(0)])
                if n_sampling_steps > 1:
                    prompt += next_token_str

            integrated_grads = torch.stack(integrated_grads, dim=0).sum(dim=0) / len(integrated_grads)
            neuron_activations = torch.stack(neuron_activations, dim=0).sum(dim=0) / len(neuron_activations)

            return integrated_grads, neuron_activations
        elif attribution_method == "max_activations":
            activations = []
            for i in range(n_sampling_steps):
                if i > 0 and self.model_type == "gpt":
                    # retokenize new inputs
                    encoded_input, mask_idx, target_label = self._prepare_inputs(
                        prompt, ground_truth
                    )
                (
                    baseline_outputs,
                    baseline_activations,
                ) = self.get_baseline_with_activations(
                    encoded_input, layer_idx, mask_idx
                )
                activations.append(baseline_activations)
                if n_sampling_steps > 1:
                    argmax_next_token = (
                        baseline_outputs.logits[:, mask_idx, :].argmax(dim=-1).item()
                    )
                    next_token_str = self.tokenizer.decode(argmax_next_token)
                    prompt += next_token_str
            activations = torch.stack(activations, dim=0).sum(dim=0) / len(activations)
            return activations.squeeze(0)
        else:
            raise NotImplementedError

    def modify_activations(
        self,
        prompt: str,
        ground_truth: str,
        neurons: List[List[int]],
        mode: str = "suppress",
        replacement_activations: Optional[torch.Tensor] = None,
        undo_modification: bool = True,
        quiet: bool = False,
    ) -> Tuple[dict, Callable]:
        results_dict = {}
        _, mask_idx, _ = self._prepare_inputs(
            prompt, ground_truth
        )  # just need to get the mask index for later - probably a better way to do this
        # get the baseline probabilities of the groundtruth being generated + the argmax / greedy completion before modifying the activations
        (
            gt_baseline_prob,
            gt_baseline_ranks,
            argmax_baseline_prob,
            argmax_completion_str,
            _,
        ) = self._generate(prompt, ground_truth)
        if not quiet:
            logger.info(
                f"\nBefore modification - groundtruth probability: {gt_baseline_prob}\n"
                f"groundtruth output rank: {gt_baseline_ranks[0][1]}\n"
                f"Argmax completion: `{argmax_completion_str}`\n"
                f"Argmax prob: {argmax_baseline_prob}\n"
            )
        results_dict["before"] = {
            "gt_prob": gt_baseline_prob,
            "gt_rank": gt_baseline_ranks[0][1],
            "argmax_completion": argmax_completion_str,
            "argmax_prob": argmax_baseline_prob,
        }

        # patch model to suppress neurons
        # store all the layers we patch so we can unpatch them later
        all_layers = set([n[0] for n in neurons])

        patch_ff_layer(
            self.model,
            mask_idx,
            mode=mode,
            neurons=neurons,
            replacement_activations=replacement_activations,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.input_ff_attr,
        )

        # get the probabilities of the groundtruth being generated + the argmax / greedy completion after modifying the activations
        new_gt_prob, new_gt_ranks, new_argmax_prob, new_argmax_completion_str, _ = self._generate(
            prompt, ground_truth
        )
        if not quiet:
            logger.info(
                f"After modification - groundtruth probability: {new_gt_prob}\n"
                f"groundtruth output rank: {new_gt_ranks[0][1]}\n"
                f"Argmax completion: `{new_argmax_completion_str}`\n"
                f"Argmax prob: {new_argmax_prob}\n"
            )
        results_dict["after"] = {
            "gt_prob": new_gt_prob,
            "gt_rank": new_gt_ranks[0][1],
            "argmax_completion": new_argmax_completion_str,
            "argmax_prob": new_argmax_prob,
        }

        unpatch_fn = partial(
            unpatch_ff_layers,
            model=self.model,
            layer_indices=all_layers,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.input_ff_attr,
        )

        if undo_modification:
            unpatch_fn()
            unpatch_fn = lambda *args: args

        return results_dict, unpatch_fn

    def suppress_knowledge(
        self,
        prompt: str,
        ground_truth: str,
        neurons: List[List[int]],
        mode: str = "suppress",
        replacement_activations: Optional[torch.Tensor] = None,
        undo_modification: bool = True,
        quiet: bool = False,
    ) -> Tuple[dict, Callable]:
        if mode == "suppress":
            """
            prompt the model with `prompt`, zeroing the activations at the positions specified by `neurons`,
            and measure the resulting affect on the ground truth probability.
            """
            return self.modify_activations(
                prompt=prompt,
                ground_truth=ground_truth,
                neurons=neurons,
                mode=mode,
                undo_modification=undo_modification,
                quiet=quiet,
            )
        elif mode == "suppress_by_replace":
            """
            prompt the model with `prompt`, replacing the activations at the positions specified by `neurons`,
            and measure the resulting affect on the ground truth probability.
            """
            return self.modify_activations(
                prompt=prompt,
                ground_truth=ground_truth,
                neurons=neurons,
                mode=mode,
                replacement_activations=replacement_activations,
                undo_modification=undo_modification,
                quiet=quiet,
            )
        else:
            raise NotImplementedError

    def enhance_knowledge(
        self,
        prompt: str,
        ground_truth: str,
        neurons: List[List[int]],
        undo_modification: bool = True,
        quiet: bool = False,
    ) -> Tuple[dict, Callable]:
        """
        prompt the model with `prompt`, multiplying the activations at the positions
        specified by `neurons` by 2, and measure the resulting affect on the ground truth probability.
        """
        return self.modify_activations(
            prompt=prompt,
            ground_truth=ground_truth,
            neurons=neurons,
            mode="enhance",
            undo_modification=undo_modification,
            quiet=quiet,
        )

    @torch.no_grad()
    def modify_weights(
        self,
        prompt: str,
        neurons: List[List[int]],
        target: str,
        mode: str = "edit",
        erase_value: str = "zero",
        undo_modification: bool = True,
        quiet: bool = False,
    ) -> Tuple[dict, Callable]:
        """
        Update the *weights* of the neural net in the positions specified by `neurons`.
        Specifically, the weights of the second Linear layer in the ff are updated by adding or subtracting the value
        of the word embeddings for `target`.
        """
        assert mode in ["edit", "erase"]
        assert erase_value in ["zero", "unk"]
        results_dict = {}

        _, _, target_label = self._prepare_inputs(prompt, target)
        # get the baseline probabilities of the target being generated + the argmax / greedy completion before modifying the weights
        (
            gt_baseline_prob,
            gt_baseline_ranks,
            argmax_baseline_prob,
            argmax_completion_str,
            argmax_tokens,
        ) = self._generate(prompt, target)
        if not quiet:
            logger.info(
                f"\nBefore modification - groundtruth probability: {gt_baseline_prob}\n"
                f"groundtruth rank: {gt_baseline_ranks[0][1]}\n"
                f"Argmax completion: `{argmax_completion_str}`\n"
                f"Argmax prob: {argmax_baseline_prob}"
            )
        results_dict["before"] = {
            "gt_prob": gt_baseline_prob,
            "gt_rank": gt_baseline_ranks[0][1],
            "argmax_completion": argmax_completion_str,
            "argmax_prob": argmax_baseline_prob,
        }

        # get the word embedding values of the baseline + target predictions
        word_embeddings_weights = self._get_word_embeddings()
        if mode == "edit":
            assert (
                self.model_type == "bert"
            ), "edit mode currently only working for bert models - TODO"
            original_prediction_id = argmax_tokens[0]
            original_prediction_embedding = word_embeddings_weights[
                original_prediction_id
            ]
            target_embedding = word_embeddings_weights[target_label]

        if erase_value == "zero":
            erase_value = 0
        else:
            assert self.model_type == "bert", "GPT models don't have an unk token"
            erase_value = word_embeddings_weights[self.unk_token]

        # modify the weights by subtracting the original prediction's word embedding
        # and adding the target embedding
        original_weight_values = []  # to reverse the action later
        for layer_idx, position in neurons:
            output_ff_weights = self._get_output_ff_layer(layer_idx)
            if self.model_type == "gpt2":
                # since gpt2 uses a conv1d layer instead of a linear layer in the ff block, the weights are in a different format
                original_weight_values.append(
                    output_ff_weights[position, :].detach().clone()
                )
            else:
                original_weight_values.append(
                    output_ff_weights[:, position].detach().clone()
                )
            if mode == "edit":
                if self.model_type == "gpt2":
                    output_ff_weights[position, :] -= original_prediction_embedding * 2
                    output_ff_weights[position, :] += target_embedding * 2
                else:
                    output_ff_weights[:, position] -= original_prediction_embedding * 2
                    output_ff_weights[:, position] += target_embedding * 2
            else:
                if self.model_type == "gpt2":
                    output_ff_weights[position, :] = erase_value
                else:
                    output_ff_weights[:, position] = erase_value

        # get the probabilities of the target being generated + the argmax / greedy completion after modifying the weights
        (
            new_gt_prob,
            new_gt_ranks,
            new_argmax_prob,
            new_argmax_completion_str,
            new_argmax_tokens,
        ) = self._generate(prompt, target)
        if not quiet:
            logger.info(
                f"\nAfter modification - groundtruth probability: {new_gt_prob}\n"
                f"groundtruth rank: {new_gt_ranks[0][1]}\n"
                f"Argmax completion: `{new_argmax_completion_str}`\n"
                f"Argmax prob: {new_argmax_prob}"
            )
        results_dict["after"] = {
            "gt_prob": new_gt_prob,
            "gt_rank": new_gt_ranks[0][1],
            "argmax_completion": new_argmax_completion_str,
            "argmax_prob": new_argmax_prob,
        }

        def unpatch_fn():
            # reverse modified weights
            for idx, (layer_idx, position) in enumerate(neurons):
                output_ff_weights = self._get_output_ff_layer(layer_idx)
                if self.model_type == "gpt2":
                    output_ff_weights[position, :] = original_weight_values[idx]
                else:
                    output_ff_weights[:, position] = original_weight_values[idx]

        if undo_modification:
            unpatch_fn()
            unpatch_fn = lambda *args: args

        return results_dict, unpatch_fn

    def edit_knowledge(
        self,
        prompt: str,
        target: str,
        neurons: List[List[int]],
        undo_modification: bool = True,
        quiet: bool = False,
    ) -> Tuple[dict, Callable]:
        return self.modify_weights(
            prompt=prompt,
            neurons=neurons,
            target=target,
            mode="edit",
            undo_modification=undo_modification,
            quiet=quiet,
        )

    def erase_knowledge(
        self,
        prompt: str,
        neurons: List[List[int]],
        erase_value: str = "zero",
        target: Optional[str] = None,
        undo_modification: bool = True,
        quiet: bool = False,
    ) -> Tuple[dict, Callable]:
        return self.modify_weights(
            prompt=prompt,
            neurons=neurons,
            target=target,
            mode="erase",
            erase_value=erase_value,
            undo_modification=undo_modification,
            quiet=quiet,
        )
