# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import copy
import importlib.util
import logging
import math
import os
import sys
import warnings
from collections import defaultdict
from itertools import accumulate
from typing import Callable, Dict, List, Optional
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
from fairseq.modules import gelu, gelu_accurate
from fairseq.modules.multihead_attention import MultiheadAttention
from torch import Tensor
import pdb

logger = logging.getLogger(__name__)


def load_ensemble_for_inference(filenames, task, model_arg_overrides=None):
    from fairseq import checkpoint_utils

    deprecation_warning(
        "utils.load_ensemble_for_inference is deprecated. "
        "Please use checkpoint_utils.load_model_ensemble instead."
    )
    return checkpoint_utils.load_model_ensemble(
        filenames, arg_overrides=model_arg_overrides, task=task
    )


def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)


def move_to_cuda(sample):
    def _move_to_cuda(tensor):
        return tensor.cuda()

    return apply_to_sample(_move_to_cuda, sample)


def get_incremental_state(
    module: MultiheadAttention,
    incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
    key: str,
) -> Optional[Dict[str, Optional[Tensor]]]:
    """Helper for getting incremental state for an nn.Module."""
    return module.get_incremental_state(incremental_state, key)


def set_incremental_state(
    module: MultiheadAttention,
    incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
    key: str,
    value: Dict[str, Optional[Tensor]],
) -> Optional[Dict[str, Dict[str, Optional[Tensor]]]]:
    """Helper for setting incremental state for an nn.Module."""
    if incremental_state is not None:
        result = module.set_incremental_state(incremental_state, key, value)
        if result is not None:
            incremental_state = result
    return incremental_state


def load_align_dict(replace_unk):
    if replace_unk is None:
        align_dict = None
    elif isinstance(replace_unk, str) and len(replace_unk) > 0:
        # Load alignment dictionary for unknown word replacement if it was passed as an argument.
        align_dict = {}
        with open(replace_unk, "r") as f:
            for line in f:
                cols = line.split()
                align_dict[cols[0]] = cols[1]
    else:
        # No alignment dictionary provided but we still want to perform unknown word replacement by copying the
        # original source word.
        align_dict = {}
    return align_dict


def print_embed_overlap(embed_dict, vocab_dict):
    embed_keys = set(embed_dict.keys())
    vocab_keys = set(vocab_dict.symbols)
    overlap = len(embed_keys & vocab_keys)
    logger.info("found {}/{} types in embedding file".format(overlap, len(vocab_dict)))


def parse_embedding(embed_path):
    """Parse embedding text file into a dictionary of word and embedding tensors.

    The first line can have vocabulary size and dimension. The following lines
    should contain word and embedding separated by spaces.

    Example:
        2 5
        the -0.0230 -0.0264  0.0287  0.0171  0.1403
        at -0.0395 -0.1286  0.0275  0.0254 -0.0932
    """
    embed_dict = {}
    with open(embed_path) as f_embed:
        next(f_embed)  # skip header
        for line in f_embed:
            pieces = line.rstrip().split(" ")
            embed_dict[pieces[0]] = torch.Tensor(
                [float(weight) for weight in pieces[1:]]
            )
    return embed_dict


def load_embedding(embed_dict, vocab, embedding):
    for idx in range(len(vocab)):
        token = vocab[idx]
        if token in embed_dict:
            embedding.weight.data[idx] = embed_dict[token]
    return embedding


def replace_unk(hypo_str, src_str, alignment, align_dict, unk):
    from fairseq import tokenizer

    # Tokens are strings here
    hypo_tokens = tokenizer.tokenize_line(hypo_str)
    # TODO: Very rare cases where the replacement is '<eos>' should be handled gracefully
    src_tokens = tokenizer.tokenize_line(src_str) + ["<eos>"]
    for i, ht in enumerate(hypo_tokens):
        if ht == unk:
            src_token = src_tokens[alignment[i]]
            # Either take the corresponding value in the aligned dictionary or just copy the original value.
            hypo_tokens[i] = align_dict.get(src_token, src_token)
    return " ".join(hypo_tokens)


def post_process_prediction(
    hypo_tokens, src_str, alignment, align_dict, tgt_dict, remove_bpe=None
):
    hypo_str = tgt_dict.string(hypo_tokens, remove_bpe)
    if align_dict is not None:
        hypo_str = replace_unk(
            hypo_str, src_str, alignment, align_dict, tgt_dict.unk_string()
        )
    if align_dict is not None or remove_bpe is not None:
        # Convert back to tokens for evaluating with unk replacement or without BPE
        # Note that the dictionary can be modified inside the method.
        hypo_tokens = tgt_dict.encode_line(hypo_str, add_if_not_exist=True)
    return hypo_tokens, hypo_str, alignment


def make_positions(tensor, padding_idx: int, onnx_trace: bool = False):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx


def strip_pad(tensor, pad):
    return tensor[tensor.ne(pad)]


def buffered_arange(max):
    if not hasattr(buffered_arange, "buf"):
        buffered_arange.buf = torch.LongTensor()
    if max > buffered_arange.buf.numel():
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]


def convert_padding_direction(
    src_tokens, padding_idx, right_to_left=False, left_to_right=False
):
    assert right_to_left ^ left_to_right
    pad_mask = src_tokens.eq(padding_idx)
    if not pad_mask.any():
        # no padding, return early
        return src_tokens
    if left_to_right and not pad_mask[:, 0].any():
        # already right padded
        return src_tokens
    if right_to_left and not pad_mask[:, -1].any():
        # already left padded
        return src_tokens
    max_len = src_tokens.size(1)
    range = buffered_arange(max_len).type_as(src_tokens).expand_as(src_tokens)
    num_pads = pad_mask.long().sum(dim=1, keepdim=True)
    if right_to_left:
        index = torch.remainder(range - num_pads, max_len)
    else:
        index = torch.remainder(range + num_pads, max_len)
    return src_tokens.gather(1, index)


def item(tensor):
    if hasattr(tensor, "item"):
        return tensor.item()
    if hasattr(tensor, "__getitem__"):
        return tensor[0]
    return tensor


def clip_grad_norm_(params, max_norm):
    params = list(params)
    if len(params) == 1:
        p = params[0]
        grad_norm = torch.norm(p)
        if grad_norm > max_norm > 0:
            clip_coef = max_norm / (grad_norm + 1e-6)
            p.mul_(clip_coef)
        return grad_norm
    elif max_norm > 0:
        return torch.nn.utils.clip_grad_norm_(params, max_norm)
    else:
        return torch.sqrt(
            sum(p.grad.data.norm() ** 2 for p in params if p.grad is not None)
        )


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)


def _match_types(arg1, arg2):
    """Convert the numerical argument to the same type as the other argument"""

    def upgrade(arg_number, arg_structure):
        if isinstance(arg_structure, tuple):
            return (arg_number, arg_number)
        elif isinstance(arg_structure, dict):
            arg = copy.deepcopy(arg_structure)
            for k in arg:
                arg[k] = upgrade(arg_number, arg_structure[k])
            return arg
        else:
            return arg_number

    if isinstance(arg1, float) or isinstance(arg1, int):
        return upgrade(arg1, arg2), arg2
    elif isinstance(arg2, float) or isinstance(arg2, int):
        return arg1, upgrade(arg2, arg1)

    return arg1, arg2


def resolve_max_positions(*args):
    """Resolve max position constraints from multiple sources."""

    def map_value_update(d1, d2):
        updated_value = copy.deepcopy(d1)
        for key in d2:
            if key not in updated_value:
                updated_value[key] = d2[key]
            else:
                updated_value[key] = min(d1[key], d2[key])
        return updated_value

    def nullsafe_min(l):
        minim = None
        for item in l:
            if minim is None:
                minim = item
            elif item is not None and item < minim:
                minim = item
        return minim

    max_positions = None
    for arg in args:
        if max_positions is None:
            max_positions = arg
        elif arg is not None:
            max_positions, arg = _match_types(max_positions, arg)
            if isinstance(arg, float) or isinstance(arg, int):
                max_positions = min(max_positions, arg)
            elif isinstance(arg, dict):
                max_positions = map_value_update(max_positions, arg)
            else:
                max_positions = tuple(map(nullsafe_min, zip(max_positions, arg)))

    return max_positions


def import_user_module(args):
    module_path = getattr(args, "user_dir", None)
    if module_path is not None:
        module_path = os.path.abspath(args.user_dir)
        if not os.path.exists(module_path):
            fairseq_rel_path = os.path.join(
                os.path.dirname(__file__), "..", args.user_dir
            )
            if os.path.exists(fairseq_rel_path):
                module_path = fairseq_rel_path
        module_parent, module_name = os.path.split(module_path)

        if module_name not in sys.modules:
            sys.path.insert(0, module_parent)
            importlib.import_module(module_name)
            sys.path.pop(0)


def softmax(x, dim: int, onnx_trace: bool = False):
    if onnx_trace:
        return F.softmax(x.float(), dim=dim)
    else:
        return F.softmax(x, dim=dim, dtype=torch.float32)


def log_softmax(x, dim: int, onnx_trace: bool = False):
    if onnx_trace:
        return F.log_softmax(x.float(), dim=dim)
    else:
        return F.log_softmax(x, dim=dim, dtype=torch.float32)


def get_perplexity(loss, round=2, base=2):
    if loss is None:
        return 0.
    return np.round(np.power(base, loss), round)


def deprecation_warning(message, stacklevel=3):
    # don't use DeprecationWarning, since it's ignored by default
    warnings.warn(message, stacklevel=stacklevel)


def get_activation_fn(activation: str) -> Callable:
    """ Returns the activation function corresponding to `activation` """
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return gelu
    elif activation == "gelu_fast":
        deprecation_warning(
            "--activation-fn=gelu_fast has been renamed to gelu_accurate"
        )
        return gelu_accurate
    elif activation == "gelu_accurate":
        return gelu_accurate
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))


def get_available_activation_fns() -> List:
    return [
        "relu",
        "gelu",
        "gelu_fast",  # deprecated
        "gelu_accurate",
        "tanh",
        "linear",
    ]


@contextlib.contextmanager
def eval(model):
    is_training = model.training
    model.eval()
    yield
    model.train(is_training)


def has_parameters(module):
    try:
        next(module.parameters())
        return True
    except StopIteration:
        return False


def set_torch_seed(seed):
    # Set seed based on args.seed and the update number so that we get
    # reproducible results when resuming from checkpoints
    assert isinstance(seed, int)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def parse_alignment(line):
    """
    Parses a single line from the alingment file.

    Args:
        line (str): String containing the alignment of the format:
            <src_idx_1>-<tgt_idx_1> <src_idx_2>-<tgt_idx_2> ..
            <src_idx_m>-<tgt_idx_m>. All indices are 0 indexed.

    Returns:
        torch.IntTensor: packed alignments of shape (2 * m).
    """
    alignments = line.strip().split()
    parsed_alignment = torch.IntTensor(2 * len(alignments))
    for idx, alignment in enumerate(alignments):
        src_idx, tgt_idx = alignment.split("-")
        parsed_alignment[2 * idx] = int(src_idx)
        parsed_alignment[2 * idx + 1] = int(tgt_idx)
    return parsed_alignment


def get_token_to_word_mapping(tokens, exclude_list):
    n = len(tokens)
    word_start = [int(token not in exclude_list) for token in tokens]
    word_idx = list(accumulate(word_start))
    token_to_word = {i: word_idx[i] for i in range(n)}
    return token_to_word


def extract_hard_alignment(attn, src_sent, tgt_sent, pad, eos):
    tgt_valid = ((tgt_sent != pad) & (tgt_sent != eos)).nonzero().squeeze(dim=-1)
    src_invalid = ((src_sent == pad) | (src_sent == eos)).nonzero().squeeze(dim=-1)
    src_token_to_word = get_token_to_word_mapping(src_sent, [eos, pad])
    tgt_token_to_word = get_token_to_word_mapping(tgt_sent, [eos, pad])
    alignment = []
    if len(tgt_valid) != 0 and len(src_invalid) < len(src_sent):
        attn_valid = attn[tgt_valid]
        attn_valid[:, src_invalid] = float("-inf")
        _, src_indices = attn_valid.max(dim=1)
        for tgt_idx, src_idx in zip(tgt_valid, src_indices):
            alignment.append(
                (
                    src_token_to_word[src_idx.item()] - 1,
                    tgt_token_to_word[tgt_idx.item()] - 1,
                )
            )
    return alignment


def new_arange(x, *size):
    """
    Return a Tensor of `size` filled with a range function on the device of x.
    If size is empty, using the size of the variable x.
    """
    if len(size) == 0:
        size = x.size()
    return torch.arange(size[-1], device=x.device).expand(*size).contiguous()

def n_gram(list_words, n = 2):
    set_gram= set()
    count = {}
    l = len(list_words)
    if n == 1:
        for i in range(l):
            word = list_words[i]
            if word not in set_gram:
                set_gram.add(word)
                count[word] = 1
            else:
                set_gram.add((word,count[word]))
                count[word] += 1
    elif n == 2:
        for i in range(l-1):
            word = (list_words[i],list_words[i+1])
            if word not in set_gram:
                set_gram.add(word)
                count[word] = 1
            else:
                set_gram.add((word,count[word]))
                count[word] += 1
    elif n == 3:
        for i in range(l-2):
            word = (list_words[i],list_words[i+1], list_words[i+2])
            if word not in set_gram:
                set_gram.add(word)
                count[word] = 1
            else:
                set_gram.add((word,count[word]))
                count[word] += 1
    elif n == 4:
        for i in range(l-3):
            word = (list_words[i],list_words[i+1], list_words[i+2], list_words[i+3])
            if word not in set_gram:
                set_gram.add(word)
                count[word] = 1
            else:
                set_gram.add((word,count[word]))
                count[word] += 1
    else:
        exit()
    
    return set_gram

def n_grams(list_words):
    set_1gram, set_2gram, set_3gram, set_4gram = set(), set(), set(), set()
    count = {}
    l = len(list_words)
    for i in range(l):
        word = list_words[i]
        if word not in set_1gram:
            set_1gram.add(word)
            count[word] = 1
        else:
            set_1gram.add((word,count[word]))
            count[word] += 1
    count = {}

    for i in range(l-1):
        word = (list_words[i],list_words[i+1])
        if word not in set_2gram:
            set_2gram.add(word)
            count[word] = 1
        else:
            set_2gram.add((word,count[word]))
            count[word] += 1

    count = {}

    for i in range(l-2):
        word = (list_words[i],list_words[i+1], list_words[i+2])
        if word not in set_3gram:
            set_3gram.add(word)
            count[word] = 1
        else:
            set_3gram.add((word,count[word]))
            count[word] += 1
    count = {}

    for i in range(l-3):
        word = (list_words[i],list_words[i+1], list_words[i+2], list_words[i+3])
        if word not in set_4gram:
            set_4gram.add(word)
            count[word] = 1
        else:
            set_4gram.add((word,count[word]))
            count[word] += 1

    return set_1gram, set_2gram, set_3gram, set_4gram

def my_sentence_rouge(references, hypothesis):
    reference = references[0]
    ref_grams = n_gram(reference, n=2)
    hyp_grams = n_gram(hypothesis, n=2)
    match_grams = hyp_grams.intersection(ref_grams)
    ref_count = len(ref_grams)
    hyp_count = len(hyp_grams)
    match_count = len(match_grams)
    rouge = float(match_count) / float(max(ref_count,hyp_count))
    return rouge

def my_sentence_gleu(references, hypothesis):
    reference = references[0]
    ref_grams = n_grams(reference)
    hyp_grams = n_grams(hypothesis)
    match_grams = [x.intersection(y) for (x,y) in zip(ref_grams, hyp_grams)]
    ref_count = sum([len(x) for x in ref_grams])
    hyp_count = sum([len(x) for x in hyp_grams])
    match_count = sum([len(x) for x in match_grams])
    gleu = float(match_count) / float(max(ref_count,hyp_count))
    return gleu

def my_sentence_bleu(references, hypothesis):
    reference = references[0]
    ref_grams = n_grams(reference)
    hyp_grams = n_grams(hypothesis)
    match_grams = [x.intersection(y) for (x,y) in zip(ref_grams, hyp_grams)]
    hyp_count = [len(x) for x in hyp_grams]
    match_count = [len(x) for x in match_grams]
    p = [float(match_count[i])/(float(hyp_count[i]+1e-6)) for i in range(4)]
    bleu = (p[0] * p[1] * p[2] * p[3])**0.25
    return bleu

def shape(targets, target_lens):
    list_targets = []
    begin = 0
    end = 0
    for length in target_lens:
        end += length
        list_targets.append([str(index) for index in targets[begin:end]])
        begin += length
    return list_targets

def parallel_reward(inputs):
    (sample_idx, list_samples, list_targets, count, target_lens) = inputs
    l_samples = shape(sample_idx, target_lens)
    rewards = []
    for j in range(count):
        for k in range(len(l_samples[j])):
            t = l_samples[j][k]
            l_samples[j][k] = list_samples[j][k]
            reward = my_sentence_rouge([l_samples[j]], list_targets[j])
            l_samples[j][k] = t
            rewards.append(reward)
    return rewards

def parallel_reward_tra(inputs):
    (sample_idx, all_list_samples, list_targets, count, target_lens) = inputs
    l_samples = shape(sample_idx, target_lens)
    length = len(all_list_samples)
    rewards = []
    for i in range(length+1):
        rewards.append([])
    for j in range(count):
        for k in range(len(l_samples[j])):
            for i in range(length):
                t = l_samples[j][k]
                l_samples[j][k] = all_list_samples[i][j][k]
                reward = my_sentence_rouge([l_samples[j]], list_targets[j])
                l_samples[j][k] = t
                rewards[i].append(reward)
            t = l_samples[j][k]
            l_samples[j][k] = 3
            reward = my_sentence_rouge([l_samples[j]], list_targets[j])
            l_samples[j][k] = t
            rewards[length].append(reward)
    return rewards

def twogram_match( targets, target_lens, probs):

    batch_size = len(target_lens)
    batch_match = []
    end = 0
    for i in range(batch_size):
        begin = end
        end = begin + target_lens[i]
        curr_tar = targets[i]

        if target_lens[i] < 2:
            batch_match.append((0,1e-5))
            continue

        two_grams = Counter()
        for j in range(len(curr_tar) - 1):
            two_grams[(curr_tar[j], curr_tar[j+1])] += 1

        gram_1, gram_2 = [], []
        gram_count = []
        for two_gram in two_grams:
            gram_1.append(int(two_gram[0]))
            gram_2.append(int(two_gram[1]))
            gram_count.append(two_grams[two_gram])

        match_gram_1 = probs[begin:end-1, gram_1]
        match_gram_2 = probs[begin+1:end, gram_2]
        match_gram = match_gram_1 * match_gram_2
        match_gram = torch.sum(match_gram, dim = 0).view(-1,1)

        gram_count = torch.HalfTensor(gram_count).cuda(probs.get_device()).view(-1,1)
        match_gram = torch.min(torch.cat([match_gram,gram_count],dim = -1), dim = -1)[0]
        match_gram = torch.sum(match_gram)
        batch_match.append((match_gram, target_lens[i] - 1))

    return batch_match

def threegram_match(targets, target_lens, probs):

    batch_size = len(target_lens)
    batch_match = []
    end = 0
    for i in range(batch_size):
        begin = end
        end = begin + target_lens[i]
        curr_tar = targets[i]

        if target_lens[i] < 3:
            batch_match.append((0,1e-5))
            continue

        three_grams = Counter()
        for j in range(len(curr_tar) - 2):
            three_grams[(curr_tar[j], curr_tar[j+1], curr_tar[j+2])] += 1

        gram_1, gram_2, gram_3 = [], [], []
        gram_count = []
        for three_gram in three_grams:
            gram_1.append(int(three_gram[0]))
            gram_2.append(int(three_gram[1]))
            gram_3.append(int(three_gram[2]))
            gram_count.append(three_grams[three_gram])

        match_gram_1 = probs[begin:end-2, gram_1]
        match_gram_2 = probs[begin+1:end-1, gram_2]
        match_gram_3 = probs[begin+2:end, gram_3]
        match_gram = match_gram_1 * match_gram_2 * match_gram_3
        match_gram = torch.sum(match_gram, dim = 0).view(-1,1)

        gram_count = torch.HalfTensor(gram_count).cuda(probs.get_device()).view(-1,1)
        match_gram = torch.min(torch.cat([match_gram,gram_count],dim = -1), dim = -1)[0]
        match_gram = torch.sum(match_gram)
        batch_match.append((match_gram, target_lens[i] - 2))

    return batch_match

def fourgram_match(targets, target_lens, probs):

    batch_size = len(target_lens)
    batch_match = []
    end = 0
    for i in range(batch_size):
        begin = end
        end = begin + target_lens[i]
        curr_tar = targets[i]
        if target_lens[i] < 4:
            batch_match.append((torch.Tensor([0]).cuda(probs.get_device()),1e-5))
            continue

        four_grams = Counter()
        for j in range(len(curr_tar) - 3):
            four_grams[(curr_tar[j], curr_tar[j+1] , curr_tar[j+2],  curr_tar[j+3])] += 1

        gram_1, gram_2, gram_3, gram_4 = [], [], [], []
        gram_count = []
        for four_gram in four_grams:
            gram_1.append(int(four_gram[0]))
            gram_2.append(int(four_gram[1]))
            gram_3.append(int(four_gram[2]))
            gram_4.append(int(four_gram[3]))
            gram_count.append(four_grams[four_gram])

        match_gram_1 = probs[begin:end-3, gram_1]
        match_gram_2 = probs[begin+1:end-2, gram_2]
        match_gram_3 = probs[begin+2:end-1, gram_3]
        match_gram_4 = probs[begin+3:end, gram_4]
        match_gram = match_gram_1 * match_gram_2 * match_gram_3 * match_gram_4
        match_gram = torch.sum(match_gram, dim = 0).view(-1,1)

        gram_count = torch.HalfTensor(gram_count).cuda(probs.get_device()).view(-1,1)
        match_gram = torch.min(torch.cat([match_gram,gram_count],dim = -1), dim = -1)[0]
        match_gram = torch.sum(match_gram)
        batch_match.append((match_gram, target_lens[i] - 3))

    return batch_match

