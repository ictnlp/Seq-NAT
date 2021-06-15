# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch.nn.functional as F
import torch
from torch import Tensor

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

@register_criterion("nat_loss")
class LabelSmoothedDualImitationCriterion(FairseqCriterion):
    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument(
            '--label-smoothing',
            default=0.,
            type=float,
            metavar='D',
            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def _compute_loss(
        self, outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=1.0
    ):
        """
            outputs: batch x len x d_model
            targets: batch x len
            masks:   batch x len

            policy_logprob: if there is some policy
                depends on the likelihood score as rewards.
        """

        def mean_ds(x: Tensor, dim=None) -> Tensor:
            return (
                x.float().mean().type_as(x)
                if dim is None
                else x.float().mean(dim).type_as(x)
            )
        if masks is not None:
            outputs, targets = outputs[masks], targets[masks]

        if masks is not None and not masks.any():
            nll_loss = torch.tensor(0)
            loss = nll_loss
        else:
            logits = F.log_softmax(outputs, dim=-1)
            if targets.dim() == 1:
                losses = F.nll_loss(logits, targets.to(logits.device), reduction='none')

            else:  # soft-labels
                losses = F.kl_div(logits, targets.to(logits.device), reduction='none')
                losses = losses.sum(-1)

            nll_loss = mean_ds(losses)
            if label_smoothing > 0:
                loss = nll_loss * (
                    1 - label_smoothing) - mean_ds(logits) * label_smoothing
            else:
                loss = nll_loss

        loss = loss * factor
        return {"name": name, "loss": loss, "nll_loss": nll_loss, "factor": factor}

    def _custom_loss(self, loss, name="loss", factor=1.0):
        return {"name": name, "loss": loss, "factor": factor}

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        nsentences, ntokens = sample["nsentences"], sample["ntokens"]

        # B x T
        src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )
        tgt_tokens, prev_output_tokens = sample["target"], sample["prev_target"]

        outputs = model(src_tokens, src_lengths, prev_output_tokens, tgt_tokens)
        losses, nll_loss = [], []

        for obj in outputs:
            if outputs[obj].get("loss", None) is None:
                _losses = self._compute_loss(
                    outputs[obj].get("out"),
                    outputs[obj].get("tgt"),
                    outputs[obj].get("mask", None),
                    outputs[obj].get("ls", 0.0),
                    name=obj + '-loss',
                    factor=outputs[obj].get("factor", 1.0)
                )
            else:
                _losses = self._custom_loss(
                    outputs[obj].get("loss"),
                    name=obj + '-loss',
                    factor=outputs[obj].get("factor", 1.0)
                )

            losses += [_losses]
            if outputs[obj].get("nll_loss", False):
                nll_loss += [_losses.get("nll_loss", 0.0)]

        loss = sum(l["loss"] for l in losses)
        nll_loss = sum(l for l in nll_loss) if len(nll_loss) > 0 \
            else loss.new_tensor(0)

        # NOTE:
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
        sample_size = 1
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }

        for l in losses:
            logging_output[l["name"]] = (
                utils.item(l["loss"].data / l["factor"])
                if reduce
                else l[["loss"]].data / l["factor"]
            )

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        sample_size = utils.item(sum(log.get("sample_size", 0) for log in logging_outputs))
        loss = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        nll_loss = utils.item(sum(log.get("nll_loss", 0) for log in logging_outputs))

        metrics.log_scalar('loss', loss / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss / sample_size / math.log(2), sample_size, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['loss'].avg))

        for key in logging_outputs[0]:
            if key[-5:] == "-loss":
                val = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(
                    key[:-5],
                    val / sample_size / math.log(2) if sample_size > 0 else 0.0,
                    sample_size,
                    round=3,
                )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


@register_criterion("nat_seq_loss")
class SeqCriterion(LabelSmoothedDualImitationCriterion):
    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument(
            '--label-smoothing',
            default=0.,
            type=float,
            metavar='D',
            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument(
            '--use-ngram',
            action="store_true")
        parser.add_argument(
            '--use-rl',
            action="store_true")
        parser.add_argument(
            '--rl-type',
            type=str,
            choices=["base","topk","traverse"])
        parser.add_argument(
            '--n',
            default=2,
            type=int)
        parser.add_argument(
            '--topk',
            default=5,
            type=int)
        parser.add_argument(
            '--p',
            default=1,
            type=float)
        # fmt: on
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        nsentences, ntokens = sample["nsentences"], sample["ntokens"]

        # B x T
        src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )
        tgt_tokens, prev_output_tokens = sample["target"], sample["prev_target"]
        outputs = model(src_tokens, src_lengths, prev_output_tokens, tgt_tokens)
        losses, nll_loss = [], []
        for obj in outputs:
            if outputs[obj].get("loss", None) is None:
                _losses = self._compute_loss(
                    outputs[obj].get("out"),
                    outputs[obj].get("tgt"),
                    outputs[obj].get("mask", None),
                    outputs[obj].get("ls", 0.0),
                    name=obj + '-wordloss',
                    factor=outputs[obj].get("factor", 1.0)
                )
                if obj is 'word_ins':
                    _losses["loss"] = 0
                    if self.args.use_ngram:
                        gram_losses = self._compute_gram_loss(
                            outputs[obj].get("out"),
                            outputs[obj].get("tgt"),
                            outputs[obj].get("mask", None),
                        )
                        _losses["loss"] += gram_losses
                    if self.args.use_rl:
                        rl_losses = self._compute_reward_loss(
                            outputs[obj].get("out"),
                            outputs[obj].get("tgt"),
                            outputs[obj].get("mask", None),
                        )
                        _losses["loss"] += rl_losses
            losses += [_losses]
            if outputs[obj].get("nll_loss", False):
                nll_loss += [_losses.get("nll_loss", 0.0)]

        loss = sum(l["loss"] for l in losses)
        nll_loss = sum(l for l in nll_loss) if len(nll_loss) > 0 \
            else loss.new_tensor(0)

        # NOTE:
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
        sample_size = 1
        logging_output = {
            "loss": utils.item(loss.data) if reduce else loss.data,
            "nll_loss": utils.item(nll_loss.data) if reduce else nll_loss.data,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }
        for l in losses:
            logging_output[l["name"]] = (
                utils.item(l["loss"].data / l["factor"])
                if reduce
                else l[["loss"]].data / l["factor"]
            )

        return loss, sample_size, logging_output

    def _compute_gram_loss(self, outputs, targets, masks=None):
        """
            outputs: batch x len x d_model
            targets: batch x len
            masks:   batch x len
        """
        if self.args.n == 1:
            loss = self._compute_bow_loss(outputs, targets, masks)
            return loss

        batch_size, length = targets.size()
        if masks is not None:
            outputs, targets = outputs[masks], targets[masks]
        else:
            masks = torch.ones(batch_size, length)

        probs = F.softmax(outputs)
        target_lens = torch.sum(masks, dim = -1).long().tolist()
        targets = targets.data.tolist()
        targets = utils.shape(targets, target_lens)

        matchs = []
        if self.args.n == 2:
            gram_match = utils.twogram_match(targets, target_lens, probs)
        elif self.args.n == 3:
            gram_match = utils.threegram_match(targets, target_lens, probs)
        elif self.args.n == 4:
            gram_match = utils.fourgram_match(targets, target_lens, probs)
        else:
            raise NotImplementedError

        for i in range(batch_size):
            matchs.append(gram_match[i][0]/gram_match[i][1])
        loss = -1 * sum(matchs).div(batch_size)
        return loss

    def _compute_reward_loss(self, outputs, targets, masks=None):
        if self.args.rl_type == "base":
            return self._compute_reward_loss_rlbase(outputs, targets, masks)
        elif self.args.rl_type == "topk":
            return self._compute_reward_loss_rltopk(outputs, targets, masks)
        elif self.args.rl_type == "traverse":
            return self._compute_reward_loss_rltraverse(outputs, targets, masks)
        else:
            raise NotImplementedError

    def _compute_bow_loss(self, outputs, targets, masks=None):
        """
            outputs: batch x len x d_model
            targets: batch x len
            masks:   batch x len
        """
        batch_size, length, vocab_size = outputs.size()
        if masks is None:
            masks = torch.ones(batch_size, length)
        masks = masks.float().view(batch_size, length, 1)
        
        probs = F.softmax(outputs, dim = -1)
        bow = torch.sum(probs, dim = 1)
        ref_bow = torch.zeros(batch_size, vocab_size).cuda(probs.get_device())
        ones = torch.ones(batch_size, vocab_size).cuda(probs.get_device())
        ref_bow.scatter_add_(-1, targets, ones)
        loss = torch.mean(torch.norm(bow-ref_bow,p=self.args.p,dim=-1)).div(length)
        #loss = 1 - torch.mean(torch.cosine_similarity(bow, ref_bow))
        return loss

    def compute_step_reward(self, sample_times, workers, sample_index, sample_prob, targets, target_lens):

        list_targets = utils.shape(targets,target_lens)
        list_samples = utils.shape(sample_index,target_lens)
        count = len(list_samples)
        rewards = []
        sample_idxs = [torch.multinomial(sample_prob,1).data.view(-1).tolist() for i in range(sample_times)]
        inputs = [(sample_idxs[i], list_samples, list_targets, count, target_lens) for i in range(sample_times)]
        pool = ProcessPoolExecutor(max_workers=workers)
        rewards = list(pool.map(utils.parallel_reward, inputs))
        rewards = torch.Tensor(rewards).cuda(sample_prob.get_device())
        rewards = torch.mean(rewards,dim = 0)
        
        return rewards

    def compute_traverse_step_reward(self, sample_times, workers, all_sample_index, sample_prob, targets, target_lens):

        list_targets = utils.shape(targets,target_lens)
        all_list_samples = [utils.shape(sample_index,target_lens) for sample_index in all_sample_index]
        count = len(all_list_samples[0])
        rewards = []
        sample_idxs = [torch.multinomial(sample_prob,1).data.view(-1).tolist() for i in range(sample_times)]
        inputs = [(sample_idxs[i], all_list_samples, list_targets, count, target_lens) for i in range(sample_times)]
        pool = ProcessPoolExecutor(max_workers=workers)
        rewards = list(pool.map(utils.parallel_reward_tra, inputs))
        rewards = torch.Tensor(rewards).cuda(sample_prob.get_device())
        rewards = torch.mean(rewards,dim = 0)
        
        return rewards

    def compute_sentence_reward(self, sample_index, sample_prob, targets, target_lens):

        list_targets = utils.shape(targets,target_lens)
        list_samples = utils.shape(sample_index,target_lens)
        count = len(list_samples)
        rewards = []
        for i in range(count):
            reward = utils.my_sentence_rouge([list_samples[i]], list_targets[i])
            for k in range(len(list_samples[i])):
                rewards.append(reward)
        rewards = torch.Tensor(rewards).cuda(sample_prob.get_device())
        return rewards

    def _compute_reward_loss_rltopk(self, outputs, targets, masks=None):

        outputs, targets = outputs[masks], targets[masks]
        probs = F.softmax(outputs, dim = -1)
        target_lens = torch.sum(masks, dim = -1).long().tolist()
        targets = targets.data.tolist()

        top_probs, top_index = torch.topk(probs, self.args.topk, dim = -1)
        weight = torch.sum(top_probs, dim = -1).detach()

        res_probs = torch.zeros(probs.size()).cuda(probs.get_device())
        res_probs.data.copy_(probs.data)
        res_probs.scatter_add_(1, top_index, -1 * top_probs)
        sample_index = torch.multinomial(res_probs,1)
        del res_probs
        sample_prob = torch.gather(probs, -1, sample_index)
        sample_index = sample_index.data.view(-1).tolist()
        top_index = top_index.t().data.tolist()

        if self.args.topk != 0:
            rewards = self.compute_traverse_step_reward(10, 10, top_index, probs, targets, target_lens)
            rewards = rewards[0:self.args.topk]
            rewards = torch.t(rewards)
            loss_traverse = -1 * torch.sum(top_probs * rewards)
        else:
            loss_traverse = 0

        reward = self.compute_step_reward(10, 10, sample_index, probs, targets, target_lens)
        loss_sample = torch.sum((-1 * (1-weight) * torch.log(sample_prob).view(-1) * reward),dim = 0)

        loss = (loss_sample + loss_traverse).div(len(targets))
        return loss

    def _compute_reward_loss_rlbase(self, outputs, targets, masks=None):

        outputs, targets = outputs[masks], targets[masks]
        probs = F.softmax(outputs)
        target_lens = torch.sum(masks, dim = -1).long().tolist()
        targets = targets.data.tolist()

        sample_index = torch.multinomial(probs,1)
        sample_prob = torch.gather(probs, -1, sample_index)
        sample_index = sample_index.data.view(-1).tolist()
        reward = self.compute_sentence_reward(sample_index, probs, targets, target_lens)        
        loss_sample = torch.sum((-1 * torch.log(sample_prob).view(-1) * reward),dim = 0)
        loss = loss_sample.div(len(targets))
        
        return loss

    def _compute_reward_loss_rltraverse(self, outputs, targets, masks=None):

        batch_size, length, vocab_size = outputs.size()
        outputs = outputs.view(batch_size*length,vocab_size)
        targets = targets.view(-1)
        probs = F.softmax(outputs, dim = -1)
        target_lens = [length] * batch_size
        targets = targets.data.tolist()

        rewards = torch.zeros(batch_size * length, vocab_size).cuda(probs.get_device())

        ref_words = []
        for i in range(length):
            ref_words.append([])
            for j in range(batch_size*length):
                ref_words[i].append(0)

        for i in range(length):
            for j in range(batch_size):
                ref_words[i][j*length: (j+1)*length] = [targets[j*length + i]] * length
        ref_rewards = self.compute_traverse_step_reward(10, 10, ref_words, probs, targets, target_lens)
        ref_rewards = ref_rewards.view(length + 1, batch_size * length, 1)
        ref_words = torch.LongTensor(ref_words).view(length, batch_size * length, 1).cuda(probs.get_device())
        rewards += ref_rewards[length]

        for i in range(length):
            rewards.scatter_(1, ref_words[i], ref_rewards[i])
        loss = - torch.sum(torch.sum(probs*rewards, dim=-1) * masks.view(-1)).div(len(targets))

        return loss
