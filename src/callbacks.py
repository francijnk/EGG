import torch
import wandb
import argparse
import pandas as pd

from collections import OrderedDict, defaultdict

from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from src.eval import (
    message_entropy,
    relative_message_entropy,
    entropy_message_as_a_whole,
    mutual_info_sent_received,
    compute_mi,
    compute_max_rep,
    compute_top_sim,
)
from src.channels import Channel, NoChannel, DeletionChannel, ErasureChannel
from src.util import crop_messages
from src.interaction import Interaction

from egg.core.callbacks import Callback, CustomProgress
# from egg.zoo.language_bottleneck import intervention

from typing import Dict, Any


class EpochProgress(Progress):
    class CompletedColumn(ProgressColumn):
        def render(self, task):
            """Calculate common unit for completed and total."""
            download_status = f"{int(task.completed)}/{int(task.total)} ep"
            return Text(download_status, style="progress.download")

    class TransferSpeedColumn(ProgressColumn):
        """Renders human readable transfer speed."""

        def render(self, task):
            """Show data transfer speed."""
            speed = task.speed
            if speed is None:
                return Text("?", style="progress.data.speed")
            speed = f"{1 / speed:,.{2}f}"
            return Text(f"{speed} s/ep", style="progress.data.speed")

    def __init__(self, *args, **kwargs):
        super(EpochProgress, self).__init__(*args, **kwargs)


class CustomProgressBarLogger(Callback):
    """
    Displays a progress bar with information about the current epoch and the
    epoch progression. Based on the progress bar logger from
    egg/core/callbacks.py.
    """

    def __init__(
        self,
        opts: argparse.Namespace,
        train_data_len: int = 0,
        test_data_len: int = 0,
    ):
        """
        :param n_epochs: total number of epochs
        :param train_data_len: length of the dataset generation for training
        :param test_data_len: length of the dataset generation for testing
        """

        self.n_epochs = opts.n_epochs
        self.train_data_len = train_data_len
        self.test_data_len = test_data_len
        self.results_folder = opts.results_folder
        self.filename = opts.filename
        self.step = opts.validation_freq
        self.wandb = opts.wandb_project is not None
        self.current_step = 0
        self.display_nn = opts.channel != 'none'

        if self.wandb:
            if opts.wandb_group is None:
                group = opts.channel \
                    if opts.channel is not None and opts.error_prob > 0. \
                    else 'baseline'
            else:
                group = opts.wandb_group

            wandb.init(
                project=opts.wandb_project,
                group=group,
                id=opts.wandb_run_id,
                entity=opts.wandb_entity,
            )
            wandb.config.update(opts)

        self.history = defaultdict(lambda: defaultdict(list))
        self.hide_cols = [
            'entropy_max', 
            'entropy_inp', 'entropy_cat', 'entropy_shape', 'entropy_color',
            'entropy_xpos', 'entropy_ypos', 'entropy_rotation', 'entropy_attr',
            'redundancy_', 'mutual_info', 'variation', 'proficiency_rotation',
        ]

        self.progress = CustomProgress(
            TextColumn(
                "[bold]{task.fields[cur_epoch]}/{task.fields[n_epochs]} "
                "| [blue]{task.fields[mode]}",
                justify="right"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%", "•",
            CustomProgress.TransferSpeedColumn(), "•",
            TimeRemainingColumn(elapsed_when_finished=True),
            use_info_table=False)
        self.live = Live(self.generate_live_table())
        self.console = self.live.console

        self.live.start()

        self.test_p = self.progress.add_task(
            description="",
            mode="Validate",
            cur_epoch=0,
            n_epochs=self.n_epochs,
            start=False,
            visible=False,
            total=self.test_data_len)
        self.train_p = self.progress.add_task(
            description="",
            mode="Train",
            cur_epoch=0,
            n_epochs=self.n_epochs,
            start=False,
            visible=False,
            total=self.train_data_len)

        self.style = defaultdict(str)
        self.style.update({'train': 'grey58'})

    def build_od(self, logs, loss, epoch, phase):
        od = OrderedDict()
        od["epoch"] = epoch
        od['phase'] = phase
        od["loss"] = loss
        aux = logs.aux
        for k, v in aux.items():
            if isinstance(v, torch.Tensor):
                aux[k] = v.to(torch.float16).mean().item()
            elif isinstance(v, list):
                aux[k] = sum(v) / len(v)
        od.update(aux)
        return od

    def format_values(self, val):
        if (isinstance(val, tuple) and val[0] is None) \
                or val is None or val != val:
            return '–'
        elif isinstance(val, int):
            return str(val)
        elif not isinstance(val, str):
            return f'{val:.2f}'
        else:
            return val

    def get_row(self, od, noise=True, header=False):
        row = Table(
            expand=True, box=None, show_header=header,
            show_footer=False, padding=(0, 1), pad_edge=True)

        row_values = []
        for colname in ('epoch', 'phase', 'messages'):
            if colname == 'messages' and not self.display_nn:
                od['messages'] = 'baseline'
                continue
            elif colname == 'messages' and self.display_nn:
                od['messages'] = 'sent' if noise else 'received'
                row_values.append('sent' if noise else 'received')
            else:
                row_values.append(str(od[colname]))
            row.add_column(
                colname,
                justify='left',
                ratio=0.5 if colname != 'messages' else 1)

        for colname in od:
            if any(colname.startswith(c) for c in self.hide_cols) \
                    or colname.endswith('_nn') \
                    or colname in ('epoch', 'phase', 'messages'):
                continue
            if noise:
                value = self.format_values(od[colname])
            else:
                value = self.format_values(
                    od[f'{colname}_nn'] if f'{colname}_nn' in od
                    else od[colname])
            row_values.append(value)
            print_name = (
                colname
                .replace('entropy', 'H')
                .replace('redundancy', 'R')
                .replace('proficiency', 'C')
                .replace('msg_as_a_whole', 'whole_msg')
                .replace('actual_', '')
                .replace('_msg_', '_'))
            row.add_column(
                print_name,
                justify='right',
                ratio=1)
        if not header:
            row.add_row(*row_values, style=self.style[od['phase']])
        return row

    def log_to_wandb(self, data: Dict[str, Any], **kwargs):
        if self.wandb:
            wandb.log(data, **kwargs)

    def generate_live_table(self, od=None):
        live_table = Table(
            expand=True, box=None, show_header=False,
            show_footer=False, padding=(0, 0), pad_edge=True)
        if od:
            header = self.get_row(od=od, header=True)
            live_table.add_row(header)
        live_table.add_row(self.progress)
        return live_table

    def on_epoch_begin(self, epoch: int):
        self.progress.reset(
            task_id=self.train_p,
            total=self.train_data_len,
            start=False,
            visible=False,
            cur_epoch=epoch,
            n_epochs=self.n_epochs,
            mode="Train",
        )
        self.progress.start_task(self.train_p)
        self.progress.update(self.train_p, visible=True)

    def on_batch_end(
        self,
        logs: Interaction,
        loss: float,
        batch_id: int,
        is_training: bool = True
    ):
        if is_training:
            self.current_step += 1
            self.progress.update(self.train_p, refresh=True, advance=1)
            self.log_to_wandb({
                "batch_loss": loss,
                "batch_step": self.current_step,
                # "batch_reinf_sg": torch.mean(logs.aux['reinf_sg']).item(),
                # "batch_baseline": logs.aux['baseline'],
            }, commit=True)
        else:
            self.progress.update(self.test_p, refresh=True, advance=1)

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        od = self.build_od(logs, loss, epoch, 'train')
        if epoch == self.step or self.step == 1:
            self.live.update(self.generate_live_table(od))
        if epoch % self.step == 0:
            for k, v in od.items():
                self.history['train'][k].append(v)
            row = self.get_row(od)
            self.console.print(row)
            if self.display_nn:
                row = self.get_row(od, noise=False)
                self.console.print(row)

        self.trainer = None
        if self.wandb:
            wb_dict = {'epoch': epoch}
            wb_dict.update({f'train/{k}': v for k, v in od.items()})
            self.log_to_wandb(wb_dict)

        self.progress.stop_task(self.train_p)
        self.progress.update(self.train_p, visible=False)

        # if the datalen is zero update with the one epoch just ended
        if self.train_data_len == 0:
            self.train_data_len = self.progress.tasks[self.train_p].completed

        self.progress.reset(
            task_id=self.train_p,
            total=self.train_data_len,
            start=False,
            visible=False,
            cur_epoch=epoch,
            n_epochs=self.n_epochs,
            mode="Train",
        )

    def on_validation_begin(self, epoch: int):
        self.progress.reset(
            task_id=self.test_p,
            total=self.test_data_len,
            start=False,
            visible=True,
            cur_epoch=epoch,
            n_epochs=self.n_epochs,
            mode="Test",
        )

        self.progress.start_task(self.test_p)
        self.progress.update(self.test_p, visible=True)

    def on_validation_end(self, loss: float, logs: Interaction, epoch: int):
        self.progress.stop_task(self.test_p)
        self.progress.update(self.test_p, visible=False)

        # if the datalen is zero update with the one epoch just ended
        if self.test_data_len == 0:
            self.test_data_len = self.progress.tasks[self.test_p].completed

        self.progress.reset(
            task_id=self.test_p,
            total=self.test_data_len,
            start=False,
            visible=False,
            cur_epoch=epoch,
            n_epochs=self.n_epochs,
            mode="Test",
        )

        od = self.build_od(logs, loss, epoch, 'test')

        for k, v in od.items():
            self.history['test'][k].append(v)
        row = self.get_row(od)
        self.console.print(row)
        if self.display_nn:
            row = self.get_row(od, noise=False)
            self.console.print(row)

        if self.wandb:
            wb_dict = {'epoch': epoch}
            wb_dict.update({f'test/{k}': v for k, v in od.items()})
            self.log_to_wandb(wb_dict)

    def on_train_begin(self, trainer_instance):
        self.trainer = trainer_instance
        if self.wandb:
            wandb.watch(self.trainer.game, log='all')

    def on_train_end(self):
        self.progress.stop()
        self.live.stop()

        if self.results_folder is not None:
            for history_dict in self.history.values():
                if 'loss_nn' in history_dict:
                    del history_dict['loss_nn']
            df = pd.concat([
                pd.DataFrame(history_dict)
                for history_dict in self.history.values()
            ])
            cols = ['epoch', 'phase', 'messages']
            df = df[cols + [c for c in df.columns if c not in cols]]
            filename = f'{self.filename}-training-history.csv'
            dump_path = self.results_folder / filename
            df.to_csv(dump_path, index=False)


class TrainingEvaluationCallback(Callback):
    def __init__(self, opts: argparse.Namespace, channel: Channel):
        self.vocab_size = opts.vocab_size
        self.max_len = opts.max_len
        self.channel = channel
        self.error_prob = opts.error_prob
        self.image_input = opts.image_input
        self.max_samples = 10000
        self.eval_samples = opts.eval_test_samples

        self.train_logits = None
        self.train_logits_nn = None

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        self.train_logits = logs.logits
        self.train_logits_nn = logs.logits_nn
        self.compute(logs, training=True)

    def on_validation_end(self, loss: float, logs: Interaction, epoch: int):
        self.compute(logs, training=False)

    # @staticmethod
    # def get_attribute(attribute_dict, key):
    #     keys = [
    #         k.replace('target_', '') for k in attribute_dict
    #         if k.startswith('target_')]
    #     n_distractors = len(attribute_dict) // len(keys) - 1
    #     prefixes = ['target_'] + [f'distr{i}_' for i in range(n_distractors)]
    #     values = [attribute_dict[prefix + key] for prefix in prefixes]
    #     return torch.cat(values, dim=1)

    def compute(self, logs: Interaction, training: bool):

        def trim(tensor: torch.Tensor, strict=False):
            # selects up to self.max_samples (or self.eval_samples if strict)
            # of most recent positions from the tensor
            max_size = self.max_samples if not strict else self.eval_samples
            return tensor if len(tensor) <= max_size else tensor[-max_size:]

        messages = crop_messages(logs.message) \
            if logs.message.dim() == 2 \
            else crop_messages(logs.message.argmax(-1))

        attr_keys = [k for k in logs.aux_input if k.startswith('target')]
        attr = torch.cat([logs.aux_input[k] for k in attr_keys], dim=1)

        unique_msg, categorized_msg = \
            torch.unique(messages, dim=0, return_inverse=True)
        logs.aux['lexicon_size'] = len(unique_msg)
        logs.aux['actual_vocab_size'] = torch.unique(messages).numel()

        entropy, length_probs = message_entropy(trim(logs.logits))
        max_entropy = self.channel.max_message_entropy(length_probs, True)
        # logs.aux['expected_length'] = (torch.arange(logs.logits.size(1)) * length_probs.cpu()).sum(-1)
        logs.aux['entropy_msg_as_a_whole'] = entropy_message_as_a_whole(
            logs.logits, max_len=self.max_len, vocab_size=self.vocab_size,
            erasure_channel=isinstance(self.channel, ErasureChannel),
            n_samples=(10 if training else 200))
        logs.aux['entropy_msg'] = entropy
        logs.aux['entropy_max'] = max_entropy
        logs.aux['MI_sent_received'] = mutual_info_sent_received(
            logits_sent=logs.logits_nn,
            logits_received=logs.logits,
            max_len=self.max_len,
            vocab_size=self.vocab_size,
            erasure_channel=isinstance(self.channel, ErasureChannel),
            n_samples=(10 if training else 200),
        )
        logs.aux['max_rep'] = compute_max_rep(messages)
        logs.aux['redundancy'] = 1 - entropy / max_entropy
        if not training:
            logs.aux['KLD_train_test'] = relative_message_entropy(
                trim(self.train_logits), trim(logs.logits))
            logs.aux['KLD_test_train'] = relative_message_entropy(
                trim(logs.logits), trim(self.train_logits))
        else:
            logs.aux['KLD_train_test'] = None
            logs.aux['KLD_test_train'] = None

        if self.image_input:
            logs.aux['topsim'] = compute_top_sim(
                trim(attr, True),
                trim(messages, True))

            mi_attr = compute_mi(trim(logs.logits), trim(attr), entropy)
            logs.aux['entropy_attr'] = mi_attr['entropy_attr']
            for i, key in enumerate(attr_keys):
                key = key.replace('target_', '')
                logs.aux.update({
                    k.replace('attr_dim', key): v[i]
                    for k, v in mi_attr.items() if 'attr_dim' in k
                })
        else:
            logs.aux['topsim'] = compute_top_sim(
                trim(logs.sender_input, True),
                trim(messages, True))

            # assign a different number to every input vector
            _, input_cat = torch.unique(
                logs.sender_input, return_inverse=True, dim=0)
            input_cat = input_cat.unsqueeze(-1).to(torch.float)
            logs.aux.update({
                k.replace('attr', 'input'): v for k, v
                in compute_mi(
                    trim(logs.logits),
                    trim(input_cat),
                    entropy,
                ).items()
            })
            logs.aux.update({
                k.replace('attr', 'cat'): v for k, v
                in compute_mi(
                    trim(logs.logits),
                    trim(attr),
                    entropy,
                ).items()
            })

        # compute measure values for messages before noise is applied
        if not isinstance(self.channel, NoChannel):
            messages = crop_messages(logs.message_nn) \
                if logs.message_nn.dim() == 2 \
                else crop_messages(logs.message_nn.argmax(-1))

            unique_msg, categorized_msg = \
                torch.unique(messages, dim=0, return_inverse=True)

            logs.aux['loss_nn'] = None
            logs.aux['lexicon_size_nn'] = len(unique_msg)
            logs.aux['actual_vocab_size_nn'] = torch.unique(messages).numel()

            entropy, length_probs = message_entropy(trim(logs.logits_nn))
            max_entropy = self.channel.max_message_entropy(length_probs, False)
            # logs.aux['expected_length_nn'] = (torch.arange(logs.logits_nn.size(1)) * length_probs.cpu()).sum()
            logs.aux['max_rep_nn'] = compute_max_rep(messages)
            logs.aux['entropy_msg_as_a_whole_nn'] = entropy_message_as_a_whole(
                logs.logits_nn, max_len=self.max_len, vocab_size=self.vocab_size,
                erasure_channel=False, n_samples=(10 if training else 200))
            logs.aux['entropy_msg_nn'] = entropy
            logs.aux['entropy_max_nn'] = max_entropy
            logs.aux['redundancy_nn'] = 1 - entropy / max_entropy
            if not training:
                logs.aux['KLD_train_test_nn'] = relative_message_entropy(
                    trim(self.train_logits_nn),
                    trim(logs.logits_nn))
                logs.aux['KLD_test_train_nn'] = relative_message_entropy(
                    trim(logs.logits_nn),
                    trim(self.train_logits))
            else:
                logs.aux['KLD_train_test_nn'] = None
                logs.aux['KLD_test_train_nn'] = None

            if self.image_input:
                for i, key in enumerate(attr_keys):
                    key = key.replace('target_', '')
                    logs.aux.update({
                        k.replace('attr_dim', f'{key}_nn'): v[i]
                        for k, v in compute_mi(
                            trim(logs.logits_nn),
                            trim(attr),
                            entropy_message=entropy,
                        ).items() if 'attr_dim' in k
                    })

                logs.aux['topsim_nn'] = compute_top_sim(
                    trim(attr, True),
                    trim(messages, True),
                )
            else:
                logs.aux.update({
                    k.replace('attr', 'input_nn'): v for k, v in
                    compute_mi(
                        trim(logs.logits_nn),
                        trim(input_cat),
                        entropy_message=entropy,
                    ).items() if k != 'entropy_msg'
                })
                logs.aux.update({
                    k.replace('attr', 'cat_nn'): v
                    for k, v in compute_mi(
                        trim(logs.logits_nn),
                        trim(attr),
                        entropy_message=entropy,
                    ).items()
                    if k != 'entropy_msg'
                })

                logs.aux['topsim_nn'] = compute_top_sim(
                    trim(logs.sender_input, True),
                    trim(messages, True))
