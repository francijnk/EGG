import torch
import wandb
import argparse
import numpy as np
import pandas as pd
from scipy.stats import binom
from torch.distributions.categorical import Categorical
import pyitlib.discrete_random_variable as drv

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
    mutual_info_sent_received,
    compute_mi,
    compute_topsim,
)
from src.channels import Channel, NoChannel, ErasureChannel
from src.util import crop_messages
from src.interaction import Interaction

from egg.core.util import find_lengths
from egg.core.callbacks import Callback, CustomProgress

from typing import Dict, Any


class ReceiverResetCallback(Callback):
    """
    We do not reset parameters in our experimens.
    """
    def __init__(self, game, opts):
        self.game = game
        self.reset_epochs = [] if opts.receiver_reset_freq is None \
            else range(0, opts.n_epochs - 1, opts.receiver_reset_freq)

    def on_epoch_begin(self, epoch: int):
        if epoch - 1 in self.reset_epochs and epoch > 1:
            self.game.receiver.cell.reset_parameters()
            self.game.receiver.message_encoder.reset_parameters()
            self.game.receiver.input_encoder.reset_parameters()


class TemperatureAnnealer(Callback):
    def __init__(self, game, opts):
        if opts.temperature_start is None:
            opts.temperature_start = opts.temperature
            print('temperature_start not specified, using temperature instead')
        if opts.temperature_end is None:
            opts.temperature_end = opts.temperature
            print('temperature_end not specified, using temperature instead')

        self.agent = game.sender
        self.temperatures = np.geomspace(
            opts.temperature_start,
            opts.temperature_end,
            num=opts.n_epochs,
        )

    def on_epoch_begin(self, epoch: int, *args):
        self.agent.temperature = self.temperatures[epoch - 1]


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
        self.display_nn = opts.channel is not None
        self.reset_epochs = [] if opts.receiver_reset_freq is None \
            else range(0, opts.n_epochs - 1, opts.receiver_reset_freq)

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
        self.hidden_cols = [
            'entropy_max', 'time',
            'entropy_target', 'entropy_target_category', 'entropy_selected',
            'entropy_shape', 'entropy_color',
            'entropy_x', 'entropy_y', 'entropy_shade', 'entropy_attr',
            'redundancy_msg',
            'actual_vocab_size',
            'proficiency_msg_rotation',
            'entropy_msg_1', 'entropy_msg_mc',
            # 'entropy_msg_mc_minimax', 'entropy_msg_mc_james-stein',
            'mutual_info_msg_color', 'mutual_info_msg_shape',
            'mutual_info_msg_shade',
            'mutual_info_msg_x', 'mutual_info_msg_y',
            # 'proficiency_msg_target',
            # 'proficiency_msg_target_category',
            'proficiency_msg_selected_',
            'kld_test_train',
            'mutual_info_msg_target', 'mutual_info_msg_selected',
        ]
        if opts.temperature_start is None and opts.temperature_end is None:
            self.hidden_cols.append('temperature')

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
                od['messages'] = 'received' if noise else 'sent'
                row_values.append('received' if noise else 'sent')
            else:
                row_values.append(str(od[colname]))
            row.add_column(
                colname,
                justify='left',
                ratio=0.5 if colname != 'messages' else 1)

        for colname in od:
            if (
                any(colname.startswith(c) for c in self.hidden_cols)
                or colname.endswith('_nn')
                or colname in ('epoch', 'phase', 'messages')
            ):
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
                .replace('proficiency', 'U')
                .replace('actual_', '')
                .replace('_msg_', '_')
                .replace('mutual_info', 'MI')
                .replace('target_', '')
            )
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

        if epoch in self.reset_epochs:
            self.console.rule("[italic]Receiver parameter reset")

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

        # some metrics are computed only for the most recent samples
        self.trim_samples = 10000
        self.trim_samples_strict = 2000  # for top sim
        self.eval_samples = opts.eval_test_samples

        # for KLD
        self.train_logits = None
        self.train_logits_nn = None

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        self.train_logits = logs.logits
        self.train_logits_nn = logs.logits_nn

        self.evaluate(logs, training=True)

    def on_validation_end(self, loss: float, logs: Interaction, epoch: int):
        self.evaluate(logs, training=False)

    def evaluate(self, logs: Interaction, training: bool):

        def trim(tensor: torch.Tensor, strict=False):
            # selects up to N most recent positions from the tensor
            # where N = trim_samples or N = min(trim_samples, eval_samples)
            max_size = self.trim_samples if not strict \
                else self.trim_samples_strict
            return tensor if len(tensor) <= max_size else tensor[-max_size:]

        mc_samples = 20 if training else 100
        if self.image_input:
            mc_samples *= 2

        def entropy_benchmark(noise: bool):
            logits = trim(logs.logits if noise else logs.logits_nn)
            distr = Categorical(logits=logits)

            sample_mc = distr.sample((mc_samples,))
            size = sample_mc.size()
            sample_mc = crop_messages(sample_mc.reshape(size[0] * size[1], size[2]))
            _, sample_mc = torch.unique(sample_mc, return_inverse=True, dim=0)

            sample_single = crop_messages(distr.sample())
            _, sample_single = torch.unique(sample_single, return_inverse=True, dim=0)

            vocab_size = self.vocab_size if isinstance(self.channel, ErasureChannel) \
                else self.vocab_size - 1
            n_messages = np.sum(vocab_size ** np.arange(self.max_len + 1))
            for mc in (True, False):
                for estimator in ('ml', 'perks', 'james-stein', 'minimax'):
                    key = f'entropy_msg_{"mc" if mc else 1}_{estimator}'
                    if not noise:
                        key += '_nn'
                    logs.aux[key] = drv.entropy(
                        (sample_mc if mc else sample_single).cpu().numpy(),
                        Alphabet_X=np.arange(n_messages),
                        estimator=estimator.upper(),
                    ).item()

        messages = crop_messages(logs.message.argmax(-1))

        # target object attributes
        idx = torch.arange(len(messages), device=messages.device).long()
        attr = torch.stack([
            a[idx, logs.labels] for a in logs.aux_input.values()
        ], dim=-1)

        # selected object attributes
        lengths = find_lengths(messages)
        s_objs = logs.receiver_output.argmax(-1)[idx, lengths - 1]
        s_attr = torch.stack([
            a[idx, s_objs] for a in logs.aux_input.values()
        ], dim=-1)  # selected object attributes

        unique_msg, categorized_msg = \
            torch.unique(messages, dim=0, return_inverse=True)
        logs.aux['lexicon_size'] = len(unique_msg)
        logs.aux['actual_vocab_size'] = torch.unique(messages).numel()

        entropy, length_probs = message_entropy(trim(logs.logits))
        entropy_nn, length_probs_nn = message_entropy(trim(logs.logits_nn))

        max_entropy = self.channel.max_message_entropy(
            length_probs, True)  # given actual message length
        # logs.aux['expected_length'] = torch.sum(
        #     torch.arange(logs.logits.size(1) * length_probs.cpu()), dim=-1)
        logs.aux['entropy_msg'] = entropy
        logs.aux['entropy_max'] = max_entropy

        entropy_benchmark(True)

        logs.aux['redundancy'] = 1 - entropy / max_entropy
        if not training:
            logs.aux['kld_train_test'] = relative_message_entropy(
                trim(self.train_logits),
                trim(logs.logits))
            logs.aux['kld_test_train'] = relative_message_entropy(
                trim(logs.logits),
                trim(self.train_logits))
        else:
            logs.aux['kld_train_test'] = None
            logs.aux['kld_test_train'] = None

        mi_kwargs = {
            'max_len': self.max_len,
            'vocab_size': self.vocab_size,
            'n_samples': mc_samples,
            'erasure_channel': isinstance(self.channel, ErasureChannel),
        }
        logs.aux['mutual_info_sent_received'] = mutual_info_sent_received(
            logits_sent=trim(logs.logits_nn),
            channel=self.channel,
            entropy_sent=entropy_nn,
            entropy_received=entropy,
            **mi_kwargs)

        topsim_args = (
            trim(attr if self.image_input else logs.sender_input, True),
            trim(messages, True),
        )
        logs.aux.update(
            topsim=compute_topsim(*topsim_args, normalize=False),
            topsim_norm_max=compute_topsim(*topsim_args, normalize=True),
        )

        mi_kwargs['entropy_message'] = entropy
        if self.image_input:
            mi_target = compute_mi(trim(logs.logits), trim(attr), **mi_kwargs)
            logs.aux.update(
                entropy_target=mi_target['entropy_attr'],
                mutual_info_msg_target=mi_target['mutual_info_msg_attr'],
                proficiency_msg_target=mi_target['proficiency_msg_attr'],
            )
            for i, key in enumerate(logs.aux_input):
                logs.aux.update({
                    k.replace('attr_dim', f'target_{key}'): v[i]
                    for k, v in mi_target.items() if 'attr_dim' in k
                })
            else:
                mi_selected = compute_mi(
                    trim(logs.logits), trim(s_attr),
                    **mi_kwargs,
                )
                logs.aux.update(
                    entropy_selected=mi_selected['entropy_attr'],
                    mutual_info_msg_selected=mi_selected['mutual_info_msg_attr'],
                    proficiency_msg_selected=mi_selected['proficiency_msg_attr'],
                )
                for i, key in enumerate(logs.aux_input):
                    logs.aux.update({
                        k.replace('attr_dim', f'selected_{key}'): v[i]
                        for k, v in mi_selected.items() if 'attr_dim' in k
                    })
        else:
            logs.aux['topsim_cosine'] = compute_topsim(
                *topsim_args, meaning_distance='cosine', normalize=False)
            logs.aux['topsim_cosine_norm_max'] = compute_topsim(
                *topsim_args, meaning_distance='cosine', normalize=True)

            # categorize the input vectors
            _, target = torch.unique(
                logs.sender_input, return_inverse=True, dim=0)
            target = target.unsqueeze(-1).to(torch.float)
            logs.aux.update({
                k.replace('attr', 'target'): v for k, v
                in compute_mi(
                    trim(logs.logits), trim(target), **mi_kwargs,
                ).items()
            })
            logs.aux.update({
                k.replace('attr', 'target_category'): v for k, v in
                compute_mi(trim(logs.logits), trim(attr), **mi_kwargs).items()
            })
            _, selected = torch.unique(
                logs.receiver_input[idx, s_objs],
                return_inverse=True, dim=0)
            selected = selected.unsqueeze(-1).to(torch.float)
            logs.aux.update({
                k.replace('attr', 'selected'): v for k, v
                in compute_mi(
                    trim(logs.logits), trim(selected), **mi_kwargs
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

            max_entropy_nn = self.channel.max_message_entropy(
                length_probs_nn, False)  # length-adjusted
            # logs.aux['expected_length_nn'] = torch.sum(
            #     torch.arange(logs.logits_nn.size(1)) * length_probs.cpu()))
            logs.aux['entropy_msg_nn'] = entropy_nn
            logs.aux['entropy_max_nn'] = max_entropy_nn

            entropy_benchmark(False)

            logs.aux['redundancy_nn'] = 1 - entropy_nn / max_entropy_nn
            if not training:
                logs.aux['kld_train_test_nn'] = relative_message_entropy(
                    trim(self.train_logits_nn),
                    trim(logs.logits_nn))
                logs.aux['kld_test_train_nn'] = relative_message_entropy(
                    trim(logs.logits_nn),
                    trim(self.train_logits_nn))
            else:
                logs.aux['kld_train_test_nn'] = None
                logs.aux['kld_test_train_nn'] = None

            topsim_args = (
                trim(attr if self.image_input else logs.sender_input, True),
                trim(messages, True),
            )
            logs.aux.update(
                topsim_nn=compute_topsim(*topsim_args, normalize=False),
                topsim_norm_max_nn=compute_topsim(*topsim_args, normalize=True),
            )

            mi_kwargs.update(erasure_channel=False, entropy_message=entropy_nn)
            if self.image_input:
                mi_target = compute_mi(
                    trim(logs.logits_nn), trim(attr), **mi_kwargs
                )
                logs.aux.update(
                    mutual_info_msg_target_nn=mi_target['mutual_info_msg_attr'],
                    proficiency_msg_target_nn=mi_target['proficiency_msg_attr'],
                )
                for i, key in enumerate(logs.aux_input):
                    logs.aux.update({
                        k.replace('attr_dim', f'target_{key}_nn'): v[i]
                        for k, v in mi_target.items() if 'attr_dim' in k
                    })
                mi_selected = compute_mi(
                    trim(logs.logits_nn), trim(s_attr), **mi_kwargs
                )
                logs.aux.update(
                    entropy_selected_nn=mi_selected['entropy_attr'],
                    mutual_info_msg_selected_nn=mi_selected['mutual_info_msg_attr'],
                    proficiency_msg_selected_nn=mi_selected['proficiency_msg_attr'],
                )
                for i, key in enumerate(logs.aux_input):
                    logs.aux.update({
                        k.replace('attr_dim', f'selected_{key}_nn'): v[i]
                        for k, v in mi_selected.items() if 'attr_dim' in k
                    })
            else:
                logs.aux['topsim_cosine_nn'] = compute_topsim(
                    *topsim_args, meaning_distance='cosine', normalize=False)
                logs.aux['topsim_cosine_norm_nn'] = compute_topsim(
                    *topsim_args, meaning_distance='cosine', normalize=True)

                logs.aux.update({
                    k.replace('attr', 'target_nn'): v for k, v in
                    compute_mi(
                        trim(logs.logits_nn), trim(target), **mi_kwargs
                    ).items() if k != 'entropy_msg'
                })
                logs.aux.update({
                    k.replace('attr', 'target_category_nn'): v for k, v
                    in compute_mi(
                        trim(logs.logits_nn), trim(attr), **mi_kwargs
                    ).items() if k != 'entropy_msg'
                })
                logs.aux.update({
                    k.replace('attr', 'selected_nn'): v for k, v
                    in compute_mi(
                        trim(logs.logits_nn), trim(selected), **mi_kwargs
                    ).items() if k != 'entropy_msg'
                })
