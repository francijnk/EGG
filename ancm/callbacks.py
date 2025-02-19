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

from ancm.measures import (
    compute_mi,
    message_entropy,
    compute_max_rep,
    compute_top_sim,
    compute_posdis,
    compute_bosdis,
)
from ancm.channels import NoChannel
from ancm.util import crop_messages

from egg.core.callbacks import Callback, CustomProgress
from egg.core.interaction import Interaction

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
    Displays a progress bar with information about the current epoch and the epoch progression.
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
        :param use_info_table: true to add an information table on top of the progress bar
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
            'receiver_entropy', 'sender_entropy', 'VI', 'MI',
            'length_probs', 'entropy_inp', 'entropy_cat']

        self.progress = CustomProgress(
            TextColumn(
                "[bold]{task.fields[cur_epoch]}/{task.fields[n_epochs]} | [blue]{task.fields[mode]}",
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

    def format_metric_val(self, val):
        # if isinstance(val, tuple) and val[0] is not None:
        #     return (
        #         f'{self.format_metric_val(val[0])}'
        #         f' ({self.format_metric_val(val[1])})'
        #     )
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

        # row_values = [
        #     str(od['epoch']),
        #     od['phase'],
        #     'noise' if noise else 'no noise'
        # ]
        row_values = []
        for colname in ('epoch', 'phase', 'condition'):
            if colname == 'condition' and not self.display_nn:
                continue
            elif colname == 'condition' and self.display_nn:
                row_values.append('noise' if noise else 'no noise')
            else:
                row_values.append(str(od[colname]))
            row.add_column(
                colname,
                justify='left',
                ratio=0.5 if colname != 'condition' else 1)
        # colnames = {
        #     k: None for k in od
        #     if not k.endswith('_nn')
        #     and k not in ('epoch', 'phase')
        # }.keys()
        for colname in od:
            if any(colname.startswith(c) for c in self.hide_cols) \
                    or colname.endswith('_nn') \
                    or colname in ('epoch', 'phase'):
                continue
            # if any(colname.startswith(col) for col in self.hide_cols) \
            #         or colname.endswith('_nn'):
            #     continue

            # if colname == 'epoch':
            #    value = str(od['epoch'])
                # row_values.append(str(od['epoch']))
            # elif self.display_nn:
            #     value = od[colname] if f'{colname}_nn' not in od.keys() \
            #         else (od[colname], od[f'{colname}_nn'])
            #     row_values.append(self.format_metric_val(value))
            if noise:
                value = self.format_metric_val(
                    od[f'{colname}_nn'] if f'{colname}_nn' in od
                    else od[colname]
                )
            else:
                value = self.format_metric_val(od[colname])
               
            row_values.append(value)
            # row_values.append(self.format_metric_val(od[colname]))
            print_name = (
                colname
                #.replace('accuracy', 'acc')
                #.replace('length', 'len')
                .replace('entropy', 'H')
                .replace('redundancy', 'R')
                .replace('actual_', '')
                #.replace('lexicon', 'lex')
                #.replace('input', 'inp')
                #.replace('category', 'cat')
                #.replace('xpos', 'x')
                #.replace('ypos', 'y')
                .replace('_msg_', '_')
            )
            row.add_column(
                print_name,
                justify='right',  # 'left' if colname in ('phase', 'epoch') else 'center',
                ratio=0.5 if colname in ('loss', 'length') else 1)
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

    def on_batch_end(self, logs: Interaction, loss: float, batch_id: int, is_training: bool = True):
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
            mode="Train")

    def on_validation_begin(self, epoch: int):
        self.progress.reset(
            task_id=self.test_p,
            total=self.test_data_len,
            start=False,
            visible=True,
            cur_epoch=epoch,
            n_epochs=self.n_epochs,
            mode="Validate")

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
            mode="Validate")

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
            wb_dict.update({f'val/{k}': v for k, v in od.items()})
            self.log_to_wandb(wb_dict)

    def on_train_begin(self, trainer_instance):
        self.trainer = trainer_instance
        if self.wandb:
            wandb.watch(self.trainer.game, log='all')

    def on_train_end(self):
        self.progress.stop()
        self.live.stop()

        if self.results_folder is not None:
            history_df = pd.concat([
                pd.DataFrame(history_dict)
                for history_dict in self.history.values()
            ])
            filename = f'{self.filename}-training-history.csv'
            dump_path = self.results_folder / filename
            history_df.to_csv(dump_path, index=False)
            print(f"Training history saved to {dump_path}")


class TrainingEvaluationCallback(Callback):
    def __init__(
        self, vocab_size, max_len, channel,
        error_prob, sender, receiver, dataloader,
            device, image_input, bs=32):

        self.vocab_size = vocab_size
        self.max_len = max_len
        self.channel = channel
        self.error_prob = error_prob
        self.image_input = image_input

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        self.compute(logs, training=True)

    def on_validation_end(self, loss: float, logs: Interaction, epoch: int):
        self.compute(logs, training=False)

    def compute(self, logs, training=False):
        messages = crop_messages(logs.message)

        if logs.aux_input:
            aux_attribute_keys = [
                k for k in logs.aux_input if k.startswith('target')]
            aux_attributes = torch.cat([
                logs.aux_input[k] for k in aux_attribute_keys], dim=1)
        else:
            aux_attributes = None

        logs.aux['lexicon_size'] = len(torch.unique(messages, dim=0))
        logs.aux['actual_vocab_size'] = torch.unique(messages).numel()

        entropy, length_probs = message_entropy(logs.probs)
        max_entropy = self.channel.compute_max_entropy(length_probs)
        logs.aux['max_rep'] = compute_max_rep(messages)
        logs.aux['entropy_msg'] = entropy
        logs.aux['entropy_max'] = max_entropy
        logs.aux['redundancy'] = 1 - entropy / max_entropy

        # exp_len = (
        #     (torch.arange(logs.probs.size(1)) + 1)
        #     * length_probs).sum()
        # logs.aux['exp_len'] = exp_len

        vocab_size = logs.probs.size(-1)  # includes additional symbols
        vocab_size_nn = self.vocab_size  # does not include additional symbols

        if self.image_input:
            mi_attr = compute_mi(logs.probs, aux_attributes, entropy)
            logs.aux['entropy_attr'] = mi_attr['entropy_attr']
            for i, key in enumerate(aux_attribute_keys):
                logs.aux[key].update({
                    k.replace('attr_dim', key): v[i]
                    for k, v in mi_attr if 'attr_dim' in k
                })
                # if k != 'entropy_msg'})
            logs.aux['topsim'] = None if training else \
                compute_top_sim(aux_attributes, messages)
            logs.aux['posdis'] = None if training else \
                compute_posdis(aux_attributes, messages, vocab_size)
            logs.aux['bosdis'] = None if training else \
                compute_bosdis(aux_attributes, messages, vocab_size)
        else:
            _, categorized_input = torch.unique(
                logs.sender_input, return_inverse=True, dim=0)
            categorized_input = categorized_input.unsqueeze(-1).to(torch.float)
            logs.aux.update({
                k.replace('attr', 'input'): v for k, v in 
                compute_mi(logs.probs, categorized_input, entropy).items()
            })
            logs.aux.update({
                k.replace('attr', 'category'): v
                for k, v in compute_mi(
                    logs.probs, aux_attributes, entropy).items()
            })

            logs.aux['topsim'] = None if training else \
                compute_top_sim(logs.sender_input, messages)
            logs.aux['topsim_cat'] = None if training else \
                compute_top_sim(aux_attributes, messages)

        if isinstance(self.channel, NoChannel):
            messages = crop_messages(logs.message_nn)
            logs.aux['lexicon_size_nn'] = len(torch.unique(messages, dim=0))
            logs.aux['actual_vocab_size_nn'] = torch.unique(messages).numel()

            entropy, length_probs = message_entropy(logs.probs_nn)
            max_entropy = self.channel.compute_max_entropy(length_probs)
            logs.aux['max_rep_nn'] = compute_max_rep(messages)
            logs.aux['entropy_msg_nn'] = entropy
            logs.aux['entropy_max_nn'] = max_entropy
            logs.aux['redundancy_nn'] = 1 - entropy / max_entropy

            if self.image_input:
                mi_attr = compute_mi(logs.probs_nn, aux_attributes, entropy)
                logs.aux['entropy_attr'] = mi_attr['entropy_attr']
                for i, key in enumerate(aux_attribute_keys):
                    logs.aux[key].update({
                        k.replace('attr_dim', key): v[i]
                        for k, v in mi_attr if 'attr_dim' in k
                    })

                logs.aux['topsim_nn'] = None if training else \
                    compute_top_sim(aux_attributes, messages_nn)
                logs.aux['posdis_nn'] = None if training else \
                    compute_posdis(aux_attributes, messages_nn, vocab_size_nn)
                logs.aux['bosdis_nn'] = None if training else \
                    compute_bosdis(aux_attributes, messages_nn, vocab_size_nn)
            else:
                logs.aux.update({
                    k.replace('attr', 'input'): v for k, v in 
                    compute_mi(logs.probs, categorized_input, entropy).items()
                })
                logs.aux.update({
                    k.replace('attr', 'category'): v
                    for k, v in compute_mi(
                        logs.probs, aux_attributes, entropy).items()
                    if k != 'entropy_msg'
                })

                logs.aux['topsim_nn'] = None if training else \
                    compute_top_sim(logs.sender_input, messages)
                logs.aux['topsim_cat_nn'] = None if training else \
                    compute_top_sim(aux_attributes, messages)
