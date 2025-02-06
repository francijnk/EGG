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

from ancm.metrics import (
    # compute_conceptual_alignment,
    compute_max_rep,
    compute_redundancy,
    compute_adjusted_redundancy,
    compute_top_sim,
    compute_posdis,
    compute_bosdis,
    # mutual_info,
    # sequence_entropy,
    # tensor_entropy,
    compute_mi,
)
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

        self.history = defaultdict(lambda: defaultdict(dict))
        self.hide_cols = ['receiver_entropy', 'sender_entropy', 'VI', 'MI']

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

    def get_row(self, od, header=False):
        row = Table(expand=True, box=None, show_header=header,
                    show_footer=False, padding=(0, 1), pad_edge=True)

        for colname in od.keys():
            if any(colname.startswith(col) for col in self.hide_cols):
                continue

            print_name = colname.replace(
                'xpos', 'x').replace('ypos', 'y').replace('actual_', '')

            if colname == 'epoch':
                ratio = 0.5
            else:
                ratio = 1
            row.add_column(
                print_name,
                justify='left' if colname in ('phase', 'epoch') else 'right',
                ratio=ratio)
        if not header:
            row.add_row(
                str(od.pop('epoch')),
                *[self.format_metric_val(v) for k, v in od.items()
                  if not any(k.startswith(col) for col in self.hide_cols)],
                style=self.style[od['phase']])
        return row

    @staticmethod
    def format_metric_val(val):
        if val is None or val != val:
            return '–'
        elif isinstance(val, int):
            return str(val)
        elif not isinstance(val, str):
            return f'{val:.2f}'
        else:
            return val

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
            self.history['train'][epoch] = od
            row = self.get_row(od)
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

        if epoch not in self.history['val']:
            phase = 'val'
            p_key = 'val'
        else:
            phase = 'val-nn'
            p_key = 'val-no-noise'

        od = self.build_od(logs, loss, epoch, phase)
        self.history[p_key][epoch] = od
        row = self.get_row(od)
        self.console.print(row)

        if self.wandb:
            wb_dict = {'epoch': epoch}
            wb_dict.update({f'{p_key}/{k}': v for k, v in od.items()})
            self.log_to_wandb(wb_dict)

    def on_train_begin(self, trainer_instance):
        self.trainer = trainer_instance
        if self.wandb:
            wandb.watch(self.trainer.game, log='all')

    def on_train_end(self):
        self.progress.stop()
        self.live.stop()

        if self.results_folder is not None:
            history_dfs = []
            history_keys = list(self.history.keys())
            if history_keys:
                key = history_keys[0]
                od_keys = self.history[key][list(self.history[key].keys())[0]]
                for key in self.history:
                    df = pd.DataFrame({'epoch': [int(epoch) for epoch in self.history[key]]})
                    for k in od_keys:
                        df[k] = [self.history[key][epoch][k] for epoch in self.history[key]]
                    history_dfs.append(df)
                history_df = pd.concat(history_dfs)
                dump_path = self.results_folder / f'{self.filename}-training-history.csv'
                history_df.to_csv(dump_path, index=False)
                print(f"Training history saved to {dump_path}")


class TrainingMetricsCallback(Callback):
    def __init__(
        self, vocab_size, max_len, channel_type,
        error_prob, sender, receiver, dataloader,
            device, image_input, bs=32):

        self.vocab_size = vocab_size
        self.max_len = max_len
        self.channel_type = channel_type
        self.error_prob = error_prob
        self.image_input = image_input

    def on_validation_end(self, loss: float, logs: Interaction, epoch: int):
        messages = logs.message if logs.message.dim() == 2 \
            else logs.message.argmax(-1)

        if logs.aux_input:
            aux_attribute_keys = [
                k for k in logs.aux_input if k.startswith('target')]
            aux_attributes = torch.cat([
                logs.aux_input[k] for k in aux_attribute_keys], dim=1)
        else:
            aux_attributes = None

        lexicon_size = torch.unique(messages, dim=0).shape[0]
        actual_vocab = torch.unique(torch.flatten(messages), dim=0)
        actual_vocab_size = actual_vocab.size(0)
        vocab_size = self.vocab_size + 1 \
            if self.channel_type == 'erasure' and self.error_prob > 0. \
            else self.vocab_size

        logs.aux['lexicon_size'] = int(lexicon_size)
        logs.aux['actual_vocab_size'] = int(actual_vocab_size)

        if self.image_input:
            mi_attr = compute_mi(messages, aux_attributes, vocab_size)
            logs.aux['H_msg'] = mi_attr['entropy_msg']
            # logs.aux['H_attr'] = mi_attr['entropy_attr']
            for i, key in enumerate(aux_attribute_keys):
                k = key.replace('target_', '')
                logs.aux[f'MI_{k}'] = mi_attr['mi_msg_attr_dim'][i]
                logs.aux[f'VI_{k}'] = mi_attr['vi_msg_attr_dim'][i]
                logs.aux[f'VInorm_{k}'] = mi_attr['vi_norm_msg_attr_dim'][i]
                logs.aux[f'IS_{k}'] = mi_attr['is_msg_attr_dim'][i]
        else:
            _, categorized_input = torch.unique(
                logs.sender_input, return_inverse=True, dim=0)
            categorized_input = categorized_input.unsqueeze(-1).to(torch.float)
            mi_inp = compute_mi(messages, categorized_input, vocab_size)
            mi_cat = compute_mi(messages, aux_attributes, vocab_size)
            logs.aux['H_msg'] = mi_cat['entropy_msg']
            logs.aux['MI_inp'] = mi_inp['mi_msg_attr']
            logs.aux['VI_inp'] = mi_inp['vi_msg_attr']
            logs.aux['VInorm_inp'] = mi_inp['vi_norm_msg_attr']
            logs.aux['IS_inp'] = mi_inp['is_msg_attr']
            logs.aux['MI_cat'] = mi_cat['mi_msg_attr']
            logs.aux['VI_cat'] = mi_cat['vi_msg_attr']
            logs.aux['VInorm_cat'] = mi_cat['vi_norm_msg_attr']
            logs.aux['IS_cat'] = mi_cat['is_msg_attr']

        # redundancy
        logs.aux['max_rep'] = compute_max_rep(messages)
        logs.aux['redundancy'] = compute_redundancy(
            messages, vocab_size,
            channel=None, error_prob=0.0)
        logs.aux['redundancy_adj'] = compute_adjusted_redundancy(
            messages, channel=None, error_prob=0.0,
            symbols=torch.arange(vocab_size), erased_symbol=self.vocab_size)
        logs.aux['redundancy_adj2'] = compute_adjusted_redundancy(
            messages, channel=None, error_prob=0.0,
            symbols=actual_vocab, erased_symbol=self.vocab_size)

        # compositionality
        if self.image_input and aux_attributes is not None:
            logs.aux['topsim'] = compute_top_sim(aux_attributes, messages)
        elif not self.image_input:
            logs.aux['topsim'] = compute_top_sim(logs.sender_input, messages)
            logs.aux['topsim_cat'] = compute_top_sim(aux_attributes, messages)

        if self.image_input:
            logs.aux['posdis'] = compute_posdis(aux_attributes, messages, vocab_size)
            logs.aux['bosdis'] = compute_bosdis(
                aux_attributes, messages, vocab_size)

    def on_secondary_validation_end(self, loss: float, logs: Interaction, epoch: int):
        messages = logs.message if logs.message.dim() == 2 \
            else logs.message.argmax(-1)

        if logs.aux_input:
            aux_attribute_keys = [
                k for k in logs.aux_input if k.startswith('target')]
            aux_attributes = torch.cat([
                logs.aux_input[k] for k in aux_attribute_keys], dim=1)
        else:
            aux_attributes = None

        lexicon_size = torch.unique(messages, dim=0).shape[0]
        actual_vocab = torch.unique(torch.flatten(messages), dim=0)
        actual_vocab_size = actual_vocab.size(0)

        logs.aux['lexicon_size'] = int(lexicon_size)
        logs.aux['actual_vocab_size'] = int(actual_vocab_size)

        if self.image_input:
            mi_attr = compute_mi(messages, aux_attributes, self.vocab_size)
            logs.aux['H_msg'] = mi_attr['entropy_msg']
            # logs.aux['H_attr'] = mi_attr['entropy_attr']
            for i, key in enumerate(aux_attribute_keys):
                k = key.replace('target_', '')
                # logs.aux[f'H_{k}'] = mi_attr['entropy_attr_dim'][i]
                logs.aux[f'MI_{k}'] = mi_attr['mi_msg_attr_dim'][i]
                logs.aux[f'VI_{k}'] = mi_attr['vi_msg_attr_dim'][i]
                logs.aux[f'VInorm_{k}'] = mi_attr['vi_norm_msg_attr_dim'][i]
                logs.aux[f'IS_{k}'] = mi_attr['is_msg_attr_dim'][i]
        else:
            _, categorized_input = torch.unique(
                logs.sender_input, return_inverse=True, dim=0)
            categorized_input = categorized_input.unsqueeze(-1).to(torch.float)
            mi_inp = compute_mi(messages, categorized_input, self.vocab_size)
            mi_cat = compute_mi(messages, aux_attributes, self.vocab_size)
            logs.aux['H_msg'] = mi_cat['entropy_msg']
            # logs.aux['H_inp'] = mi_inp['entropy_attr']
            logs.aux['MI_inp'] = mi_inp['mi_msg_attr']
            logs.aux['VI_inp'] = mi_inp['vi_msg_attr']
            logs.aux['VInorm_inp'] = mi_inp['vi_norm_msg_attr']
            logs.aux['IS_inp'] = mi_inp['is_msg_attr']
            logs.aux['MI_cat'] = mi_cat['mi_msg_attr']
            logs.aux['VI_cat'] = mi_cat['vi_msg_attr']
            logs.aux['VInorm_cat'] = mi_cat['vi_norm_msg_attr']
            logs.aux['IS_cat'] = mi_cat['is_msg_attr']

        # compositionality
        if self.image_input and aux_attributes is not None:
            logs.aux['topsim'] = compute_top_sim(aux_attributes, messages)
        elif not self.image_input:
            logs.aux['topsim'] = compute_top_sim(logs.sender_input, messages)
            logs.aux['topsim_cat'] = compute_top_sim(aux_attributes, messages)

        # redundancy
        logs.aux['max_rep'] = compute_max_rep(messages)
        logs.aux['redundancy'] = compute_redundancy(
            messages, self.vocab_size,
            channel=None, error_prob=0.0)
        logs.aux['redundancy_adj'] = compute_adjusted_redundancy(
            messages, channel=None, error_prob=0.0,
            symbols=torch.arange(self.vocab_size))
        logs.aux['redundancy_adj2'] = compute_adjusted_redundancy(
            messages, channel=None, error_prob=0.0,
            symbols=actual_vocab, erased_symbol=self.vocab_size)

        # compositionality
        if self.image_input and aux_attributes is not None:
            logs.aux['topsim'] = compute_top_sim(aux_attributes, messages)
        elif not self.image_input:
            logs.aux['topsim'] = compute_top_sim(logs.sender_input, messages)
            logs.aux['topsim_cat'] = compute_top_sim(aux_attributes, messages)

        if self.image_input:
            logs.aux['posdis'] = compute_posdis(aux_attributes, messages, self.vocab_size)
            logs.aux['bosdis'] = compute_bosdis(
                aux_attributes, messages, self.vocab_size)

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        messages = logs.message if logs.message.dim() == 2 \
            else logs.message.argmax(-1)

        if logs.aux_input:
            aux_attribute_keys = [
                k for k in logs.aux_input if k.startswith('target')]
            aux_attributes = torch.cat([
                logs.aux_input[k] for k in aux_attribute_keys], dim=1)
        else:
            aux_attributes = None

        vocab_size = self.vocab_size + 1 \
            if self.channel_type == 'erasure' and self.error_prob > 0. \
            else self.vocab_size

        lexicon_size = torch.unique(messages, dim=0).shape[0]
        actual_vocab = torch.unique(torch.flatten(messages), dim=0)
        actual_vocab_size = actual_vocab.size(0)
        vocab_size = self.vocab_size + 1 \
            if self.channel_type == 'erasure' and self.error_prob > 0. \
            else self.vocab_size

        logs.aux['lexicon_size'] = int(lexicon_size)
        logs.aux['actual_vocab_size'] = int(actual_vocab_size)

        if self.image_input:
            mi_attr = compute_mi(messages, aux_attributes, vocab_size)
            logs.aux['H_msg'] = mi_attr['entropy_msg']
            for i, key in enumerate(aux_attribute_keys):
                k = key.replace('target_', '')
                logs.aux[f'MI_{k}'] = mi_attr['mi_msg_attr_dim'][i]
                logs.aux[f'VI_{k}'] = mi_attr['vi_msg_attr_dim'][i]
                logs.aux[f'VInorm_{k}'] = mi_attr['vi_norm_msg_attr_dim'][i]
                logs.aux[f'IS_{k}'] = mi_attr['is_msg_attr_dim'][i]
        else:
            mi_cat = compute_mi(messages, aux_attributes, vocab_size)
            logs.aux['H_msg'] = mi_cat['entropy_msg']
            logs.aux['MI_inp'] = None
            logs.aux['VI_inp'] = None
            logs.aux['IS_inp'] = None
            logs.aux['VInorm_inp'] = None
            logs.aux['MI_cat'] = mi_cat['mi_msg_attr']
            logs.aux['VI_cat'] = mi_cat['vi_msg_attr']
            logs.aux['VInorm_cat'] = mi_cat['vi_norm_msg_attr']
            logs.aux['IS_cat'] = mi_cat['is_msg_attr']

        # redundancy
        logs.aux['max_rep'] = compute_max_rep(messages)
        logs.aux['redundancy'] = compute_redundancy(
            messages, vocab_size,
            channel=None, error_prob=0.0)
        logs.aux['redundancy_adj'] = compute_adjusted_redundancy(
            messages, channel=None, error_prob=0.0,
            symbols=torch.arange(vocab_size), erased_symbol=self.vocab_size)
        logs.aux['redundancy_adj2'] = compute_adjusted_redundancy(
            messages, channel=None, error_prob=0.0,
            symbols=actual_vocab, erased_symbol=self.vocab_size)

        # compositinoality
        logs.aux['topsim'] = None
        if self.image_input:
            logs.aux['posdis'] = None
            logs.aux['bosdis'] = None
        else:
            logs.aux['topsim_cat'] = None
