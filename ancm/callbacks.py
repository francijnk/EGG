import torch
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
    compute_conceptual_alignment,
    compute_max_rep,
    compute_redundancy_msg,
    compute_redundancy_smb,
    compute_redundancy_smb_adjusted,
    compute_top_sim,
    compute_posdis,
    compute_bosdis,
)

from egg.core.util import find_lengths
from egg.core.callbacks import Callback, CustomProgress
from egg.core.interaction import Interaction


WORLD_DIM_THRESHOLD = 16


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
        n_epochs: int,
        train_data_len: int = 0,
        test_data_len: int = 0,
        validation_freq: int = 1,
        dump_results_folder=None,
        filename=None,
    ):
        """
        :param n_epochs: total number of epochs
        :param train_data_len: length of the dataset generation for training
        :param test_data_len: length of the dataset generation for testing
        :param use_info_table: true to add an information table on top of the progress bar
        """

        self.n_epochs = n_epochs
        self.train_data_len = train_data_len
        self.test_data_len = test_data_len
        self.dump_results_folder = dump_results_folder
        self.filename = filename
        self.step = validation_freq
        self.history = defaultdict(lambda: defaultdict(dict))
        self.hide_cols = ['receiver_entropy', 'sender_entropy']

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
            if colname in self.hide_cols:
                continue

            print_name = colname.replace('redund_smb', 'r_s').replace('redund_msg', 'r_m') \
                if not colname.startswith('actual_vocab_size') else 'vocab_size'

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
                  if k not in self.hide_cols],
                style=self.style[od['phase']])
        return row

    @staticmethod
    def format_metric_val(val):
        if val is None or val != val:
            return '–'
        elif isinstance(val, int):
            return str(val)
        elif not isinstance(val, str):
            return f'{val: 4.2f}'
        else:
            return val

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
            self.progress.update(self.train_p, refresh=True, advance=1)
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
            phase = 'val (nn)'
            p_key = 'val-no-noise'

        od = self.build_od(logs, loss, epoch, phase)
        self.history[p_key][epoch] = od
        row = self.get_row(od)
        self.console.print(row)

    def on_train_end(self):
        self.progress.stop()
        self.live.stop()

        if self.dump_results_folder is not None:
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
                dump_path = self.dump_results_folder / f'{self.filename}-training-history.csv'
                history_df.to_csv(dump_path, index=False)
                print(f"| Training history saved to {self.dump_results_folder / self.filename}-training-history.csv")


class TrainingMetricsCallback(Callback):
    def __init__(self, vocab_size, max_len, channel_type, error_prob, sender, receiver, dataloader, device, bs=32):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.channel_type = channel_type
        self.error_prob = error_prob

        # to compute speaker-listener alignment
        self.sender = sender
        self.receiver = receiver
        self.dataloader = dataloader
        self.device = device
        self.bs = bs

    def on_validation_end(self, loss: float, logs: Interaction, epoch: int):
        if logs.message is not None:
            if logs.message.dim() == 3:
                # under GS, EOS might be followed by a non-EOS symbol
                message = logs.message.argmax(-1)
                lengths = find_lengths(message)
                for i in range(message.size(0)):
                    message[i, lengths[i]:] = 0
            else:
                message = logs.message

            vocab_size = self.vocab_size + 1 \
                if self.channel_type == 'erasure' and self.error_prob > 0. \
                else self.vocab_size

            lexicon_size = torch.unique(logs.message, dim=0).shape[0]
            actual_vocab = torch.unique(torch.flatten(message), dim=0)
            actual_vocab_size = actual_vocab.size(0)

            logs.aux['lexicon_size'] = int(lexicon_size)
            logs.aux['actual_vocab_size'] = int(actual_vocab_size)
            logs.aux['alignment'] = compute_conceptual_alignment(self.dataloader, self.sender, self.receiver, self.device, self.bs)

            # redundancy
            logs.aux['max_rep'] = compute_max_rep(message)
            logs.aux['redund_msg'] = compute_redundancy_msg(logs.message, self.max_len)
            logs.aux['redund_smb'] = compute_redundancy_smb(
                message, self.max_len, self.vocab_size, self.channel_type, self.error_prob)
            logs.aux['redund_smb_adj'] = compute_redundancy_smb_adjusted(
                message, self.max_len, self.vocab_size, self.channel_type, self.error_prob)
            logs.aux['redund_smb_adj2'] = compute_redundancy_smb_adjusted(
                message, self.max_len, self.vocab_size, self.channel_type, self.error_prob, alphabet=actual_vocab)

            # compositinoality
            logs.aux['top_sim'] = compute_top_sim(logs.sender_input, logs.message)
            logs.aux['pos_dis'] = compute_posdis(logs.sender_input, logs.message)
            logs.aux['bos_dis'] = compute_bosdis(logs.sender_input, logs.message, vocab_size)

    def on_secondary_validation_end(self, loss: float, logs: Interaction, epoch: int):
        if logs.message is not None:
            if logs.message.dim() == 3:
                # under GS, EOS might be followed by a non-EOS symbol
                message = logs.message.argmax(-1)
                lengths = find_lengths(message)
                for i in range(message.size(0)):
                    message[i, lengths[i]:] = 0
            else:
                message = logs.message

            lexicon_size = torch.unique(logs.message, dim=0).shape[0]
            actual_vocab = torch.unique(torch.flatten(message), dim=0)
            actual_vocab_size = actual_vocab.size(0)

            logs.aux['lexicon_size'] = int(lexicon_size)
            logs.aux['actual_vocab_size'] = int(actual_vocab_size)
            logs.aux['alignment'] = compute_conceptual_alignment(self.dataloader, self.sender, self.receiver, self.device, self.bs)

            # redundancy
            logs.aux['max_rep'] = compute_max_rep(message)
            logs.aux['redund_msg'] = compute_redundancy_msg(logs.message, self.max_len)
            logs.aux['redund_smb'] = compute_redundancy_smb(
                message, self.max_len, self.vocab_size, channel=None, error_prob=0.0)
            logs.aux['redund_smb_adj'] = compute_redundancy_smb_adjusted(
                message, self.max_len, self.vocab_size, channel=None, error_prob=0.0)
            logs.aux['redund_smb_adj2'] = compute_redundancy_smb_adjusted(
                message, self.max_len, self.vocab_size, channel=None, error_prob=0.0, alphabet=actual_vocab)

            # compositionality
            logs.aux['top_sim'] = compute_top_sim(logs.sender_input, logs.message)
            logs.aux['pos_dis'] = compute_posdis(logs.sender_input, logs.message)
            logs.aux['bos_dis'] = compute_bosdis(logs.sender_input, logs.message, self.vocab_size)

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        if logs.message is not None:
            if logs.message.dim() == 3:
                # under GS, EOS might be followed by a non-EOS symbol
                message = logs.message.argmax(-1)
                lengths = find_lengths(message)
                for i in range(message.size(0)):
                    message[i, lengths[i]:] = 0
            else:
                message = logs.message

            # vocab_size = self.vocab_size + 1 \
            #     if self.channel_type == 'erasure' and self.error_prob > 0. \
            #     else self.vocab_size

            lexicon_size = torch.unique(logs.message, dim=0).shape[0]
            actual_vocab = torch.unique(torch.flatten(message), dim=0)
            actual_vocab_size = actual_vocab.size(0)

            logs.aux['lexicon_size'] = int(lexicon_size)
            logs.aux['actual_vocab_size'] = int(actual_vocab_size)
            logs.aux['alignment'] = None  # compute_conceptual_alignment(self.dataloader, self.sender, self.receiver, self.device, self.bs)

            # redundancy
            logs.aux['max_rep'] = compute_max_rep(message)
            logs.aux['redund_msg'] = None  # compute_redundancy_msg(logs.message, self.max_len)
            logs.aux['redund_smb'] = compute_redundancy_smb(
                message, self.max_len, self.vocab_size, self.channel_type, self.error_prob)
            logs.aux['redund_smb_adj'] = compute_redundancy_smb_adjusted(
                message, self.max_len, self.vocab_size, self.channel_type, self.error_prob)
            logs.aux['redund_smb_adj2'] = compute_redundancy_smb_adjusted(
                message, self.max_len, self.vocab_size, self.channel_type, self.error_prob, alphabet=actual_vocab)

            # compositinoality
            logs.aux['top_sim'] = None  # top_sim(logs.sender_input, logs.message)
            logs.aux['pos_dis'] = None  # posdis(logs.sender_input, logs.message)
            logs.aux['bos_dis'] = None  # bosdis(logs.sender_input, logs.message, self.vocab_size)
