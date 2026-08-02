
import torch
import torch.nn as nn
import math
from transformer.Constants import *
from transformer.Layers import EncoderLayer

def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(PAD).type(torch.float).unsqueeze(-1) # non PAD -> 1.0; PAD -> 0.0; [B, S] -> [B, S, 1]

def get_attn_key_pad_mask(seq_k, seq_q):
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)
    return padding_mask

def get_subsequent_mask(seq):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask

class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self,
            num_types, d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout):
        super().__init__()

        self.d_model = d_model

        # position vector, used for temporal encoding
        self.register_buffer(
            'position_vec',
            torch.tensor(
                [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)]
            ),
        )

        # event type embedding
        self.event_emb = nn.Embedding(num_types + 1, d_model, padding_idx=PAD)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
            for _ in range(n_layers)])
    
    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """

        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask

    def forward(self, event_type, event_time, non_pad_mask):
        """ Encode event sequences via masked self-attention. """

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        slf_attn_mask_subseq = get_subsequent_mask(event_type)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_type, seq_q=event_type)
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        tem_enc = self.temporal_enc(event_time, non_pad_mask)
        enc_output = self.event_emb(event_type)

        for enc_layer in self.layer_stack:
            enc_output += tem_enc
            enc_output, _ = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
        return enc_output

class RNN_layers(nn.Module):
    """
    Optional recurrent layers. This is inspired by the fact that adding
    recurrent layers on top of the Transformer helps language modeling.
    """

    def __init__(self, d_model, d_rnn):
        super().__init__()

        self.rnn = nn.LSTM(d_model, d_rnn, num_layers=1, batch_first=True)
        self.projection = nn.Linear(d_rnn, d_model)

    def forward(self, data, non_pad_mask):
        lengths = non_pad_mask.squeeze(2).long().sum(1).cpu()
        pack_enc_output = nn.utils.rnn.pack_padded_sequence(
            data, lengths, batch_first=True, enforce_sorted=False)
        temp = self.rnn(pack_enc_output)[0]
        out = nn.utils.rnn.pad_packed_sequence(
            temp, batch_first=True, total_length=data.size(1))[0]

        out = self.projection(out)
        return out

class Predictor(nn.Module):
    """Simple linear predictor (used in baseline mode)."""
    def __init__(self, dim, c_out=1):
        super().__init__()
        self.linear = nn.Linear(dim, c_out, bias=True)
    def forward(self, x):
        return self.linear(x)


class Transformer(nn.Module):
    def __init__(self,
            num_types,
            d_model=36,
            d_rnn=256,
            d_inner=128,
            n_layers=4,
            n_head=4,
            d_k=16,
            d_v=16,
            dropout=0.1,
            n_mix=5,
            model_type="mixture"):
        super().__init__()

        self.encoder = Encoder(
            num_types = num_types,
            d_model = d_model,
            d_inner = d_inner,
            n_layers = n_layers,
            n_head = n_head,
            d_k = d_k,
            d_v = d_v,
            dropout=dropout,
        )

        self.num_types = num_types
        self.n_mix = n_mix
        self.model_type = model_type

        # Per-dataset z-score stats of the baseline temporal heads, filled from
        # the train split. A plain attribute rather than a buffer, so it stays
        # out of state_dict; None means raw days.
        self.time_stats = None

        # intensity parameters
        self.linear = nn.Linear(d_model, num_types)
        self.alpha = nn.Parameter(torch.tensor(-0.1))
        self.beta = nn.Parameter(torch.tensor(1.0))

        self.rnn = RNN_layers(d_model, d_rnn)

        if model_type in ("baseline", "baseline_act"):
            # Baseline: direct regression heads, plus a categorical next
            # activity head for baseline_act.
            self.time_predictor = Predictor(d_model, 1)
            self.rt_predictor = Predictor(d_model, 1)
            if model_type == "baseline_act":
                self.mark_head = nn.Linear(d_model, num_types)
        else:
            # Mixture: zero-inflated LogNormal mixture and mark CE
            self.tie_head = nn.Linear(d_model, 1)
            self.time_head = nn.Linear(d_model, 3 * n_mix)
            self.mark_head = nn.Linear(d_model, num_types)


    def forward(self, event_type, event_time):
        """Return the hidden representations and the head outputs:

            baseline      enc_output, (time_pred, rt_pred)
            baseline_act  enc_output, (time_pred, rt_pred, mark_logit)
            mixture       enc_output, (tie_logit, time_params, mark_logit)
        """

        non_pad_mask = get_non_pad_mask(event_type)

        enc_output = self.encoder(event_type, event_time, non_pad_mask)
        enc_output = self.rnn(enc_output, non_pad_mask)

        if self.model_type in ("baseline", "baseline_act"):
            time_pred = self.time_predictor(enc_output)      # [B, S, 1]
            rt_pred = self.rt_predictor(enc_output)          # [B, S, 1]
            if self.model_type == "baseline_act":
                mark_logit = self.mark_head(enc_output)      # [B, S, num_types]
                return enc_output, (time_pred, rt_pred, mark_logit)
            return enc_output, (time_pred, rt_pred)
        else:
            tie_logit = self.tie_head(enc_output).squeeze(-1)
            time_params = self.time_head(enc_output)
            mark_logit = self.mark_head(enc_output)
            return enc_output, (tie_logit, time_params, mark_logit)