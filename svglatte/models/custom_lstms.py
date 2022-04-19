# Adapted from (diff the two files to see the changes made):
# https://github.com/SapienzaNLP/unify-srl/blob/75ad47ec5327a450b85a57154431638a7360b296/srl/layers/sequence_encoder.py
import pdb

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class SequenceEncoder(nn.Module):

    def __init__(self, encoder_type, latte_ingredients='hc', **kwargs):
        super(SequenceEncoder, self).__init__()

        self._latte_ingredients = latte_ingredients.lower()
        if self._latte_ingredients not in ["h", "c", "hc"]:
            raise Exception(f"latte_ingredients '{self._latte_ingredients}' not supported")

        if encoder_type == 'lstm':
            self.sequence_encoder = StackedBiLSTM(
                kwargs['lstm_input_size'],
                kwargs['lstm_hidden_size'],
                kwargs['lstm_num_layers'],
                kwargs['lstm_dropout'],
                kwargs['lstm_bidirectional'],
                kwargs['pack_sequences']
            )
        elif encoder_type == 'fc_lstm_original':
            self.sequence_encoder = FullyConnectedBiLSTMOriginal(
                kwargs['lstm_input_size'],
                kwargs['lstm_hidden_size'],
                kwargs['lstm_num_layers'],
                kwargs['lstm_dropout'],
                kwargs['lstm_bidirectional'],
                kwargs['pack_sequences']
            )
        elif encoder_type == 'fc_lstm':
            self.sequence_encoder = FullyConnectedBiLSTM(
                kwargs['lstm_input_size'],
                kwargs['lstm_hidden_size'],
                kwargs['lstm_num_layers'],
                kwargs['lstm_dropout'],
                kwargs['lstm_bidirectional'],
                kwargs['pack_sequences']
            )
        elif encoder_type == 'residual_lstm':
            self.sequence_encoder = ResidualBiLSTM(
                kwargs['lstm_input_size'],
                kwargs['lstm_hidden_size'],
                kwargs['lstm_num_layers'],
                kwargs['lstm_dropout'],
                kwargs['lstm_bidirectional'],
                kwargs['pack_sequences']
            )
        elif encoder_type == 'lstm+mha':
            self.sequence_encoder = AttentiveFullyConnectedBiLSTM(
                kwargs['lstm_input_size'],
                kwargs['lstm_hidden_size'],
                kwargs['lstm_num_layers'],
                kwargs['lstm_dropout'],
                kwargs['lstm_bidirectional'],
                kwargs['pack_sequences'],
                kwargs['mha_num_layers'],
                kwargs['mha_hidden_size'],
                kwargs['mha_num_heads'],
                kwargs['mha_dropout'],
            )
        else:
            raise Exception(f"Encoder type '{encoder_type}' not supported")

        self.output_size = 0
        if 'h' in self._latte_ingredients:
            self.output_size += self.sequence_encoder.hidden_output_size
        if 'c' in self._latte_ingredients:
            self.output_size += self.sequence_encoder.cell_output_size

    def forward(self, input_sequences, sequence_lengths=None):
        _, (h, c) = self.sequence_encoder(input_sequences, sequence_lengths)
        h, c = h[-1], c[-1]  # take only the topmost layer

        if self._latte_ingredients == "h":
            latte = h
        elif self._latte_ingredients == "c":
            latte = c
        else:
            latte = torch.cat([h, c], dim=-1)

        assert latte.shape[-1] == self.output_size
        return latte


class StackedBiLSTM(nn.Module):

    def __init__(
            self,
            lstm_input_size,
            lstm_hidden_size,
            lstm_num_layers,
            lstm_dropout,
            lstm_bidirectional,
            pack_sequences,
    ):
        super(StackedBiLSTM, self).__init__()

        self.pack_sequences = pack_sequences
        self.bidirectional = lstm_bidirectional
        self.layer_norm = nn.LayerNorm(lstm_input_size)

        self.lstm = nn.LSTM(
            lstm_input_size,
            lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout,
            bidirectional=lstm_bidirectional,
            batch_first=True)

        if lstm_bidirectional:
            self.hidden_output_size = self.cell_output_size = 2 * lstm_hidden_size
        else:
            self.hidden_output_size = self.cell_output_size = lstm_hidden_size

    def forward(self, input_sequences, sequence_lengths=None):

        total_length = input_sequences.shape[1]
        input_sequences = self.layer_norm(input_sequences)

        if self.pack_sequences:
            packed_input = pack_padded_sequence(
                input_sequences,
                sequence_lengths,
                batch_first=True,
                enforce_sorted=False)
        else:
            packed_input = input_sequences

        packed_sequence_encodings, (h, c) = self.lstm(packed_input)

        if self.pack_sequences:
            sequence_encodings, _ = pad_packed_sequence(
                packed_sequence_encodings,
                total_length=total_length,
                batch_first=True)
        else:
            sequence_encodings = packed_sequence_encodings

        if self.bidirectional:
            h = h.reshape(h.shape[0] // 2, h.shape[1], h.shape[2] * 2)
            c = c.reshape(c.shape[0] // 2, c.shape[1], c.shape[2] * 2)
        return sequence_encodings, (h, c)


class FullyConnectedBiLSTMOriginal(nn.Module):

    def __init__(
            self,
            lstm_input_size,
            lstm_hidden_size,
            lstm_num_layers,
            lstm_dropout,
            lstm_bidirectional,
            pack_sequences,
    ):
        super(FullyConnectedBiLSTMOriginal, self).__init__()

        self.pack_sequences = pack_sequences
        self.bidirectional = lstm_bidirectional

        _lstms = []
        _norms = []
        _drops = []
        _layer_input_size = lstm_input_size
        for _ in range(lstm_num_layers):
            lstm = nn.LSTM(
                _layer_input_size,
                lstm_hidden_size,
                bidirectional=lstm_bidirectional,
                batch_first=True)
            norm = nn.LayerNorm(_layer_input_size)
            drop = nn.Dropout(lstm_dropout)

            _layer_input_size += 2 * lstm_hidden_size if lstm_bidirectional else lstm_hidden_size
            _lstms.append(lstm)
            _norms.append(norm)
            _drops.append(drop)

        self.lstms = nn.ModuleList(_lstms)
        self.norms = nn.ModuleList(_norms)
        self.drops = nn.ModuleList(_drops)

        if lstm_bidirectional:
            self.hidden_output_size = self.cell_output_size = lstm_num_layers * (2 * lstm_hidden_size)
        else:
            self.hidden_output_size = self.cell_output_size = lstm_num_layers * lstm_hidden_size

    def forward(self, input_sequences, sequence_lengths=None):
        batch_size = input_sequences.shape[0]
        total_length = input_sequences.shape[1]

        h = input_sequences.new_empty((1, batch_size, 0))
        c = input_sequences.new_empty((1, batch_size, 0))

        for lstm, drop, norm in zip(self.lstms, self.drops, self.norms):
            normalized_input_sequences = norm(input_sequences)

            if self.pack_sequences:
                packed_input = pack_padded_sequence(
                    normalized_input_sequences,
                    sequence_lengths,
                    batch_first=True,
                    enforce_sorted=False)
            else:
                packed_input = input_sequences

            packed_sequence_encodings, (h_layer, c_layer) = lstm(packed_input)

            h = torch.cat([h, h_layer], dim=-1)
            c = torch.cat([c, c_layer], dim=-1)

            if self.pack_sequences:
                sequence_encodings, _ = pad_packed_sequence(
                    packed_sequence_encodings,
                    total_length=total_length,
                    batch_first=True)
            else:
                sequence_encodings = packed_sequence_encodings

            sequence_encodings = drop(sequence_encodings)
            input_sequences = torch.cat([input_sequences, sequence_encodings], dim=-1)

        output_sequences = input_sequences

        if self.bidirectional:
            h = h.reshape(h.shape[0] // 2, h.shape[1], h.shape[2] * 2)
            c = c.reshape(c.shape[0] // 2, c.shape[1], c.shape[2] * 2)
        return output_sequences, (h, c)


class FullyConnectedBiLSTM(nn.Module):

    def __init__(
            self,
            lstm_input_size,
            lstm_hidden_size,
            lstm_num_layers,
            lstm_dropout,
            lstm_bidirectional,
            pack_sequences,
    ):
        super(FullyConnectedBiLSTM, self).__init__()

        self.pack_sequences = pack_sequences
        self.bidirectional = lstm_bidirectional

        _lstms = []
        _norms = []
        _drops = []
        _layer_input_size = lstm_input_size
        for _ in range(lstm_num_layers):
            lstm = nn.LSTM(
                _layer_input_size,
                lstm_hidden_size,
                bidirectional=lstm_bidirectional,
                batch_first=True)
            norm = nn.LayerNorm(_layer_input_size)
            drop = nn.Dropout(lstm_dropout)

            _layer_input_size += 2 * lstm_hidden_size if lstm_bidirectional else lstm_hidden_size
            _lstms.append(lstm)
            _norms.append(norm)
            _drops.append(drop)

        self.lstms = nn.ModuleList(_lstms)
        self.norms = nn.ModuleList(_norms)
        self.drops = nn.ModuleList(_drops)

        if lstm_bidirectional:
            self.hidden_output_size = self.cell_output_size = lstm_num_layers * (2 * lstm_hidden_size)
        else:
            self.hidden_output_size = self.cell_output_size = lstm_num_layers * lstm_hidden_size

    def forward(self, input_sequences, sequence_lengths=None):
        batch_size = input_sequences.shape[0]
        total_length = input_sequences.shape[1]

        h = input_sequences.new_empty((2 if self.bidirectional else 1, batch_size, 0))
        c = input_sequences.new_empty((2 if self.bidirectional else 1, batch_size, 0))

        for lstm, drop, norm in zip(self.lstms, self.drops, self.norms):
            # TODO I turned the order of packing and normalization around compared to the original as I think that
            #      normalization should not take padding into account. Will evaluate both.
            if self.pack_sequences:
                packed_input = pack_padded_sequence(
                    input_sequences,
                    sequence_lengths,
                    batch_first=True,
                    enforce_sorted=False)
                packed_input = torch.nn.utils.rnn.PackedSequence(
                    norm(packed_input.data),
                    packed_input.batch_sizes,
                    packed_input.sorted_indices,
                    packed_input.unsorted_indices
                )
            else:
                packed_input = input_sequences
                packed_input = norm(packed_input)

            packed_sequence_encodings, (h_layer, c_layer) = lstm(packed_input)
            h = torch.cat([h, h_layer], dim=-1)
            c = torch.cat([c, c_layer], dim=-1)

            if self.pack_sequences:
                sequence_encodings, _ = pad_packed_sequence(
                    packed_sequence_encodings,
                    total_length=total_length,
                    batch_first=True)
            else:
                sequence_encodings = packed_sequence_encodings

            sequence_encodings = drop(sequence_encodings)
            input_sequences = torch.cat([input_sequences, sequence_encodings], dim=-1)

        output_sequences = input_sequences
        
        if self.bidirectional:
            h = h.reshape(h.shape[0] // 2, h.shape[1], h.shape[2] * 2)
            c = c.reshape(c.shape[0] // 2, c.shape[1], c.shape[2] * 2)
        return output_sequences, (h, c)


class ResidualBiLSTM(nn.Module):

    def __init__(
            self,
            lstm_input_size,
            lstm_hidden_size,
            lstm_num_layers,
            lstm_dropout,
            lstm_bidirectional,
            pack_sequences,
    ):
        super(ResidualBiLSTM, self).__init__()
        self.bidirectional = lstm_bidirectional

        self.pack_sequences = pack_sequences
        if lstm_bidirectional:
            self.input_projection = nn.Linear(lstm_input_size, 2 * lstm_hidden_size)
            self.input_norm = nn.LayerNorm(2 * lstm_hidden_size)
        else:
            self.input_projection = nn.Linear(lstm_input_size, lstm_hidden_size)
            self.input_norm = nn.LayerNorm(lstm_hidden_size)

        _lstms = []
        _norms = []
        _drops = []
        _layer_input_size = 2 * lstm_hidden_size if lstm_bidirectional else lstm_hidden_size
        for i in range(lstm_num_layers):
            lstm = nn.LSTM(
                _layer_input_size,
                lstm_hidden_size,
                bidirectional=lstm_bidirectional,
                batch_first=True)
            if lstm_bidirectional:
                norm = nn.LayerNorm(2 * lstm_hidden_size)
            else:
                norm = nn.LayerNorm(lstm_hidden_size)
            drop = nn.Dropout(lstm_dropout)

            _layer_input_size = 2 * lstm_hidden_size if lstm_bidirectional else lstm_hidden_size
            _lstms.append(lstm)
            _norms.append(norm)
            _drops.append(drop)

        self.lstms = nn.ModuleList(_lstms)
        self.norms = nn.ModuleList(_norms)
        self.drops = nn.ModuleList(_drops)

        if lstm_bidirectional:
            self.hidden_output_size = self.cell_output_size = 2 * lstm_hidden_size
        else:
            self.hidden_output_size = self.cell_output_size = lstm_hidden_size

    def forward(self, input_sequences, sequence_lengths=None):

        total_length = input_sequences.shape[1]
        input_sequences = self.input_projection(input_sequences)
        input_sequences = self.input_norm(input_sequences)

        for lstm, drop, norm in zip(self.lstms, self.drops, self.norms):
            if self.pack_sequences:
                packed_input = pack_padded_sequence(
                    input_sequences,
                    sequence_lengths,
                    batch_first=True,
                    enforce_sorted=False)
            else:
                packed_input = input_sequences

            packed_sequence_encodings, (h, c) = lstm(packed_input)

            if self.pack_sequences:
                sequence_encodings, _ = pad_packed_sequence(
                    packed_sequence_encodings,
                    total_length=total_length,
                    batch_first=True)
            else:
                sequence_encodings = packed_sequence_encodings

            sequence_encodings = norm(sequence_encodings)
            sequence_encodings = sequence_encodings + input_sequences
            sequence_encodings = drop(sequence_encodings)
            input_sequences = sequence_encodings

        if self.bidirectional:
            h = h.reshape(h.shape[0] // 2, h.shape[1], h.shape[2] * 2)
            c = c.reshape(c.shape[0] // 2, c.shape[1], c.shape[2] * 2)
        return sequence_encodings, (h, c)


class AttentiveFullyConnectedBiLSTM(nn.Module):

    def __init__(
            self,
            lstm_input_size,
            lstm_hidden_size,
            lstm_num_layers,
            lstm_dropout,
            lstm_bidirectional,
            pack_sequences,
            mha_num_layers,
            mha_hidden_size,
            mha_num_heads,
            mha_dropout,
    ):
        super(AttentiveFullyConnectedBiLSTM, self).__init__()

        self.pack_sequences = pack_sequences
        self.input_norm = nn.LayerNorm(lstm_input_size)

        _lstms = []
        _norms = []
        _drops = []
        _layer_input_size = lstm_input_size
        for i in range(lstm_num_layers):
            lstm = nn.LSTM(
                _layer_input_size,
                lstm_hidden_size,
                bidirectional=lstm_bidirectional,
                batch_first=True)
            norm = nn.LayerNorm(2 * lstm_hidden_size)
            drop = nn.Dropout(lstm_dropout)

            _layer_input_size = 2 * lstm_hidden_size if lstm_bidirectional else lstm_hidden_size
            _lstms.append(lstm)
            _norms.append(norm)
            _drops.append(drop)

        self.lstms = nn.ModuleList(_lstms)
        self.norms = nn.ModuleList(_norms)
        self.drops = nn.ModuleList(_drops)

        if lstm_bidirectional:
            self.hidden_output_size = self.cell_output_size = 2 * lstm_hidden_size
        else:
            self.hidden_output_size = self.cell_output_size = lstm_hidden_size

        _queries = []
        _keys = []
        _values = []
        _mhas = []
        for i in range(mha_num_layers):
            query = nn.Linear(lstm_hidden_size, mha_hidden_size)
            key = nn.Linear(lstm_hidden_size, mha_hidden_size)
            value = nn.Linear(lstm_hidden_size, mha_hidden_size)
            mha = nn.MultiheadAttention(mha_hidden_size, mha_num_heads, dropout=mha_dropout)
            _queries.append(query)
            _keys.append(key)
            _values.append(value)
            _mhas.append(mha)

        self.queries = nn.ModuleList(_queries)
        self.keys = nn.ModuleList(_keys)
        self.values = nn.ModuleList(_values)
        self.mhas = nn.ModuleList(_mhas)

        if lstm_bidirectional:
            self.hidden_output_size = self.cell_output_size = 2 * lstm_hidden_size
        else:
            self.hidden_output_size = self.cell_output_size = lstm_hidden_size

    def forward(self, input_sequences, sequence_lengths=None):

        total_length = input_sequences.shape[1]
        input_sequences = self.input_norm(input_sequences)
        output_sequences = input_sequences

        for lstm, drop, norm in zip(self.lstms, self.drops, self.norms):
            if self.pack_sequences:
                packed_input = pack_padded_sequence(
                    input_sequences,
                    sequence_lengths,
                    batch_first=True,
                    enforce_sorted=False)
            else:
                packed_input = input_sequences

            packed_sequence_encodings, _ = lstm(packed_input)

            if self.pack_sequences:
                sequence_encodings, _ = pad_packed_sequence(
                    packed_sequence_encodings,
                    total_length=total_length,
                    batch_first=True)
            else:
                sequence_encodings = packed_sequence_encodings

            sequence_encodings = drop(sequence_encodings)
            sequence_encodings = norm(sequence_encodings)
            input_sequences = sequence_encodings
            output_sequences = torch.cat([output_sequences, sequence_encodings], dim=-1)

        batch_size = output_sequences.shape[0]
        max_length = output_sequences.shape[1]
        input_mask = torch.arange(max_length).expand(batch_size, max_length).to(input_sequences.device)
        input_mask = input_mask >= sequence_lengths.to(input_sequences.device).unsqueeze(1)

        input_sequences = torch.transpose(output_sequences, 0, 1)
        for wq, wk, wv, mha in zip(self.queries, self.keys, self.values, self.mhas):
            q = wq(input_sequences)
            k = wk(input_sequences)
            v = wv(input_sequences)
            output_sequences, _ = mha(q, k, v, key_padding_mask=input_mask)
            input_sequences = output_sequences
        output_sequences = torch.transpose(output_sequences, 0, 1)

        pdb.set_trace()
        return output_sequences, (h, c)
