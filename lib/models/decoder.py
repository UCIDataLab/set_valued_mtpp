import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import xavier_truncated_normal, find_closest

class HawkesDecoder(nn.Module):
    """Decoder module that transforms a set of marks, timestamps, and latent vector into intensity values for different channels."""

    def __init__(
        self,
        channel_embedding,
        time_embedding,
        recurrent_hidden_size,
    ):
        super().__init__()
        self.channel_embedding = channel_embedding
        self.time_embedding = time_embedding
        self.num_channels, self.channel_embedding_size = self.channel_embedding.weight.shape

        self.recurrent_input_size = self.channel_embedding_size + recurrent_hidden_size
        self.recurrent_hidden_size = recurrent_hidden_size
        self.cell_param_network = nn.Linear(self.recurrent_input_size, 7 * recurrent_hidden_size, bias=True)  # Eq. 5a-6c in Neural Hawkes paper
        self.init_hidden_state = nn.Parameter(xavier_truncated_normal(size=(1, 6 * recurrent_hidden_size), no_average=True))

    def get_init_states(self, batch_size):
        init_states = self.init_hidden_state.expand(batch_size, -1)
        h_d, c_d, c_bar, c, delta, o = torch.chunk(init_states, 6, -1)
        return torch.tanh(h_d), torch.tanh(c_d), torch.tanh(c_bar), torch.tanh(c), F.softplus(delta), torch.sigmoid(o)

    def recurrence(self, emb_event_t, h_d_tm1, c_tm1, c_bar_tm1):
        feed = torch.cat((emb_event_t, h_d_tm1), dim=1)
        # B * 2H
        (gate_i,
         gate_f,
         gate_z,
         gate_o,
         gate_i_bar,
         gate_f_bar,
         gate_delta) = torch.chunk(self.cell_param_network(feed), 7, -1)

        gate_i = torch.sigmoid(gate_i)
        gate_f = torch.sigmoid(gate_f)
        gate_z = torch.tanh(gate_z)
        gate_o = torch.sigmoid(gate_o)
        gate_i_bar = torch.sigmoid(gate_i_bar)
        gate_f_bar = torch.sigmoid(gate_f_bar)
        gate_delta = F.softplus(gate_delta)

        c_t = gate_f * c_tm1 + gate_i * gate_z  # Eq.6a
        c_bar_t = gate_f_bar * c_bar_tm1 + gate_i_bar * gate_z  # Eq.6b

        return c_t, c_bar_t, gate_o, gate_delta


    def decay(self, c_t, c_bar_t, o_t, delta_t, duration_t):
        c_d_t = c_bar_t + (c_t - c_bar_t) * torch.exp(-delta_t * duration_t)  # Eq.7
        h_d_t = o_t * torch.tanh(c_d_t)
        return c_d_t, h_d_t


    def get_states(self, marks, timestamps, old_states=None):
        """Produce the set of hidden states from a given set of marks, timestamps, and latent vector that can then be used to calculate intensities.

        Arguments:
            marks {torch.LongTensor} -- Tensor containing mark ids that correspond to channel embeddings.
            timestamps {torch.FloatTensor} -- Tensor containing times of events that correspond to the marks.

        Keyword Arguments:
            latent_state {torch.FloatTensor} -- Latent vector that [hopefully] summarizes relevant point process dynamics from a reference point pattern. (default: {None})

        Returns:
            torch.FloatTensor -- Corresponding hidden states that represent the history of the point process.
        """

        time_deltas = self.time_embedding(timestamps)

        if marks.numel() == 0:
            recurrent_input = self.channel_embedding(torch.LongTensor([[]]))
        else:
            # mark_embeddings_sum = torch.unsqueeze(torch.matmul(torch.squeeze(marks, 0), self.channel_embedding.weight.data), 0)  # torch.Size([batch_size, num_events, embedding_size])
            mark_embeddings_sum = torch.matmul(marks, self.channel_embedding.weight.data)  # torch.Size([batch_size, num_events, embedding_size])
            # use mean value of the embeddings instead of sum
            mark_count = torch.unsqueeze(marks.sum(dim=-1), -1)
            recurrent_input = torch.div(mark_embeddings_sum, torch.clamp(mark_count, min=1))  # to allow empty set

        assert (recurrent_input.shape[-1] == (self.recurrent_input_size - self.recurrent_hidden_size))

        if old_states is None:
            h_d, c_d, c_bar, c, delta_t, o_t = self.get_init_states(time_deltas.shape[0])
        else:
            h_d, o_t, c_bar, c, delta_t, c_d = torch.chunk(old_states, 6, -1)

        # hidden_states = [torch.cat((h_d, o_t, c_bar, c, delta_t), -1)]
        hidden_states = [torch.cat((h_d, o_t, c_bar, c, delta_t, c_d), -1)]
        for i in range(time_deltas.shape[1]):
            r_input, t_input = recurrent_input[:, i, :], time_deltas[:, i, :]

            c, c_bar, o_t, delta_t = self.recurrence(r_input, h_d, c_d, c_bar)
            c_d, h_d = self.decay(c, c_bar, o_t, delta_t, t_input)
            # hidden_states.append(torch.cat((h_d, o_t, c_bar, c, delta_t), -1))
            hidden_states.append(torch.cat((h_d, o_t, c_bar, c, delta_t, c_d), -1))

        hidden_states = torch.stack(hidden_states, dim=1)
        return hidden_states


    def get_hidden_h(self, state_values, state_times, timestamps):
        """Generate the hidden state h(t) for a point process at given time t that can be used to estimate intensities

        Arguments:
            state_values {torch.FloatTensor} -- Output hidden states from `get_states` call.
            state_times {torch.FloatTensor} -- Corresponding timestamps used to generate state_values. These are the "true event times" to be compared against.
            timestamps {torch.FloatTensor} -- Times to generate intensity values for.

        Returns:
            [type] -- [description]
        """
        closest_dict = find_closest(sample_times=timestamps, true_times=state_times)  # both index and time

        padded_state_values = state_values

        selected_hidden_states = padded_state_values.gather(dim=1,
                                                            index=closest_dict["closest_indices"].unsqueeze(-1).expand(
                                                                -1, -1, padded_state_values.shape[-1]))
        time_embedding = self.time_embedding(timestamps, state_times)
        h_d, o_t, c_bar, c, delta_t, _ = torch.chunk(selected_hidden_states, 6, -1)

        _, h_t = self.decay(c, c_bar, o_t, delta_t, time_embedding)
        return h_t



class RMTPPDecoder(nn.Module):
    """Decoder module that transforms a set of marks, timestamps, and latent vector into intensity values for different channels."""

    def __init__(
            self,
            channel_embedding,
            time_embedding,
            recurrent_hidden_size,
    ):
        super().__init__()

        self.channel_embedding = channel_embedding
        self.time_embedding = time_embedding
        self.num_channels, self.channel_embedding_size = self.channel_embedding.weight.shape

        self.recurrent_input_size = self.channel_embedding_size + self.time_embedding.embedding_dim
        self.recurrent_hidden_size = recurrent_hidden_size
        self.recurrent_net = nn.LSTM(
            input_size=self.recurrent_input_size,
            hidden_size=self.recurrent_hidden_size,
            num_layers=1,
            bidirectional=False,
            batch_first=True,
        )
        self.time_to_intensity_w = nn.Parameter(xavier_truncated_normal(size=(1), no_average=True) * 0.001)
        self.register_parameter(
            name="init_hidden_state",
            param=nn.Parameter(xavier_truncated_normal(size=(1, 1, 2 * recurrent_hidden_size), no_average=True))
        )

    def get_init_states(self, batch_size):
        init_states = self.init_hidden_state.expand(1, batch_size, -1)
        h_0, c_0 = torch.chunk(init_states, 2, -1)
        return torch.tanh(h_0), torch.tanh(c_0)

    def get_states(self, marks, timestamps, old_states=None):
        """Produce the set of hidden states from a given set of marks, timestamps, and latent vector that can then be used to calculate intensities.

        Arguments:
            marks {torch.LongTensor} -- Tensor containing mark ids that correspond to channel embeddings.
            timestamps {torch.FloatTensor} -- Tensor containing times of events that correspond to the marks.

        Keyword Arguments:
            latent_state {torch.FloatTensor} -- Latent vector that [hopefully] summarizes relevant point process dynamics from a reference point pattern. (default: {None})

        Returns:
            torch.FloatTensor -- Corresponding hidden states that represent the history of the point process.
        """

        time_deltas = self.time_embedding(timestamps)
        components = []

        #components.append(self.channel_embedding(marks))
        if marks.numel() == 0:
            mark_input = self.channel_embedding(torch.LongTensor([[]]))
        else:
            mark_embeddings_sum = torch.matmul(marks, self.channel_embedding.weight.data)  # torch.Size([batch_size, num_events, embedding_size])
            # use mean value of the embeddings instead of sum
            mark_count = torch.unsqueeze(marks.sum(dim=-1), -1)
            mark_input = torch.div(mark_embeddings_sum, torch.clamp(mark_count, min=1))  # to allow empty set
        components.append(mark_input)

        components.append(time_deltas)

        recurrent_input = torch.cat(components, dim=-1)
        assert (recurrent_input.shape[-1] == (self.recurrent_input_size))

        init_state = self.get_init_states(time_deltas.shape[0])
        hidden_states = [init_state[0].squeeze(0).unsqueeze(1)]
        output_hidden_states, (ohs, ocs) = self.recurrent_net(recurrent_input, init_state)
        hidden_states.append(output_hidden_states)

        hidden_states = torch.cat(hidden_states, dim=1)
        return hidden_states

    def get_hidden_h(self, state_values, state_times, timestamps, mark_mask=1.0):
        """Get decayed hidden states for a point process.

        Arguments:
            state_values {torch.FloatTensor} -- Output hidden states from `get_states` call.
            state_times {torch.FloatTensor} -- Corresponding timestamps used to generate state_values. These are the "true event times" to be compared against.
            timestamps {torch.FloatTensor} -- Times to generate intensity values for.

        Keyword Arguments:
            latent_state {torch.FloatTensor} -- Latent vector that [hopefully] summarizes relevant point process dynamics from a reference point pattern. (default: {None})

        Returns:
            [type] -- [description]
        """
        closest_dict = find_closest(sample_times=timestamps, true_times=state_times)

        padded_state_values = state_values

        selected_hidden_states = padded_state_values.gather(dim=1, index=closest_dict["closest_indices"].unsqueeze(
            -1).expand(-1, -1, padded_state_values.shape[-1]))
        time_embedding = self.time_embedding(timestamps, state_times)
        # time_logits = self.time_to_intensity_logits(time_embedding).sum(dim=-1, keepdims=True)
        time_logits = self.time_to_intensity_w * time_embedding
        return selected_hidden_states , time_logits
