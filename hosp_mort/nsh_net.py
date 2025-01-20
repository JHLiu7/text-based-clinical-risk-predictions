import torch, math, copy
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Tuple, Optional



class NSHNet(nn.Module):
    def __init__(self, config, N, note2patient_expert=True) -> None:
        """
        Initializes the NSHNet model with a note encoder, patient encoder, and fully connected layer.
        Args:
            config (dict): Configuration dictionary containing model hyperparameters.
            N (int): Number of layers for the patient encoder.
            note2patient_expert (bool, optional): Flag to use expert knowledge in note to patient encoding. Defaults to True.
        Attributes:
            note_encoder (Token_NoteEncoder_CNN): Encoder for processing notes.
            encode_dim (int): Dimension of the encoded notes.
            patient_encoder (Note_PatientEncoder): Encoder for processing patient/patient information.
            fc (nn.Linear): Fully connected layer for final classification with mean pooling.
        """
        super().__init__()


        config = copy.deepcopy(config)

        # note encoder 
        self.note_encoder = Token_NoteEncoder_CNN(
            filter_num=config.nitre_filter_num,
            filter_sizes=config.nitre_filter_sizes,
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            dropout=config.dropout
        )

        self.encode_dim = self.note_encoder.encode_dim

        # patient encoder 
        self.patient_encoder = Note_PatientEncoder(
            hidden_size=self.encode_dim,
            dropout=config.nitre_dropout,
            use_expert=note2patient_expert,
            N=N,
            num_patient_encoder_layers=config.nitre_num_layers
        )

        # final layer w/ mean pooling
        self.fc = nn.Linear(self.encode_dim, 2)

    def forward(self, note_input, note_length, note_position, note_type, **kwargs):
        """
        Forward pass of the neural network.
        Args:
            note_input (Tensor): Input tensor containing note data.
            note_length (Tensor): Tensor containing the lengths of the notes.
            note_position (Tensor): Tensor containing the positions of the notes.
            note_type (Tensor): Tensor containing the types of the notes.
            **kwargs: Additional keyword arguments.
        Returns:
            Tensor: Logits output from the fully connected layer.
        """
        
        x = self.note_encoder(note_input, note_length, note_type)

        x = order_notes(x, note_position)

        x = self.patient_encoder(x, note_position, note_type)

        logits = self.fc(x.mean(1))

        return logits


class Note_PatientEncoder(nn.Module):
    def __init__(self, hidden_size, dropout, use_expert, N, num_patient_encoder_layers) -> None:
        """
        Initializes the neural network model.
        Args:
            hidden_size (int): The size of the hidden layers.
            dropout (float): The dropout rate to be applied.
            use_expert (bool): Flag to indicate whether to use expert layers.
            N (int): The number of experts.
            num_patient_encoder_layers (int): The number of PatientEncoder layers to be used.
        Attributes:
            layers (nn.ModuleList): A list of PatientEncoderLayer modules.
            pe (torch.Tensor): Positional encoding tensor with sinusoidal embeddings.
        """
        super().__init__()


        # layers 

        num_attention_heads =  int(hidden_size / 64)
        dim_feedforward = int(hidden_size * 4)

        self.layers = nn.ModuleList([
            PatientEncoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                use_expert=use_expert,
                N=N
            ) for _ in range(num_patient_encoder_layers)
        ])

        # pos embedding: sinusoid

        d_model = hidden_size
        max_position_embeddings = 2048
        pe = torch.zeros(max_position_embeddings, d_model)
        position = torch.arange(0, max_position_embeddings, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe, persistent=False)


    def forward(self, x, note_pos, note_type_pos):
        """Forward

        Args:
            x (Tensor): note_input, (batch, num_note, dim)
            note_pos (Tensor): Pos/order for note
            note_type_pos (Tensor): Position for note types
            time_delta (Tensor): time diff normalized, (batch, num_note)
        """

        x = x + self.pe[:, :x.size(1)]

        for layer in self.layers:
            x = layer(x, note_pos, note_type_pos)

        return x


class PatientEncoderLayer(nn.Module):
    def __init__(self, 
                hidden_size, 
                num_attention_heads, 
                dim_feedforward,
                dropout,
                use_expert=True,
                N=None,
                activation='gelu',
                layer_norm_eps=1e-5,
                pre_norm=True
                ) -> None:
        """
        Initializes the PatientEncoderLayer module.
        Args:
            hidden_size (int): The size of the hidden layers.
            num_attention_heads (int): The number of attention heads.
            dim_feedforward (int): The size of the feedforward layers.
            dropout (float): The dropout rate to be applied.
            use_expert (bool): Flag to indicate whether to use expert layers.
            N (int): The number of experts.
            activation (str): The activation function to be used.
            layer_norm_eps (float): The epsilon value for layer normalization.
            pre_norm (bool): Flag to indicate whether to use pre-normalization.
        """
        super().__init__()

        self.mha = MultiHeadAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=dropout
        )

        if use_expert:
            self.ff = FFNoteExpert(
                N=N,
                hidden_size=hidden_size,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation
            )
            self.reorder_required = True

        else:
            self.ff = FFDefault(
                hidden_size=hidden_size,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation
            )
            self.reorder_required = False


        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)


        self.pre_norm = pre_norm
        self.hidden_size = hidden_size


    def forward(self, x, note_pos, note_type_pos):
        """Forward

        Args:
            x (Tensor): note_input, (batch, num_note, dim)
            note_pos (Tensor): Pos/order for note
            note_type_pos (Tensor): Position for note types

        """

        attn_mask = (note_pos != 0).int() # 1: attend; 2: ignore

        if self.pre_norm:
            tmp_x = self.mha(
                hidden_states=self.norm1(x), 
                attention_mask=attn_mask, 
                output_attentions=False
            )[0]
            x = x + tmp_x

            if self.reorder_required:
                # remove padding, x.shape -> (num_valid_note, hidden)
                x = x.view(-1, self.hidden_size)[note_pos.flatten() != 0]

            tmp_x = self.ff(
                self.norm2(x), note_type_pos
            )
            x = x + tmp_x

            if self.reorder_required:
                # resume shape 
                x = order_notes(x, note_pos)

        else:
            raise NotImplementedError

        return x


class FFNoteExpert(nn.Module):
    def __init__(self, N, hidden_size, dim_feedforward, dropout, activation="relu"):
        super().__init__()

        act_fn = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
        }
        
        layer = nn.Sequential(
            nn.Linear(hidden_size, dim_feedforward),
            act_fn[activation],
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_size),
        )
        
        self.N = N
        self.layers = _get_clones(layer, N)

        
    def forward(self, x, note_type_pos):
        # x: (num_note, dim)
        # note_type_pos: (num_note, )
        
        assert note_type_pos.dim() == 1
        assert x.dim() == 2 
        assert note_type_pos.max() < self.N 

        layer_to_use = note_type_pos.unique()
        # start from 0

        x_out = torch.zeros_like(x)
        for layer in layer_to_use:
            mask = note_type_pos == layer

            x_type = x[mask]
            x_tmp = self.layers[layer](x_type)

            x_out[mask] = x_tmp

        return x_out


class FFDefault(nn.Module):
    def __init__(self, hidden_size, dim_feedforward, dropout, activation="relu"):
        super().__init__()

        act_fn = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
        }

        self.layer = nn.Sequential(
            nn.Linear(hidden_size, dim_feedforward),
            act_fn[activation],
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_size),
        )

    def forward(self, x, *args):
        # x: (batch, len, dim)

        assert x.dim() == 3

        return self.layer(x)
    


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads,
            attention_probs_dropout_prob, max_position_embeddings=2048,
            position_embedding_type=None, cls_token_appended=False):
        """Based on BertSelfAttention
        """
        super().__init__()

        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention "
                f"heads ({num_attention_heads})"
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

        self.init_parameters()

        self.position_embedding_type = position_embedding_type
        self.max_position_embeddings = max_position_embeddings
        self.cls_token_appended = cls_token_appended # skip relative embed for cls


    def init_parameters(self):
        for layer in [self.query, self.key, self.value]:
            _init_weights(layer)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        time_delta: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:

        # self attention
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # apply masking 
        if attention_mask is not None:
            # Apply the attention mask: same pattern as huggingface interface 
            # 1: attend  
            # 0: ignore 
            assert attention_mask.unique().sum() == 1
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * -10000.0

            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
    

class TextBaseModule(nn.Module):
    def __init__(self, vocab_size, embed_dim, dropout) -> None:
        super(TextBaseModule, self).__init__()

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.word_dropout = nn.Dropout(dropout)

    def init_embeddings(self, embeddings):
        # embeddings: np.array
        self.embed.weight = nn.Parameter(torch.from_numpy(embeddings))

    def freeze_embeddings(self, freeze=False):
        self.embed.weight.requires_grad = not freeze


class Token_NoteEncoder_CNN(TextBaseModule):
    def __init__(self, filter_num, filter_sizes, vocab_size, embed_dim, dropout) -> None:
        super().__init__(vocab_size, embed_dim, dropout)

        self.cnn_encoder = CNNAttnEncoder(
            filter_num=filter_num,
            filter_sizes=filter_sizes,
            input_dim=embed_dim
        )

        self.encode_dim = self.cnn_encoder.encode_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, note_lengths=None, *args):
        x = self.embed(x)
        x = self.word_dropout(x)

        x, attn_wts = self.cnn_encoder(x, note_lengths)

        return x

class CNNAttnEncoder(nn.Module):
    def __init__(self, filter_num, filter_sizes, input_dim) -> None:
        super().__init__()

        kernels = [int(k) for k in filter_sizes.split(',')]

        self.convs = nn.ModuleList([
            nn.Conv1d(input_dim, filter_num, kernel_size=k, padding=int(math.floor( k / 2)))
            for k in kernels
        ])

        self.encode_dim = len(kernels) * filter_num

        self.attention = AttentionWithContext(self.encode_dim, self.encode_dim)

    def forward(self, x, note_lengths=None):

        x = x.transpose(1, 2)
        x = torch.cat([torch.relu(conv(x)) for conv in self.convs], 1)
        x = x.transpose(1, 2)
        x, attn_wts = self.attention(x, note_lengths)

        return x, attn_wts


class AttentionWithContext(nn.Module):
    def __init__(self, input_size, attention_size, context_dim=1):
        super(AttentionWithContext, self).__init__()
        
        self.linear = nn.Linear(input_size, attention_size)
        self.context= nn.Linear(attention_size, context_dim)

        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.context.weight)
        
    def _masked_softmax(self, att, seq_len):
        """
        att: (num_seq, pad_seq_dim)
        seq_len: (num_seq,)
        """
        # index = torch.arange(0, int(seq_len.max())).unsqueeze(1).type_as(att)
        index = torch.arange(0, att.size(-1)).unsqueeze(1).type_as(att)
        seq_len = seq_len.type_as(att)
        # print(index, seq_len)

        mask = (index < seq_len.unsqueeze(0))

        score = torch.exp(att) * mask.transpose(0, 1)
        dist = score / torch.sum(score, dim=-1, keepdim=True)

        return dist
        
    def forward(self, seq_enc, seq_len=None):
        
        att = torch.tanh(self.linear(seq_enc))
        att = self.context(att).squeeze(-1)
        
        if seq_len is not None:
            score = self._masked_softmax(att, seq_len)
        else:
            score = torch.softmax(att, dim=-1)
        # return score
        enc_weighted = score.unsqueeze(-1) * seq_enc
        
        return enc_weighted.sum(1), score


def order_notes(rep_notes, patient_order):
    """
    patient_order: (num_patient, padded_note_len)
    """
    
    rep_notes = F.pad(rep_notes, (0,0,1,0))
    notes = rep_notes[patient_order.view(-1)]
    notes = notes.view(patient_order.size(0), patient_order.size(1), notes.size(-1))
    
    return notes


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _init_weights(module, initializer_range=0.02):
    """Initialize the weights"""
    if isinstance(module, nn.Linear):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=initializer_range)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


