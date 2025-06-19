import torch
import torch.nn as nn
from prismatic.vla.constants import ACTION_CHUNK_PER_CURVE, NUM_ACTIONS_CHUNK, ACTION_DIM, TOKEN_SEQUENCE_LINE,PROPRIO_DIM, STOP_INDEX

class Bezier_MLP_Action(nn.Module):
    """
    Three-layer action decoding MLP that outputs:
      - 14 continuous float values
      - 30 discrete tokens (highest probability selected)
    """
    def __init__(self,
                 input_dim=4096,
                 action_dim=ACTION_DIM,
                 mlp_hidden_size=4096,
                 decoder_hidden_size=1024,
                 continuous_dim=14,
                 token_seq_len=TOKEN_SEQUENCE_LINE,
                 dropout=0.1):
        super(Bezier_MLP_Action, self).__init__()

        # Front-end MLP feature extractor
        self.mlp = nn.Sequential(
            nn.Linear(input_dim* action_dim * ACTION_CHUNK_PER_CURVE, mlp_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size, mlp_hidden_size),
            nn.ReLU()
        )
        
        # Decoder MLP: three linear layers
        total_output = continuous_dim + token_seq_len
        self.decoder = nn.Sequential(
            nn.Linear(mlp_hidden_size, decoder_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden_size, decoder_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden_size, total_output)
        )

        # Store dims
        self.continuous_dim = continuous_dim
        self.token_seq_len = token_seq_len

    def forward(self, x):
        # Extract features
        feats = self.mlp(x)
        # Decode
        out = self.decoder(feats)

        # Split continuous and token logits
        floats = out[:, :self.continuous_dim]
        token_logits = out[:, self.continuous_dim:]
        # Reshape to (batch_size, token_seq_len) since each position has one logit
        token_logits = token_logits.view(-1, self.token_seq_len)
        # 从30个 token_logits 中选取最大值对应的位置（+1 使索引从1开始）
        tokens = torch.argmax(token_logits, dim=-1) + 1
        return floats, tokens

    def predict_action(self, actions_hidden_states):
        # 1) 先提取 batch_size 并 reshape 回 (batch, chunk_len, hidden_dim)
        batch_size = actions_hidden_states.size(0)
        x = actions_hidden_states.view(batch_size, NUM_ACTIONS_CHUNK//ACTION_CHUNK_PER_CURVE, -1)
        is_sequence = (x.ndim == 3)
        # 如果是单步，先统一为 (batch, 1, input_dim)
        if not is_sequence:
            x = x.unsqueeze(1)

        batch_size, seq_len, _ = x.shape
        outputs_out_float = []
        outputs_out_tokens = []

        for t in range(seq_len):
            xt = x[:, t, :]               # (batch, input_dim)
            out_float,out_tokens = self.forward(xt)
            # out_t: (batch, action_dim)
            outputs_out_float.append(out_float)
            outputs_out_tokens.append(out_tokens)


        # 拼成 (batch, seq_len, action_dim)
        actions_float = torch.stack(outputs_out_float, dim=1)
        actions_tokens = torch.stack(outputs_out_tokens, dim=1)

        # 如果原本是单步输入，就 squeeze 掉时间维度
        if not is_sequence:
            actions_float = actions_float.squeeze(1)  # -> (batch, action_dim)
            actions_tokens = actions_tokens.squeeze(1)  # -> (batch, action_dim)

        return actions_float, actions_tokens

