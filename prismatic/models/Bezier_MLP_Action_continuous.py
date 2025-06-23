import torch
import torch.nn as nn
import numpy as np
from prismatic.vla.constants import ACTION_CHUNK_PER_CURVE, NUM_ACTIONS_CHUNK, ACTION_DIM, TOKEN_SEQUENCE_LINE,PROPRIO_DIM, STOP_INDEX, BEZIER_CURVES,Debug
from prismatic.vla.datasets.DataProcess.BezierProcess import QuadraticBezier

class Bezier_MLP_Action_continuous(nn.Module):
    """
    Three-layer action decoding MLP that outputs:
      - 14 continuous float values
      - 30 discrete tokens (highest probability selected)
    """
    def __init__(self,
                 input_dim=4096,
                 action_dim=ACTION_DIM,
                 mlp_hidden_size=5120,# 4096
                 decoder_hidden_size=2048, 
                 continuous_dim=ACTION_DIM * ACTION_CHUNK_PER_CURVE,
                 token_seq_len=1,
                 dropout=0.1):
        super(Bezier_MLP_Action_continuous, self).__init__()

        self.fc1 = nn.Linear(input_dim * ACTION_CHUNK_PER_CURVE, mlp_hidden_size)
        self.fc2 = nn.Linear(mlp_hidden_size, mlp_hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Decoder MLP: three linear layers
        total_output = continuous_dim + token_seq_len
        self.decoder = nn.Sequential(
            nn.Linear(mlp_hidden_size, decoder_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden_size, decoder_hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden_size//2, total_output)
        )

        self.placeholder = nn.Parameter(torch.randn(1, input_dim), requires_grad=True)  # 用于预测第一个P0的时候进行拼接

        # Store dims
        self.continuous_dim = continuous_dim
        self.token_seq_len = token_seq_len

    def forward(self, x):
        """
        x: Tensor(shape=(N, hidden_dim))
        returns: Tensor(shape=(N, total_dim))
        """
        residual = self.fc1(x)
        out = self.relu(residual)
        out = self.dropout(out)
        out = self.fc2(out)
        
        # # 添加残差连接
        # out = out + residual
        # out = self.relu(out)

        out = self.decoder(out)
        return out


    def predict_action(self, actions_hidden_states):
        B, orig_seq_len, hidden_dim = actions_hidden_states.size()  # orig_seq_len=3
        # 在前面加一个 placeholder，得到新的 seq_len=orig_seq_len+1
        placeholder_expanded = self.placeholder.expand(B, 1, hidden_dim)  # (B,1,hidden_dim)
        x = torch.cat([placeholder_expanded, actions_hidden_states], dim=1)  # (B,4,hidden_dim)

        seq_len = BEZIER_CURVES + 1

        # flatten 为 batch 中的每个 time step
        B_new, _, _ = x.size()  # seq_len=4
        Debug("seq_len: " + str(seq_len))
        x = x.reshape(B_new * seq_len, hidden_dim * ACTION_CHUNK_PER_CURVE)  # (B*4, hidden_dim)

        # forward，假设 fc1 已经是 nn.Linear(hidden_dim, ...)
        out = self.forward(x)                       # (B*4, total_dim)
        out = out.view(B_new, seq_len, -1)          # (B,4,total_dim)

        # 以下与原来拆控制点、长度、拼 combined 的逻辑不变
        cd = self.continuous_dim
        pt_dim = cd // 2

        P1 = out[..., :pt_dim]
        P2 = out[..., pt_dim:2*pt_dim]

        P0 = P2[:,:-1,:]
        P1 = P1[:,1:,:]
        P2 = P2[:,1:,:]


        # 4) 长度预测
        length_token = out[..., cd]             # (B, seq_len, num_length_classes)
        lengths = (0.5 + length_token) *  TOKEN_SEQUENCE_LINE
        # 把所有小于 1 的值都置为 1
        # lengths = torch.clamp(lengths, min=1.001,max = 2 -1 + 0.002)# test should be TOKEN_SEQUENCE_LINE - 1
        lengths = torch.clamp(lengths, min=1,max = TOKEN_SEQUENCE_LINE -1)# test should be TOKEN_SEQUENCE_LINE - 1
        lengths = lengths.unsqueeze(-1)

        lengths[:,1,0] = 1 # test delate later 

        lengths = lengths[:,1:,:]

        P1 = - (0.25 * P0 + 0.25 * P2 - P1) * 2 # form dot to line

        combined = torch.cat([P0, P1, P2, lengths], dim=-1)
        return combined

