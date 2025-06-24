import torch
import torch.nn as nn
import numpy as np
from prismatic.vla.constants import ACTION_CHUNK_PER_CURVE, NUM_ACTIONS_CHUNK, ACTION_DIM, TOKEN_SEQUENCE_LINE,PROPRIO_DIM, STOP_INDEX, BEZIER_CURVES,Debug
from prismatic.vla.datasets.DataProcess.BezierProcess import QuadraticBezier

class MLP_Action_Actionhead(nn.Module):
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
                 continuous_dim=ACTION_DIM * TOKEN_SEQUENCE_LINE,
                 token_seq_len=0,
                 dropout=0.1):
        super(MLP_Action_Actionhead, self).__init__()

        self.fc1 = nn.Linear(input_dim * (ACTION_CHUNK_PER_CURVE + 1), mlp_hidden_size)
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
        B, seq_len, hidden_dim = actions_hidden_states.size()
        # 把 (B, seq_len, hidden_dim) 展平成 (B, seq_len*hidden_dim)
        x = actions_hidden_states.view(B, seq_len * hidden_dim)
        out = self.forward(x)               
        out = out.view(B, TOKEN_SEQUENCE_LINE, ACTION_DIM)
        return out


