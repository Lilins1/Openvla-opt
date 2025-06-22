import torch
import torch.nn as nn
import numpy as np
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
                 mlp_hidden_size=4096,# 4096
                 decoder_hidden_size=2048, 
                 continuous_dim=ACTION_DIM * ACTION_CHUNK_PER_CURVE,
                 token_seq_len=TOKEN_SEQUENCE_LINE,
                 dropout=0.1):
        super(Bezier_MLP_Action, self).__init__()

        # Front-end MLP feature extractor
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * ACTION_CHUNK_PER_CURVE, mlp_hidden_size),
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
            nn.Linear(decoder_hidden_size, decoder_hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden_size//2, total_output)
        )

        self.placeholder = nn.Parameter(torch.randn(1, input_dim), requires_grad=True)#用于预测第一个P0的时候进行拼接，


        # Store dims
        self.continuous_dim = continuous_dim
        self.token_seq_len = token_seq_len

    # def forward(self, x):
    #     # Extract features
    #     feats = self.mlp(x)
    #     # Decode
    #     out = self.decoder(feats)
    #     line = []
    #     for seg in out:
    #         palceholder = seg[:self.continuous_dim/ACTION_CHUNK_PER_CURVE]
    #         P1 = seg[:self.continuous_dim/ACTION_CHUNK_PER_CURVE]
    #         P2 = seg[self.continuous_dim/ACTION_CHUNK_PER_CURVE:self.continuous_dim]
    #         token_logits = seg[self.continuous_dim:]
    #         line_length = torch.argmax(token_logits, dim=-1) + 1

    #         line.append([palceholder,P1,P2,line_length])

    #     return line
    
    def forward(self, x):
        """
        x: Tensor(shape=(N, hidden_dim))
        returns: Tensor(shape=(N, total_dim))
        """
        feats = self.mlp(x)           # (N, decoder_input_dim)
        out   = self.decoder(feats)   # (N, total_dim)
        return out

    # def predict_action(self, actions_hidden_states):
    #     B = actions_hidden_states.size(0)
    #     seq_len = NUM_ACTIONS_CHUNK // ACTION_CHUNK_PER_CURVE
    #     hidden_dim = actions_hidden_states.size(-1)
    #     out_len = self.continuous_dim + self.token_seq_len

    #     # 1) reshape 到 (B * seq_len, hidden_dim)
    #     x = actions_hidden_states.view(B, seq_len, -1)
    #     x = x.reshape(B * seq_len, hidden_dim * ACTION_CHUNK_PER_CURVE)

    #     # 2) 一次性前向
    #     out = self.forward(x)  # (B*seq_len, total_dim)
    #     out = out.view(B,seq_len,-1)
    #     out = out.reshape(B,seq_len,out_len)

    #     # 3) 拆分出控制点和平移长度 logits
    #     cd = self.continuous_dim
    #     # 每段 Bézier 输出的连续维度是 continuous_dim，
    #     # 其中前三份分别是 P0, P1, P2，每份长度 continuous_dim / 3
    #     pt_dim = cd // 2

    #     out_put_curves  =[]
    #     for out_seg in out:
    #         P0 = torch.zeros_like(out_seg[0, :pt_dim])  # 初始化为 0
    #         curve = []
    #         for seg in out_seg:
    #             P1 = seg[0          : pt_dim]
    #             P2 = seg[pt_dim     : 2*pt_dim]
    #             P0 = P2
    #             token_logits = seg[cd:]     # (B*seq_len, num_length_classes)
    #             # 4) 计算每段的长度预测
    #             lengths = torch.argmax(token_logits, dim=-1) + 1  # (B*seq_len,)
    #             curve.append([P0, P1, P2,lengths])
    #         out_put_curves.append(curve)
                
    #     return out_put_curves

    def predict_action(self, actions_hidden_states):
        B = actions_hidden_states.size(0)
        seq_len = NUM_ACTIONS_CHUNK // ACTION_CHUNK_PER_CURVE
        hidden_dim = actions_hidden_states.size(-1)
        out_len = self.continuous_dim + self.token_seq_len

        # 1) reshape 到 (B * seq_len, hidden_dim * ACTION_CHUNK_PER_CURVE)
        x = actions_hidden_states.view(B, seq_len, -1)
        x = x.reshape(B * seq_len, hidden_dim * ACTION_CHUNK_PER_CURVE)

        # 2) 一次性前向
        out = self.forward(x)                      # (B*seq_len, total_dim)
        out = out.view(B, seq_len, out_len)       # (B, seq_len, out_len)

        # 3) 拆分出控制点和平移长度 logits
        cd = self.continuous_dim                  # continuous dim
        pt_dim = cd // 2                          # 2 control points per segment

        # P1 and P2
        P1 = out[..., :pt_dim]                    # (B, seq_len, pt_dim)
        P2 = out[..., pt_dim:2*pt_dim]            # (B, seq_len, pt_dim)

        # P0: 第一段全0，后续段 P0 = 上一段 P2
        zeros = torch.zeros_like(P1[:, :1, :])    # (B,1,pt_dim)
        P0_rest = P2[:, :-1, :]                   # (B, seq_len-1, pt_dim)
        P0 = torch.cat([zeros, P0_rest], dim=1)   # (B, seq_len, pt_dim)

        # 4) 长度预测
        length_logits = out[..., cd:]             # (B, seq_len, num_length_classes)
        lengths = torch.argmax(length_logits, dim=-1).unsqueeze(-1).float() + 1.0  # (B, seq_len,1)

        # 5) 合并成一个 Tensor: (B, seq_len, pt_dim*3 + 1)
        combined = torch.cat([P0, P1, P2, lengths], dim=-1)

        return combined
