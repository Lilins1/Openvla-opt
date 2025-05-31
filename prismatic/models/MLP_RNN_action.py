import torch
import torch.nn as nn
from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX

class MLP_RNN_ActionHead(nn.Module):
    def __init__(self, 
                 input_dim=4096, 
                 mlp_hidden_size=4096,
                 rnn_hidden_size=1024,
                 action_dim=7,
                 num_layers=2,
                 rnn_type='rnn'):
        """
        混合模型参数:
        input_size: 输入特征维度 (4096)
        mlp_hidden_size: MLP输出维度 (4096)
        rnn_hidden_size: RNN隐藏层大小 (1024)
        output_size: 输出类别数 (7)
        num_layers: RNN层数 2
        rnn_type: 'rnn'/'lstm'/'gru'
        总内存 = 183,776,519 x 4 bytes = 735.11 MB
        """

        super(MLP_RNN_ActionHead, self).__init__()
        
        # 前端MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * action_dim, 768 * action_dim),  # 第一层降维
            nn.ReLU(),
            nn.Dropout(0.2),              # 防止过拟合
            nn.Linear(768 * action_dim, mlp_hidden_size),  # 输出mlp_hidden_size维特征
            nn.ReLU()
        )

        
        
        # 后端RNN
        self.rnn_type = rnn_type.lower()
        self.rnn_hidden_size = rnn_hidden_size
        self.num_layers = num_layers
        
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=mlp_hidden_size,
                hidden_size=rnn_hidden_size,
                num_layers=num_layers,
                batch_first=True
            )
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=mlp_hidden_size,
                hidden_size=rnn_hidden_size,
                num_layers=num_layers,
                batch_first=True
            )
        else:  # 默认基础RNN
            self.rnn = nn.RNN(
                input_size=mlp_hidden_size,
                hidden_size=rnn_hidden_size,
                num_layers=num_layers,
                batch_first=True,
                nonlinearity='relu'
            )
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(rnn_hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, action_dim)
        )
    
    def forward(self, x, prev_state=None):
        """
        前向传播
        x: 输入序列 (batch_size, seq_len, input_size)
        或单步输入 (batch_size, input_size)
        
        prev_state: 前一个状态 (RNN) 或 (h, c) (LSTM)
        """
        # 这里我们用 self.rnn.weight_hh_l0 参数的 dtype
        target_dtype = self.rnn.weight_hh_l0.dtype
        x = x.to(dtype=target_dtype)


        # 保存原始维度
        batch_size = x.size(0)
        is_sequence = len(x.shape) == 3
        
        # 单步处理时添加序列维度
        if not is_sequence:
            x = x.unsqueeze(1)  # (batch, 1, input_size)
        
        # 应用MLP到每个时间步
        # 重塑: (batch, seq, input) -> (batch*seq, input)
        original_shape = x.shape
        x = x.reshape(-1, original_shape[-1])
        x = self.mlp(x)
        # 恢复序列维度: (batch, seq, mlp_hidden_size)
        x = x.view(original_shape[0], original_shape[1], -1)
        
        # 初始化 hidden state 时也用同样的 dtype
        if prev_state is None:
            if self.rnn_type == 'lstm':
                h0 = torch.zeros(self.num_layers, batch_size, self.rnn_hidden_size,
                                dtype=target_dtype, device=x.device)
                c0 = torch.zeros_like(h0)
                state = (h0, c0)
            else:
                state = torch.zeros(self.num_layers, batch_size, self.rnn_hidden_size,
                                    dtype=target_dtype, device=x.device)
        else:
            # 如果是从外面传进来的 state，确保它也转到同 dtype
            if isinstance(prev_state, tuple):
                state = (prev_state[0].to(dtype=target_dtype),
                        prev_state[1].to(dtype=target_dtype))
            else:
                state = prev_state.to(dtype=target_dtype)

        
        # RNN处理
        if self.rnn_type == 'lstm':
            out, (hn, cn) = self.rnn(x, state)
            new_state = (hn, cn)
        else:
            out, new_state = self.rnn(x, state)
        
        # 取最后一个时间步的输出
        if is_sequence:
            out = out[:, -1, :]  # (batch, hidden_size)
        else:
            out = out.squeeze(1)  # (batch, hidden_size)
        
        # 最终分类层
        out = self.fc(out)
        
        # 返回输出和新状态
        return out, new_state
    
    def predict_action(self, actions_hidden_states, prev_state=None):
        """
        逐步生成多组动作。

        Args:
            x: Tensor，单步输入形状 (batch, input_dim)
               或多步输入形状 (batch, seq_len, input_dim)
            prev_state: 上一时刻 RNN/LSTM 隐状态，默认为 None（自动初始化）

        Returns:
            actions: Tensor，形状 (batch, seq_len, action_dim) 或 (batch, action_dim)
            state: 最终的 RNN/LSTM 隐状态
        """
        # 1) 先提取 batch_size 并 reshape 回 (batch, chunk_len, hidden_dim)
        batch_size = actions_hidden_states.size(0)
        x = actions_hidden_states.view(batch_size, NUM_ACTIONS_CHUNK, -1)
        is_sequence = (x.ndim == 3)
        # 如果是单步，先统一为 (batch, 1, input_dim)
        if not is_sequence:
            x = x.unsqueeze(1)

        batch_size, seq_len, _ = x.shape
        state = None
        outputs = []

        for t in range(seq_len):
            xt = x[:, t, :]               # (batch, input_dim)
            out_t, prev_state = self.forward(xt, prev_state)
            # out_t: (batch, action_dim)
            outputs.append(out_t)
            if state == None:
                state = prev_state

        # 拼成 (batch, seq_len, action_dim)
        actions = torch.stack(outputs, dim=1)

        # 如果原本是单步输入，就 squeeze 掉时间维度
        if not is_sequence:
            actions = actions.squeeze(1)  # -> (batch, action_dim)

        return actions, state
    
    def init_hidden(self, batch_size, device='cpu'):
        """初始化隐藏状态"""
        if self.rnn_type == 'lstm':
            return (
                torch.zeros(self.num_layers, batch_size, self.rnn_hidden_size).to(device),
                torch.zeros(self.num_layers, batch_size, self.rnn_hidden_size).to(device)
            )
        else:
            return torch.zeros(self.num_layers, batch_size, self.rnn_hidden_size).to(device)