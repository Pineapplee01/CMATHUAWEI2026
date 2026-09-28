"""观测条件交叉注意力补全层（推导2-7、2-8）。

query可逐层更新，memory固定为保留观测；padding阻断不可读key，
两次残差和LayerNorm分别对应注意力及前馈更新。
"""
from torch import nn



class CompletionLayer(nn.Module):
    def __init__(self, hidden, heads, dropout):
        super().__init__()
        self.cross = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.ff = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(2 * hidden, hidden))

    def forward(self, query, memory, padding):
        message, _ = self.cross(query, memory, memory, key_padding_mask=padding,
                                need_weights=False)
        query = self.norm1(query + message)
        return self.norm2(query + self.ff(query))

