# 问题一分析报告（改良统一时间网格）

> 质量规则见 [三种模态无效片段识别方法](三种模态无效片段识别方法.md)；播放轴见 [统一时间戳—+标准化](统一时间戳—+标准化)。

## 1. 时间网格与 support

片段播放时长 $D$，共用等分窗

$$
B_k=\Bigl[\frac{kD}{L},\frac{(k+1)D}{L}\Bigr),\quad L=50,\ k=0,\ldots,49.
$$

音、视频共用 $\{B_k\}$。模态实际支持上界 $T^{(m)}\le D$：

$$
S_k^{(m)}=\mathbf1\bigl(B_k\cap[0,T^{(m)})\neq\emptyset\bigr).
$$

$S_k^{(m)}=0$ 时：$M_k^{(m)}=0$，特征置零；不拉伸、不填末帧。音视源帧先重采样到 `common_hop_sec=0.04`，再聚到 $\{B_k\}$。文本为 WordPiece 词元轴（`max_length=50` 截断/填充），与 $B_k$ **不同步**。

## 2. 质量掩码

$$
M_j^{(T)}=a_j,\qquad
M_k^{(A)}=S_k^{(A)}\,\mathbf1\{N_k^{A}\ge K_{\min}\}\,\mathbf1\{\mathrm{RMS}_k\ge\tau_A\},
$$

$$
M_k^{(V)}=S_k^{(V)}\,\mathbf1\{n_k>0\}\,\mathbf1\{\rho_k^{V}\ge\tau_V\}.
$$

$\tau_A$：批内窗 RMS 的 `tau_a_percentile` 百分位。阈值见 `config_q1.json`。

## 3. 重叠聚合与输出

原始帧支持 $J_j^{(m)}$，权重 $w_{kj}=\operatorname{len}(B_k\cap J_j^{(m)})$：

$$
\bar{\mathbf f}_k=
\frac{\sum_{j:q_j=1}w_{kj}\mathbf f_j}{\sum_{j:q_j=1}w_{kj}}
\quad(D_k>0),\quad
\mathbf y_k=\begin{cases}\bar{\mathbf f}_k,&M_k=1,\\\mathbf0,&M_k=0.\end{cases}
$$

$$
\mathbf Y^{(T)}\in\mathbb R^{N\times50\times768},\ 
\mathbf Y^{(A)}\in\mathbb R^{N\times50\times74},\ 
\mathbf Y^{(V)}\in\mathbb R^{N\times50\times35}.
$$

可选 `--standardize`：文本 / 音 / 视有效位估参→标准化→无效再置零。保留全部样本；无 MFA。
