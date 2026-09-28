"""共用绘图样式；各题图表生成逻辑在各题目录。"""

def pyplot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.sans-serif': ['DejaVu Sans', 'Arial', 'SimHei'],
                         'axes.unicode_minus': False, 'figure.dpi': 150})
    return plt
