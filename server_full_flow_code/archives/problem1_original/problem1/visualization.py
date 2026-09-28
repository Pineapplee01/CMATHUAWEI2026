"""问题1的时间覆盖图。"""
from shared.plotting import pyplot



def alignment_plots(sid, intervals, masks, output):
    plt = pyplot(); output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    for m, name in enumerate(('text','audio','vision')):
        for (a,b), valid in zip(intervals, masks[m]):
            if valid: ax.broken_barh([(a,b-a)], (m-.3,.6))
    ax.set(xlabel='Clip time (seconds)', yticks=[0,1,2], yticklabels=['text','audio','vision'],
           title=f'{sid}: temporal coverage')
    fig.savefig(output / f'{sid}.png', dpi=300, bbox_inches='tight'); plt.close(fig)

