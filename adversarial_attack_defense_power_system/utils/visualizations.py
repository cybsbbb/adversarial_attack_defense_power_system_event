import matplotlib.pyplot as plt


def sample_visualization(event, save_path='./tmp.png', title="", issave=False, isshow=False):
    plt.style.context(['ieee', 'no-latex'])
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 10))
    plt.subplots_adjust(wspace=0, hspace=0)
    axes[0].plot(event[:, :, 0])
    axes[1].plot(event[:, :, 1])
    axes[2].plot(event[:, :, 2])
    axes[3].plot(event[:, :, 3])
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.suptitle(title, fontsize=20)
    if issave:
        plt.savefig(save_path)
    if isshow:
        plt.show()
    plt.close()
    return 0
