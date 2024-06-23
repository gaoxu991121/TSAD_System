from matplotlib import pyplot as plt
import matplotlib
from mpl_toolkits import axisartist
# matplotlib.use('Agg')

def colorTransform(r,g,b,a:float = 1.0):
    return (r/255,g/255,b/255,a)


def plotAllResult(x_axis,y_axises,title = "",labels = [],save_path = "",threshold = None,segments = [],isScore=True,f1_pair = []):
    plt.rcParams.update({'font.size': 14})  # 设置全局字体大小

    fig = plt.figure(dpi=300, figsize=(10, 5))
    for index in range(len(y_axises)):
        y_axis = y_axises[index]
        ax = axisartist.Subplot(fig, len(y_axises),1,index+1)
        # 将绘图区对象添加到画布中
        fig.add_axes(ax)

        if isScore:
            ax.plot(x_axis, y_axis, color=colorTransform(70, 130, 180), linewidth=0.8)
            isScore = False

            if threshold != None:
                ax.axhline(y=threshold, color=(0, 0, 0.9, 0.75), linestyle='--', linewidth=3)  # 使用红色虚线

        else:
            ax.plot(x_axis, y_axis, color=colorTransform(57, 136, 145), linewidth=0.8)
            for pair in f1_pair:
                ax.axvline(x=pair[0], color=colorTransform(131,136,186), linestyle='-.', linewidth=2)  # 使用红色虚线

        # if len(labels) > 0:
        #     ax.plot(x_axis, labels, color='red', label='label', linestyle='--',linewidth=2)

        if len(segments) > 0:
            for item in segments:
                ax.fill_betweenx([-1, 1.1], item[0], item[1], color=colorTransform(255, 228, 181, 0.8))
                # 添加垂直线
                # plt.axvline(x=item[0], ymin=-1, ymax=1, color='red', linestyle='--', linewidth=2)
                # plt.axvline(x=item[1], ymin=-1, ymax=1, color='red', linestyle='--', linewidth=2)



        # 设置x轴和y轴可见性
        ax.axis['top'].set_visible(False)
        ax.axis['right'].set_visible(False)
        ax.axis['bottom'].set_visible(True)
        ax.axis['left'].set_visible(True)

        ax.axis['left'].set_axisline_style("->", size=1.5)
        ax.axis['bottom'].set_axisline_style("->", size=1.5)

        ax.axis[:].line.set_linewidth(2)

        # 设置只显示x轴和y轴

        ax.xaxis.set_ticks_position('bottom')
        # 设置x轴和y轴的范围
        axis_xlim = list(range(1, len(x_axis) + 1))
        axis_xlim_length = len(axis_xlim)
        # ax.set_xlim(0, axis_xlim_length)

        # 设置x轴和y轴的刻度位置和标签
        ax.set_xticks([])
        ax.set_xticklabels([])

        y_ticklabels = ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']

        if isScore:
            ax.yaxis.set_ticks_position('left')
            ax.set_yticks([0, 1])
            ax.set_ylim(0, 1)
            ax.set_yticklabels(["0", "1"])
        else:
            ax.set_yticks([0, 1])
            ax.set_ylim(0, 1)
            ax.set_yticklabels(["0", "1"])
    # 添加图例和标题
    # plt.legend(loc='upper right')
    # plt.title(title)
    plt.tight_layout()  # 调整子图布局
    plt.subplots_adjust(left=0.15, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.3)
    if save_path != "":
        plt.savefig(save_path)

    # 显示图表
    # plt.show()

