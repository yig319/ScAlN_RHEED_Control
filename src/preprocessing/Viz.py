import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
import seaborn as sns
from scipy.signal import savgol_filter

def set_labels(ax, xlabel=None, ylabel=None, title=None, xlim=None, ylim=None, yaxis_style='sci', 
            logscale=False, legend=None, ticks_both_sides=True):
    """
    Set labels and other properties of the given axes.

    Args:
        ax: Axes object.
        xlabel (str, optional): X-axis label. Defaults to None.
        ylabel (str, optional): Y-axis label. Defaults to None.
        title (str, optional): Plot title. Defaults to None.
        xlim (tuple, optional): X-axis limits. Defaults to None.
        ylim (tuple, optional): Y-axis limits. Defaults to None.
        yaxis_style (str, optional): Y-axis style. Defaults to 'sci'.
        logscale (bool, optional): Use log scale on the y-axis. Defaults to False.
        legend (list, optional): Legend labels. Defaults to None.
        ticks_both_sides (bool, optional): Display ticks on both sides of the axes. Defaults to True.

    Returns:
        None
    """
    if type(xlabel) != type(None): ax.set_xlabel(xlabel)
    if type(ylabel) != type(None): ax.set_ylabel(ylabel)
    if type(title) != type(None): ax.set_title(title)
    if type(xlim) != type(None): ax.set_xlim(xlim)
    if type(ylim) != type(None): ax.set_ylim(ylim)
    if yaxis_style == 'sci':
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useLocale=False)    
    if logscale: ax.set_yscale("log") 
    if legend: ax.legend(legend)
    ax.tick_params(axis="x",direction="in")
    ax.tick_params(axis="y",direction="in")
    if ticks_both_sides:
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')

def trim_axes(axs, N):
    """
    Reduce *axs* to *N* Axes. All further Axes are removed from the figure.
    """
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]

def show_images(images, labels=None, img_per_row=8, img_height=1, show_colorbar=False, 
                clim=3, scale_0_1=False, hist_bins=None, show_axis=False):
    
    '''
    Plots multiple images in grid.
    
    images
    labels: labels for every images;
    img_per_row: number of images to show per row;
    img_height: height of image in axes;
    show_colorbar: show colorbar;
    clim: int or list of int, value of standard deviation of colorbar range;
    scale_0_1: scale image to 0~1;
    hist_bins: number of bins for histogram;
    show_axis: show axis
    '''
    
    assert type(images) == list or type(images) == np.ndarray, "do not use torch.tensor for hist"
    if type(clim) == list:
        assert len(images) == len(clim), "length of clims is not matched with number of images"

    def scale(x):
        if x.min() < 0:
            return (x - x.min()) / (x.max() - x.min())
        else:
            return x/(x.max() - x.min())
    
    h = images[0].shape[1] // images[0].shape[0]*img_height + 1
    if not labels:
        labels = range(len(images))
        
    n = 1
    if hist_bins: n +=1
        
    fig, axes = plt.subplots(n*len(images)//img_per_row+1*int(len(images)%img_per_row>0), img_per_row, 
                             figsize=(16, n*h*len(images)//img_per_row+1))
    trim_axes(axes, len(images))

    for i, img in enumerate(images):
        
#         if torch.is_tensor(x_tensor):
#             if img.requires_grad: img = img.detach()
#             img = img.numpy()
            
        if scale_0_1: img = scale(img)
        
        if len(images) <= img_per_row and not hist_bins:
            index = i%img_per_row
        else:
            index = (i//img_per_row)*n, i%img_per_row

        axes[index].title.set_text(labels[i])
        im = axes[index].imshow(img)
        if show_colorbar:
            m, s = np.mean(img), np.std(img) 
            if type(clim) == list:
                im.set_clim(m-clim[i]*s, m+clim[i]*s) 
            else:
                im.set_clim(m-clim*s, m+clim*s) 

            fig.colorbar(im, ax=axes[index])
            
        if show_axis:
            axes[index].tick_params(axis="x",direction="in", top=True)
            axes[index].tick_params(axis="y",direction="in", right=True)
        else:
            axes[index].axis('off')

        if hist_bins:
            index_hist = (i//img_per_row)*n+1, i%img_per_row
            h = axes[index_hist].hist(img.flatten(), bins=hist_bins)
        
    plt.show()
    

# from m3_learning.viz by Joshua Agar
from matplotlib import patheffects

def number_to_letters(num):
    letters = ''
    while num >= 0:
        num, remainder = divmod(num, 26)
        letters = chr(97 + remainder) + letters
        num -= 1  # decrease num by 1 because we have processed the current digit
    return letters

def layout_fig(graph, mod=None, figsize=None, layout='compressed', **kwargs):
    """Utility function that helps lay out many figures

    Args:
        graph (int): number of graphs
        mod (int, optional): value that assists in determining the number of rows and columns. Defaults to None.

    Returns:
        tuple: figure and axis
    """

    # sets the kwarg values
    for key, value in kwargs.items():
        exec(f'{key} = value')

    # Sets the layout of graphs in matplotlib in a pretty way based on the number of plots

    if mod is None:
        # Select the number of columns to have in the graph
        if graph < 3:
            mod = 2
        elif graph < 5:
            mod = 3
        elif graph < 10:
            mod = 4
        elif graph < 17:
            mod = 5
        elif graph < 26:
            mod = 6
        elif graph < 37:
            mod = 7

    if figsize is None:
        figsize = (3 * mod, 3 * (graph // mod + (graph % mod > 0)))

    # builds the figure based on the number of graphs and a selected number of columns
    fig, axes = plt.subplots(
        graph // mod + (graph % mod > 0),
        mod,
        figsize=figsize, layout=layout
    )

    # deletes extra unneeded axes
    axes = axes.reshape(-1)
    for i in range(axes.shape[0]):
        if i + 1 > graph:
            fig.delaxes(axes[i])

    return fig, axes[:graph]


def imagemap(ax, data, colorbars=True, clim=None, divider_=True, cbar_number_format="%.1e", **kwargs):
    """pretty way to plot image maps with standard formats

    Args:
        ax (ax): axes to write to
        data (array): data to write
        colorbars (bool, optional): selects if you want to show a colorbar. Defaults to True.
        clim (array, optional): manually sets the range of the colorbars. Defaults to None.
    """

    if data.ndim == 1:
        data = data.reshape(
            np.sqrt(data.shape[0]).astype(
                int), np.sqrt(data.shape[0]).astype(int)
        )

    cmap = plt.get_cmap("viridis")

    if clim is None:
        im = ax.imshow(data, cmap=cmap)
    else:
        im = ax.imshow(data, clim=clim, cmap=cmap)

    ax.set_yticklabels("")
    ax.set_xticklabels("")
    ax.set_yticks([])
    ax.set_xticks([])

    if colorbars:
        if divider_:
            # adds the colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="10%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax, format=cbar_number_format)
        else:
            cb = plt.colorbar(im, fraction=0.046, pad=0.04)
            cb.ax.tick_params(labelsize=6, width=0.05)




def labelfigs(axes, number=None, style="wb",
              loc="tl", string_add="", size=8,
              text_pos="center", inset_fraction=(0.15, 0.15), **kwargs):

    # initializes an empty string
    text = ""

    # Sets up various color options
    formatting_key = {
        "wb": dict(color="w", linewidth=.75),
        "b": dict(color="k", linewidth=0),
        "w": dict(color="w", linewidth=0),
    }

    # Stores the selected option
    formatting = formatting_key[style]

    xlim = axes.get_xlim()
    ylim = axes.get_ylim()

    x_inset = (xlim[1] - xlim[0]) * inset_fraction[1]
    y_inset = (ylim[1] - ylim[0]) * inset_fraction[0]

    if loc == 'tl':
        x, y = xlim[0] + x_inset, ylim[1] - y_inset
    elif loc == 'tr':
        x, y = xlim[1] - x_inset, ylim[1] - y_inset
    elif loc == 'bl':
        x, y = xlim[0] + x_inset, ylim[0] + y_inset
    elif loc == 'br':
        x, y = xlim[1] - x_inset, ylim[0] + y_inset
    elif loc == 'ct':
        x, y = (xlim[0] + xlim[1]) / 2, ylim[1] - y_inset
    elif loc == 'cb':
        x, y = (xlim[0] + xlim[1]) / 2, ylim[0] + y_inset
    else:
        raise ValueError(
            "Invalid position. Choose from 'tl', 'tr', 'bl', 'br', 'ct', or 'cb'.")

    text += string_add

    if number is not None:
        text += number_to_letters(number)

    text_ = axes.text(x, y, text, va='center', ha='center',
                      path_effects=[patheffects.withStroke(
                      linewidth=formatting["linewidth"], foreground="k")],
                      color=formatting["color"], size=size, **kwargs
                      )

    text_.set_zorder(np.inf)
