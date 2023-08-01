import numpy as np
import os
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, 'src')
from utils import ensure_dir

def show_image(image, title=None, save_path=None):
    '''
    Given np.array image, display using matplotlib
    '''
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    plt.imshow(image)
    # Remove tick marks
    plt.tick_params(bottom=False, left=False)

    # Add title
    if title is not None:
        plt.title(title)

    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

def make_grid(flattened, items_per_row):
    '''
    Given a 1D list of items and how many elements per row, return a 2D list. Last row padded with None
    Helper to be called before show_image_rows()

    Arg(s):
        flattened : list[any]
            1D list of anything
        items_per_row : int
            number of elements per row

    Returns:
        list[list[any]] : 2D list of elements in a grid
    '''
    length = len(flattened)
    grid = []
    for i in range(0, length, items_per_row):
        if i + items_per_row <= length:
            grid.append(flattened[i: i + items_per_row])
        else:
            padded_row = flattened[i:]
            if type(padded_row) != list:
                padded_row = list(padded_row)
            while len(padded_row) < items_per_row:
                padded_row.append(None)
            grid.append(padded_row)
    return grid


def show_image_rows(images,
                    image_titles=None,
                    image_borders=None,
                    image_size=(2.5, 2.5),
                    row_labels=None,
                    figure_title=None,
                    font_size=12,
                    subplot_padding=None,
                    save_path=None,
                    show=True):
    """
    Display rows of images

    Arg(s):
        images : list[list[np.array]]
            2D array of images to display
            images can be in format of C x H x W or H x W x C
        image_titles : list[list[str]] or None
            2D array of image labels, must be same shape as images
        image_borders : list[list[str]], str, or None
            color of borders for each image
        image_size : (float, float)
            width, height of each image
        row_labels : list[str]
            list of labels for each row, must be same length as len(images)
        figure_title : str
            title for overall figure
        font_size : int
            font size
        subplot_padding : float, (float, float) or None
            padding around each subplot
            if tuple, (hpad, wpad)
        save_path : str
            path to save figure to
    """

    n_rows, n_cols = len(images), len(images[0])
    # Shape sanity checks
    if image_titles is not None:
        assert len(image_titles) == n_rows
        assert len(image_titles[0]) == n_cols
    if row_labels is not None:
        assert len(row_labels) == n_rows

    # Assign border colors
    if image_borders is not None:
        # If
        if type(image_borders) == str:
            borders_row = [image_borders for i in range(n_cols)]
            image_borders = [borders_row for i in range(n_rows)]

        # Sanity check shapes
        assert len(image_borders) == n_rows
        assert len(image_borders[0]) == n_cols


    fig, axs = plt.subplots(n_rows, n_cols, figsize=(image_size[0] * n_cols, image_size[1] * n_rows))

    for row in range(n_rows):
        for col in range(n_cols):
            # Obtain correct axis
            if n_rows == 1 and n_cols == 1:
                ax = axs
            elif n_rows == 1:
                ax = axs[col]
            elif n_cols == 1:
                ax = axs[row]
            else:
                ax = axs[row, col]

            # Display the image
            image = images[row][col]
            # For padding
            if image is not None:
                # Matplotlib expects RGB channel to be in the rck
                if image.shape[0] == 3:
                    image = np.transpose(image, (1, 2, 0))

                ax.imshow(image)

                # Display row text if first image in row
                if row_labels is not None and col == 0:
                    ax.set_ylabel(row_labels[row], fontsize=font_size)
                # Display image title
                if image_titles is not None:
                    ax.set_title(image_titles[row][col], fontsize=font_size)

            # Change border color
            if image_borders is not None:
                plt.setp(ax.spines.values(), color=image_borders[row][col], linewidth=2.0)
            else:
                for loc in ['top', 'bottom', 'right', 'left']:
                    ax.spines[loc].set_visible(False)
                # pass
            # Remove tick marks
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])


    # Set figure title
    if figure_title is not None:
        fig.suptitle(figure_title, fontsize=font_size)

    # Pad if number is provided
    if subplot_padding is not None:
        if type(subplot_padding) == tuple:
            plt.tight_layout(h_pad=subplot_padding[0], w_pad=subplot_padding[1])
        else:
            plt.tight_layout(pad=subplot_padding)
    # Save if path is provided
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    if show:
        plt.show()

    return fig, axs

def bar_graph(data,
              fig=None,
              ax=None,
              display_values=False,
              errors=None,
              labels=None,
              groups=None,
              title=None,
              xlabel=None,
              xlabel_rotation=0,
              ylabel=None,
              ylim=None,
              fig_size=None,
              save_path=None,
              show=True):
    '''
    Given data, make a bar graph

    Arg(s):
        data : N x C np.array
            N : number of bar groups (that would display on a legend)
            C : number of bar classes
        fig : plt.figure
            Optional figure to pass in
        ax : plt axis
            Optional axis to pass in
        display_values : bool
            Boolean to display values of each bar or not
        errors : N x C np.array of errors for each bar
            N : number of bar groups
            C : number of bar classes
        labels : list[str]
            C length list of labels for each bar along x-axis
        groups : list[str]
            N list of group names in legend
        title : str
            title for bar graph
        xlabel : str
            label for x-axis
        ylabel : str
            label for y-axis
        xlabel_rotation : int
            how much to rotate x labels by if they overlap
        fig_size : (float, float)
            (width, height) of figure size
        save_path : str
            if not None, the path to save bar graph to
    '''
    if fig is None and ax is None:
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    else:
        assert fig is not None and ax is not None, "fig and ax must both or neither be None"


    if type(data) == list and type(data[0]) == list:
        data = np.array(data)
    elif type(data) == list:
        data = np.stack(data, axis=0)
    assert len(data.shape) == 2, "Expected 2D data, received {}D data.".format(len(data.shape))
    n_groups, n_classes = data.shape
    # If no errors passed,
    if errors is None:
        errors = np.zeros_like(data)
    # Parameters for bar graphs
    x_pos = np.arange(n_classes)
    width = 0.8 / n_groups
    if labels is None:
        labels = ["" for i in range(n_classes)]
    if groups is None:
        groups = ["" for i in range(n_groups)]

    mid_idx = n_groups // 2
    # Edge case of 1 group
    if n_groups == 1:
        ax.bar(x_pos,
            data[0],
            yerr=errors[0],
            alpha=0.75,
            edgecolor='black',
            capsize=10,
            label=groups[0],
            width=width)
    elif n_groups % 2 == 0: # Even number of groups
        for group_idx, (group_data, group_errors) in enumerate(zip(data, errors)):
            if group_idx < mid_idx:
                ax.bar(x_pos - width * ((mid_idx - group_idx) * 2 - 1) / 2,
                       group_data,
                       yerr=group_errors,
                       alpha=0.75,
                       edgecolor='black',
                       capsize=10,
                       label=groups[group_idx],
                       width=width)
            else:
                ax.bar(x_pos + width * ((group_idx - mid_idx) * 2 + 1) / 2,
                       group_data,
                       yerr=group_errors,
                       alpha=0.75,
                       edgecolor='black',
                       capsize=10,
                       label=groups[group_idx],
                       width=width)

    else:  # Odd number of groups
        for group_idx, (group_data, group_errors) in enumerate(zip(data, errors)):
            if group_idx < mid_idx:
                ax.bar(x_pos - 1 / 2 + width * group_idx,
                    group_data,
                    yerr=group_errors,
                    alpha=0.75,
                    edgecolor='black',
                    capsize=10,
                    label=groups[group_idx],
                    width=width)
            elif group_idx == mid_idx:
                ax.bar(x_pos - width / 2,
                    group_data,
                    yerr=group_errors,
                    alpha=0.75,
                    edgecolor='black',
                    capsize=10,
                    label=groups[group_idx],
                    width=width)
            else:
                ax.bar(x_pos - width / 2 + (group_idx - mid_idx) * width,
                    group_data,
                    yerr=group_errors,
                    alpha=0.75,
                    ecolor='black',
                    capsize=10,
                    label=groups[group_idx],
                    width=width)

    # Set prettiness
    ax.set_xticks(x_pos, labels)
    plt.setp(ax.get_xticklabels(), rotation=xlabel_rotation)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if groups is not None:
        ax.legend()

    # Set ylimits
    if ylim is not None and len(ylim) == 2:
        ax.set_ylim(ylim)

    # Display values above each bar
    if display_values:
        for rect in ax.patches:
            y = rect.get_height()
            x = rect.get_x() + rect.get_width() / 2

            if type(y) == float or isinstance(y, np.floating):
                value = '{:.2f}'.format(y)
            else:
                value = str(y)
            ax.annotate(
                value,
                xy=(x, y),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=12
            )
    if fig_size is not None:
        fig.set_figheight(fig_size[1])
        fig.set_figwidth(fig_size[0])
    plt.tight_layout()

    # If save_path is not None, save graph
    if save_path is not None:
        if not os.path.isdir(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path)

    # Show figure
    if show:
        plt.show()
    # plt.close()
    return fig, ax

def histogram(data,
              multi_method='side',
              weights=None,
              n_bins=10,
              labels=None,
              data_range=None,
              alpha=1.0,
              colors=None,
              title=None,
              xlabel=None,
              ylabel=None,
              xlim=None,
              ylim=None,
              marker=None,
              fig_size=None,
              save_path=None,
              show=True):
    '''
    Plot histogram of data provided

    Arg(s):
        data : np.array or sequence of np.array
            Data for histogram
        multi_method : str
            'side' or 'overlap'
        weights : np.array or sequence of np.array
            Weights for each data point if not None
        n_bins : int
            number of bins for histogram
        labels : list[str]
            label for each type of histogram (should be same number of sequences as data)
        data_range : (float, float)
            upper and lower range of bins (default is max and min)
        fig_size : (float, float) or None
            (width, height) of figure size or None
    '''

    assert multi_method in ['side', 'overlap'], "Unrecognized multi_method: {}".format(multi_method)

    if type(data) == np.ndarray and len(data.shape) == 2:
        data = data.tolist()
    n_data = len(data)

    if labels is None:
        labels = [None for i in range(n_data)]
    if colors is None:
        colors = [None for i in range(n_data)]

    if type(data) == np.ndarray and len(data.shape) == 1:
        if labels[0] is None:
                hist_return = plt.hist(data,
                    weights=weights,
                    bins=n_bins,
                    range=data_range,
                    # color=colors[0],
                    edgecolor='black',
                    alpha=alpha)
        else:
            hist_return = plt.hist(data,
                    weights=weights,
                    bins=n_bins,
                    label=labels[0],
                    range=data_range,
                    # color=colors[0],
                    edgecolor='black',
                    alpha=alpha)
    else:
        # Overlapping histograms
        if multi_method == 'overlap':
            hist_return = []
            for cur_idx, cur_data in enumerate(data):
                hist_return.append(plt.hist(cur_data,
                     bins=n_bins,
                     weights=weights[cur_idx],
                     label=labels[cur_idx],
                     range=data_range,
                     color=colors[cur_idx],
                     edgecolor='black',
                    alpha=alpha))
        # Side by side histogram
        else:
            hist_return = plt.hist(data,
                 bins=n_bins,
                 weights=weights,
                 label=labels,
                 range=data_range,
                 color=None,
                 edgecolor='black',
                 alpha=alpha)

    # Marker is a vertical line marking original
    if marker is not None:
        plt.axvline(x=marker, color='r')

    # Make legend
    if labels is not None:
        plt.legend()
    # Set title and axes labels
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if fig_size is not None:
        plt.figure(figsize=fig_size)
    if save_path is not None:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.clf()

    return hist_return # (bins, bin_values, _)

def plot(xs,
         ys,
         fig=None,
         ax=None,
         labels=None,
         alpha=1.0,
         marker_size=5,
         colors=None,
         point_annotations=None,
         title=None,
         xlabel=None,
         ylabel=None,
         xlimits=None,
         ylimits=None,
         scatter=True,
         line=True,
         highlight=None,
         highlight_label=None,
         fig_size=None,
         save_path=None,
         show=False):
    '''
    Arg(s):
        xs : list[list[float]]
            x values
        ys : list[list[float]]
            y values
        ax : plt.subplot axis
            optional axis to plot on
        labels : list[str]
            line labels for the legend
        alpha : int
            transparency of points
        marker_size : int
            size of markers for scatter plot
        colors : list[str]
            color for each list in xs
        point_annotations : list[list[any]]
            optional per point annotations
        title : str
            title of plot
        xlabel : str
            x-axis label
        ylabel : str
            y-axis label
        xlimits : [float, float] or None
            limits for x-axis
        ylimits : [float, float] or None
            limits for y-axis
        scatter : bool or list[bool]
            denoting if should show each data point or not
        line : bool or list[bool]
            denoting if should connect lines or not
        highlight : (list[float], list[float])
            tuple of data point(s) to accentuate
        highlight_label : str or None
            label for the highlighted point or line
        fig_size : (float, float) or None
            (width, height) of figure or None
        save_path : str
            path to save graph to
        show : bool
            whether or not to display graph

    Returns:
        fig, ax
            figure and axes of plot
    '''
    if fig is None and ax is None:
        plt.clf()
        if fig_size is not None:
            fig = plt.figure(figsize=fig_size)
        else:
            fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    else:
        assert fig is not None and ax is not None, "fig and ax must both or neither be None"

    n_lines = len(xs)
    if labels is None:
        labels = [None for i in range(n_lines)]

    assert len(ys) == n_lines, "ys list must be same length as xs. Received {} and {}".format(len(ys), n_lines)
    assert len(labels) == n_lines, "Labels list must be same length as xs. Received {} and {}".format(len(labels), n_lines)

    if colors is not None:
        assert len(colors) == n_lines, "Length of color array must match length of xs. Received {} and {}".format(len_colors, n_lines)

    # if alphas is not None:
    #     assert type(alphas) == float or len(alphas) == n_lines

    # Determine plot types
    if type(scatter) == bool:
        scatter = [scatter for i in range(n_lines)]
    else:
        len(scatter) == n_lines, "scatter list must be same length as xs. Received {} and {}".format(len(scatter), n_lines)
    if type(line) == bool:
        line = [line for i in range(n_lines)]
    else:
        len(line) == n_lines, "line list must be same length as xs. Received {} and {}".format(len(line), n_lines)


    # Plot lines
    for idx in range(n_lines):
        x = xs[idx]
        y = ys[idx]
        label = labels[idx]

        if point_annotations is not None:
            point_annotation = point_annotations[idx]
        else:
            point_annotation = None
        format_str = 'o'

        if scatter[idx] and line[idx]:
            format_str = '-o'
        elif not scatter[idx] and line[idx]:
            format_str = '-'

        # Add color
        if colors is not None:
            format_str += colors[idx]

        if label is not None:
            ax.plot(x, y,
            format_str,
            alpha=alpha,
            markersize=marker_size,
            zorder=1,
            label=label)
        else:
            ax.plot(x, y,
            format_str,
            alpha=alpha,
            markersize=marker_size,
            zorder=1)

        # Annotate points
        if point_annotation is not None:
            for pt_idx, annotation in enumerate(point_annotation):
                ax.annotate(annotation, (x[pt_idx], y[pt_idx]))

    # Highlight certain point or line
    if highlight is not None:
        highlight_x, highlight_y = highlight
        zorder = 3
        # Is a point
        if len(highlight_x) == 1:
            format_str = 'ys'
            if highlight_label is not None:
                ax.plot(
                    highlight_x,
                    highlight_y,
                    format_str,
                    markersize=marker_size,
                    zorder=zorder,
                    label=highlight_label)
            else:
                ax.plot(
                    highlight_x,
                    highlight_y,
                    format_str,
                    markersize=marker_size,
                    zorder=zorder)
        else:  # is a line
            format_str = 'r--'
            if highlight_label is not None:
                ax.plot(
                    highlight_x,
                    highlight_y,
                    format_str,
                    zorder=zorder,
                    label=highlight_label)
            else:
                ax.plot(
                    highlight_x,
                    highlight_y,
                    format_str,
                    zorder=zorder)

    # Add limits to axes
    if xlimits is not None:
        ax.set_xlim(xlimits)
    if ylimits is not None:
        ax.set_ylim(ylimits)

    # Set title and labels
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if labels[0] is not None:
        ax.legend()


    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    if show:
        plt.show()

    return fig, ax


def boxplot(data=None,
            labels=None,
            xlabel=None,
            xlabel_rotation=0,
            ylabel=None,
            title=None,
            highlight=None,
            highlight_label=None,
            save_path=None,
            show=True):
    '''
    Create boxplot for each element in data

    Arg(s):
        data : list[list[float]]
            x values
        labels : list[str]
            line labels for the legend
        xlabel : str
            x-axis label
        xlabel_rotation : int
            how much to rotate x labels by if they overlap
        ylabel : str
            y-axis label
        title : str
            title of plot
        highlight : float
            horizontal line value
        save_path : str
            path to save graph to
        show : bool
            whether or not to display graph

    '''

    plt.close('all')
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # Boxplot
    ax.boxplot(
        x=data,
        labels=labels)

    # Add highlight
    if highlight is not None:
        # ax = _plot_highlight(
        #     ax=ax,
        #     highlight=highlight,
        #     highlight_label=highlight_label)
        ax.axhline(
            y=highlight,
            xmin=0,
            xmax=1)
    # Set title and labels
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # Set xlabel rotation
    plt.setp(ax.get_xticklabels(), rotation=xlabel_rotation)

    # Display legend
    ax.legend()

    if save_path is not None:
        plt.savefig(save_path)

    if show:
        plt.show()

    return fig, ax


def _plot_highlight(ax,
                    highlight,
                    highlight_label=None,
                    marker_size=10):
    # if highlight is not None:
    highlight_x, highlight_y = highlight
    zorder = 3
    # Is a point
    if len(highlight_x) == 1:
        format_str = 'ys'
        if highlight_label is not None:
            ax.plot(
                highlight_x,
                highlight_y,
                format_str,
                markersize=marker_size,
                zorder=zorder,
                label=highlight_label)
        else:
            ax.plot(
                highlight_x,
                highlight_y,
                format_str,
                markersize=marker_size,
                zorder=zorder)
    else:  # is a line
        format_str = 'r--'
        if highlight_label is not None:
            ax.plot(
                highlight_x,
                highlight_y,
                format_str,
                zorder=zorder,
                label=highlight_label)
        else:
            ax.plot(
                highlight_x,
                highlight_y,
                format_str,
                zorder=zorder)
    return ax

def pie_chart(sizes,
              labels,
              fig=None,
              ax=None,
              pct_distance=0.75,
              label_distance=1.1,
              relative=True,
              colors=None,
              save_path=None,
              show=True):
    if fig is None or ax is None:
        plt.clf()
        fig, ax = plt.subplots()
    if relative:
        autopct = '%1.2f%%'

    # Check lengths of data passed in
    assert len(sizes) == len(labels), "Received {} length array for sizes and {} length array for labels".format(
        len(sizes), len(labels))
    if colors is not None:
        assert len(sizes) == len(colors), "Received invalid length array for colors ({}). Expected {}.".format(
            len(colors), len(sizes))
    # Plot pie chart
    ax.pie(
        sizes,
        labels=labels,
        pctdistance=pct_distance,
        labeldistance=label_distance,
        autopct=autopct,
        colors=colors)

    if save_path is not None:
        plt.savefig(save_path)

    if show:
        plt.show()

    return fig, ax