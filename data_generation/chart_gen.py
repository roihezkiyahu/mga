import random

import matplotlib.ticker as ticker
# from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from tqdm import tqdm
from data_generation.gen_ax_fig import *
from data_generation.rand_cont import random_operation_n_times
from plot_functions.mga_plt import plot_image_with_boxes
from utils.util_funcs import get_bboxes, safe_literal_eval, annotation_to_labels, linear_regression, is_numeric
import matplotlib.transforms as mtransforms
import matplotlib.pyplot as plt
from itertools import chain
# from scipy import stats
plt.ticklabel_format(style='plain')


def generate_line_chart(x, y, line_color='blue', grid_style="both", x_title=None, y_title=None, graph_title=None,
                        theme="default", line_style='-', marker_style=None, figsize=(6, 4), name=None,
                        rotate=False, show=True, cont=False):
    """
    Generates a line chart using matplotlib, with options for customization.

    Args:
        x (array-like): The x-coordinates of the data points.
        y (array-like): The y-coordinates of the data points.
        line_color (str, optional): The color of the line. Defaults to 'blue'.
        grid_style (str, optional): The style of grid to use. Options are "both", "x", "y", or "none". Defaults to "both".
        x_title (str, optional): The title of the x-axis. Defaults to None.
        y_title (str, optional): The title of the y-axis. Defaults to None.
        graph_title (str, optional): The title of the graph. Defaults to None.
        theme (str, optional): The theme to apply to the chart. Defaults to 'default'.
        line_style (str, optional): The style of the line. Defaults to '-'.
        marker_style (str, optional): The style of the markers. Defaults to None.
        figsize (tuple, optional): The size of the figure. Defaults to (6, 4).
        name (str, optional): The name to save the figure as. Defaults to None.
        rotate (bool, optional): Whether to rotate x-axis labels. Defaults to False.
        show (bool, optional): Whether to show the plot. Defaults to True.
        cont (bool, optional): Whether to generate a continuous line plot. Defaults to False.

    Returns:
        tuple: A tuple containing a dictionary with data information and the name of the saved file.

    """
    set_style(theme)
    fig, ax = plt.subplots(figsize=figsize)
    set_grid(grid_style, ax)
    set_title(ax, x_title, y_title, graph_title)
    if not cont:
        ax.plot(x, y, color=line_color, linestyle=line_style, marker=marker_style)
        ax.set_xticks(x)
        ax.set_xticklabels(x, rotation=0 if not rotate else 45,
                           ha='right' if rotate else 'center')  # Rotate for better visibility if needed
    else:
        if isinstance(x[0], str):
            mix_x, max_x = np.random.randint(5, 50), np.random.randint(52, 150)
        else:
            mix_x, max_x = np.min(x), np.max(x)
        x_cont, y_cont = random_operation_n_times(mix_x, max_x, np.random.random()*1.5+0.5, np.random.randint(1, 5))
        ax.plot(x_cont, y_cont, color=line_color, linestyle=line_style, marker=None)
        if np.random.random() < 0.25:
            tick_positions = np.linspace(np.min(x_cont)*np.random.randint(85, 90) / 100,
                                         np.max(x_cont)*np.random.randint(105, 115) / 100, len(x))
            ax.set_xticks(tick_positions)
            valid_ticks = np.all([tick_positions > np.min(x_cont), tick_positions < np.max(x_cont)], axis=0)
            x_ticks = ax.get_xticks()[valid_ticks]
            ax.set_xticklabels(x, rotation=0 if not rotate else 45, ha='right' if rotate else 'center')
            x = list(np.array(x)[valid_ticks])
            x_cont = np.array(x_cont)
            y = list(np.array(y_cont)[[np.argmin(np.abs(x_-x_cont)) for x_ in x_ticks]])
        else:
            tick_positions = np.linspace(np.quantile(x_cont, np.random.randint(1, 10)/100),
                                         np.quantile(x_cont, np.random.randint(90, 99)/100), len(x))
            ax.set_xticks(tick_positions)
            x_ticks = ax.get_xticks()
            ax.set_xticklabels(x)
            x_cont = np.array(x_cont)
            y = list(np.array(y_cont)[[np.argmin(np.abs(x_-x_cont)) for x_ in x_ticks]])
            ax.set_xticklabels(x, rotation=0 if not rotate else 45, ha='right' if rotate else 'center')  # Rotate for better visibility if needed
    if rotate:
        labels = ax.get_xticklabels()
        offset = mtransforms.ScaledTranslation((np.random.uniform(1, 2)*figsize[0]/ax.figure.get_dpi()), 0, ax.figure.dpi_scale_trans)
        for label in labels:
            label.set_transform(label.get_transform() + offset)


    set_ax_loc_rotate(ax, rotate)
    data_dict = {
        'chart-type': 'line',
        **extract_ax_data(ax, fig, x if not cont else x_ticks, y, name, data_type="visual-elements.lines"),
        'visual-elements.bars': [],
        'visual-elements.boxplots': [],
        'visual-elements.dot points': [],
        'visual-elements.scatter points': []
    }
    final_name = save_file(name, fig, data_dict)
    if show:
        plt.show()
    plt.close(fig)
    return data_dict, final_name


def random_generate_line_chart(x, y, x_title=None, y_title=None, graph_title=None, name=None, show=True, cont=False):
    """
    Generates a line chart with random visual parameters.

    Args:
        x (array-like): Array-like object containing the x-values.
        y (array-like): Array-like object containing the y-values.
        x_title (str, optional): Title for the x-axis. Defaults to None.
        y_title (str, optional): Title for the y-axis. Defaults to None.
        graph_title (str, optional): Title for the graph. Defaults to None.
        name (str, optional): Name for the saved file. Defaults to None.
        show (bool, optional): Whether to display the generated graph. Defaults to True.
        cont (bool, optional): Whether to convert to continus data. Defaults to False.

    Returns:
        tuple: A tuple containing the data dictionary and the final file name.
    """
    color_palette = ['blue', 'red', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'pink', "black"]
    grid_style_choices = ["both", "x", "y", "none"]
    theme_choices = ['dark_background', 'default', 'grayscale', 'dark_gray']
    line_color = random.choice(color_palette)
    grid_style = np.random.choice(grid_style_choices, p=[0.1, 0.1, 0.1, 0.7])
    theme = np.random.choice(theme_choices, p=[0.1, 0.6, 0.2, 0.1])
    if theme == 'dark_background' and line_color == "black":
        theme = 'default'
    line_style = "-"
    marker_style = np.random.choice(['o', 'x', None], p=[0.2, 0.2, 0.6])
    figsize_w = random.randint(4, 13)
    figsize_h = random.randint(max(4, int(figsize_w / 3.7)), min(int(figsize_w * 1.5), 9))
    figsize = (figsize_w, figsize_h)
    data_dict, final_name = generate_line_chart(x, y, line_color=line_color, grid_style=grid_style, x_title=x_title,
                                                y_title=y_title, graph_title=graph_title,
                                                theme=theme, line_style=line_style, marker_style=marker_style,
                                                figsize=figsize, name=name, show=show, cont=cont)
    if not check_text_overlap(data_dict, figsize_h):
        return data_dict, final_name
    os.remove(f"{final_name}.jpg")
    os.remove(f"{final_name}.json")
    if figsize_w < 10:
        figsize_w = random.randint(figsize_w + 2, 13)
        figsize_h = random.randint(max(5, int(figsize_w / 3)), min(int(figsize_w * 1.5), 9))
        figsize = (figsize_w, figsize_h)
    if random.random() < 0.5:
        data_dict, final_name = generate_line_chart(x, y, line_color=line_color, grid_style=grid_style, x_title=x_title,
                                                    y_title=y_title, graph_title=graph_title,
                                                    theme=theme, line_style=line_style, marker_style=marker_style,
                                                    figsize=figsize, name=name, show=show, cont=cont)
        if not check_text_overlap(data_dict, figsize_h):
            return data_dict, final_name
        os.remove(f"{final_name}.jpg")
        os.remove(f"{final_name}.json")
    return generate_line_chart(x, y, line_color=line_color, grid_style=grid_style, x_title=x_title, y_title=y_title,
                               graph_title=graph_title,
                               theme=theme, line_style=line_style, marker_style=marker_style, figsize=figsize,
                               name=name, rotate=45, show=show, cont=cont)


def generate_scatter_chart(x, y, color='blue', grid_style="both", theme="white", show_regression_line=False,
                           x_title=None, y_title=None, graph_title=None, figsize=(6, 4), name=None, show=True,
                           change_background_prob=0.1):
    """
    Generates a scatter chart with given parameters.

    Args:
        x (array-like): Array-like object containing the x-values.
        y (array-like): Array-like object containing the y-values.
        color (str, optional): Color for the scatter points. Defaults to 'blue'.
        grid_style (str, optional): Grid style for the graph. Defaults to 'both'.
        theme (str, optional): Theme for the graph. Defaults to 'white'.
        show_regression_line (bool, optional): Whether to show the regression line. Defaults to False.
        x_title (str, optional): Title for the x-axis. Defaults to None.
        y_title (str, optional): Title for the y-axis. Defaults to None.
        graph_title (str, optional): Title for the graph. Defaults to None.
        figsize (tuple, optional): Figure size for the graph. Defaults to (6, 4).
        name (str, optional): Name for the saved file. Defaults to None.
        show (bool, optional): Whether to display the generated graph. Defaults to True.
        change_background_prob (float, optional): Probability to change the background color. Defaults to 0.1.

    Returns:
        tuple: A tuple containing the data dictionary and the final file name.
    """

    set_style(theme)
    fig, ax = plt.subplots(figsize=figsize)

    # 10% chance to change the background color
    if random.random() < change_background_prob:
        bg_color = random_background_color(exclude=color)
        ax.set_facecolor(bg_color)


    set_grid(grid_style, ax)
    set_title(ax, x_title, y_title, graph_title)
    ax.scatter(x, y, label='Sample Scatter', color=color, marker='o')
    set_ax_loc_rotate(ax, False)

    if show_regression_line:
        # model = LinearRegression().fit(np.array(x).reshape(-1, 1), y)
        # y_pred = model.predict(np.array(x).reshape(-1, 1))
        y_pred, slope, intercept, r_squared = linear_regression(x, y)
        ax.plot(x, y_pred, color='red', linestyle='--', label='Regression Line')

    data_dict = {
        'chart-type': 'scatter',
        **extract_ax_data(ax, fig, x, y, name, 'visual-elements.scatter points'),
        'visual-elements.bars': [],
        'visual-elements.boxplots': [],
        'visual-elements.dot points': [],
        'visual-elements.lines': [],
    }
    final_name = save_file(name, fig, data_dict)
    if show:
        plt.show()
    plt.close(fig)
    return data_dict, final_name


def random_generate_scatter_chart(x, y, x_title=None, y_title=None, graph_title=None, name=None, show=True):
    """
    Generates a scatter chart with random visual parameters.

    Args:
        x (array-like): Array-like object containing the x-values.
        y (array-like): Array-like object containing the y-values.
        x_title (str, optional): Title for the x-axis. Defaults to None.
        y_title (str, optional): Title for the y-axis. Defaults to None.
        graph_title (str, optional): Title for the graph. Defaults to None.
        name (str, optional): Name for the saved file. Defaults to None.
        show (bool, optional): Whether to display the generated graph. Defaults to True.

    Returns:
        tuple: A tuple containing the data dictionary and the final file name.
    """
    color_palette = ['blue', 'red', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'pink', "black"]
    color = random.choice(color_palette)
    grid_style = np.random.choice(["both", "x", "y", "none"], p=[0.1, 0.1, 0.1, 0.7])
    theme = np.random.choice(['dark_background', 'default', 'grayscale', 'dark_gray'], p=[0.1, 0.6, 0.2, 0.1])
    show_regression_line = np.random.choice([True, False], p=[0.1, 0.9])
    if theme == 'dark_background' and color == "black":
        theme = 'default'
    figsize_w = random.randint(4, 13)
    figsize_h = random.randint(max(4, int(figsize_w / 3.7)), min(int(figsize_w * 1.5), 9))
    figsize = (figsize_w, figsize_h)
    # Generate and return the scatter chart
    return generate_scatter_chart(x, y, color=color, grid_style=grid_style, theme=theme,
                                  show_regression_line=show_regression_line,
                                  x_title=x_title, y_title=y_title, graph_title=graph_title, figsize=figsize, name=name,
                                  show=show)


def generate_bar_chart(categories, values, color='blue', grid_style="both", theme="default", orientation="vertical",
                       x_title=None, y_title=None, graph_title=None, figsize=(6, 4), name=None, rotate=False,
                       show=True, long=False):
    """
    Generates a bar chart with given parameters.

    Args:
        categories (array-like): Array-like object containing the categories.
        values (array-like): Array-like object containing the values for each category.
        color (str, optional): Color for the bars. Defaults to 'blue'.
        grid_style (str, optional): Grid style for the graph. Defaults to 'both'.
        theme (str, optional): Theme for the graph. Defaults to 'default'.
        orientation (str, optional): Orientation of the bars ("vertical" or "horizontal"). Defaults to "vertical".
        x_title (str, optional): Title for the x-axis. Defaults to None.
        y_title (str, optional): Title for the y-axis. Defaults to None.
        graph_title (str, optional): Title for the graph. Defaults to None.
        figsize (tuple, optional): Figure size for the graph. Defaults to (6, 4).
        name (str, optional): Name for the saved file. Defaults to None.
        rotate (bool, optional): Whether to rotate the category labels. Defaults to False.
        show (bool, optional): Whether to display the generated graph. Defaults to True.
        long (bool, optional): Whether the chart is long or not. Defaults to False.

    Returns:
        tuple: A tuple containing the data dictionary and the final file name.
    """
    set_style(theme)
    fig, ax = plt.subplots()
    set_grid(grid_style, ax)
    set_title(ax, x_title, y_title, graph_title)
    set_ax_loc_rotate(ax, rotate, long)
    if rotate:
        ax.set_xticklabels(categories if orientation == "vertical" else [int(val) for val in values],
                           rotation=45, ha='right')  # Rotate for better visibility if needed
        labels = ax.get_xticklabels()
        offset = mtransforms.ScaledTranslation((np.random.uniform(1, 2)*figsize[0]/ax.figure.get_dpi()), 0,
                                               ax.figure.dpi_scale_trans)

        for label in labels:
            label.set_transform(label.get_transform() + offset)
    if long:
        ax.tick_params(axis='x', labelsize=random.randint(8, 10))
        ax.tick_params(axis='y', labelsize=random.randint(6, 10))

    if orientation == "vertical":
        ax.bar(categories, values, label='Sample Vertical Bar', color='none' if random.random() < 0.2 else color,
               edgecolor=color)
        chart_type = 'vertical_bar'
    else:
        ax.barh(categories, values, label='Sample Horizontal Bar', color='none' if random.random() < 0.2 else color,
                edgecolor=color)
        chart_type = 'horizontal_bar'

    # Extract properties
    data_dict = {
        'chart-type': chart_type,
        **extract_ax_data(ax, fig, categories, values, name, 'visual-elements.bars', bar_chart_type=chart_type),
        'visual-elements.boxplots': [],
        'visual-elements.dot points': [],
        'visual-elements.lines': [],
        'visual-elements.scatter points': []
    }
    final_name = save_file(name, fig, data_dict)
    if show:
        plt.show()
    plt.close(fig)  # close the plot to avoid displaying
    return data_dict, final_name


def random_generate_bar_chart(categories, values, x_title=None, y_title=None, graph_title=None, name=None, show=True,
                              horizontal_prob=0.25, long=False):
    """
    Generates a bar chart with random visual parameters.

    Args:
        categories (array-like): Array-like object containing the categories.
        values (array-like): Array-like object containing the values for each category.
        x_title (str, optional): Title for the x-axis. Defaults to None.
        y_title (str, optional): Title for the y-axis. Defaults to None.
        graph_title (str, optional): Title for the graph. Defaults to None.
        name (str, optional): Name for the saved file. Defaults to None.
        show (bool, optional): Whether to display the generated graph. Defaults to True.
        horizontal_prob (float, optional): Probability to make the chart horizontal. Defaults to 0.25.
        long (bool, optional): Whether the chart is has long text or not. Defaults to False.

    Returns:
        tuple: A tuple containing the data dictionary and the final file name or an empty dictionary and "img" string.
    """
    color_palette = ['blue', 'red', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'pink', "black"]
    color = random.choice(color_palette)
    grid_style = np.random.choice(["both", "x", "y", "none"], p=[0.1, 0.1, 0.1, 0.7])
    theme = np.random.choice(['dark_background', 'default', 'grayscale', 'dark_gray'], p=[0.2, 0.5, 0.2, 0.1])
    if theme == 'dark_background' and color == "black":
        theme = 'default'
    orientation = "horizontal" if random.random() < horizontal_prob else "vertical"  # 10% chance for horizontal
    if os.sep in name:
        name = os.path.join(os.path.dirname(name), f"{orientation}_{os.path.basename(name)}")
    else:
        name = f"{orientation}_{name}"
    figsize_w = random.randint(4, 13) if not long else random.randint(10, 15)
    figsize_h = random.randint(max(4 if not long else 6, int(figsize_w / 3.7)),
                               min(int(figsize_w * 1.5), 9 if not long else 11))
    figsize = (figsize_w, figsize_h)
    data_dict, final_name = generate_bar_chart(categories, values, color=color, grid_style=grid_style, theme=theme,
                                               orientation=orientation,
                                               x_title=x_title, y_title=y_title, graph_title=graph_title,
                                               figsize=figsize, name=name, show=show, long=long)
    if not check_text_overlap(data_dict, figsize_h):
        return data_dict, final_name
    os.remove(f"{final_name}.jpg")
    os.remove(f"{final_name}.json")
    figsize_w = random.randint(figsize_w + 1, 14 if not long else 16)
    figsize_h = random.randint(max(6, int(figsize_w / 3)), min(int(figsize_w * 1.5), 10))
    figsize = (figsize_w, figsize_h)
    if (random.random() < 0.5 and figsize_w < 10) or long:
        data_dict, final_name = generate_bar_chart(categories, values, color=color, grid_style=grid_style, theme=theme,
                                                   orientation=orientation,
                                                   x_title=x_title, y_title=y_title, graph_title=graph_title,
                                                   figsize=figsize, name=name, show=show, long=long)
        if not check_text_overlap(data_dict, figsize_h):
            return data_dict, final_name
        os.remove(f"{final_name}.jpg")
        os.remove(f"{final_name}.json")
    if not long:
        return generate_bar_chart(categories, values, color=color, grid_style=grid_style, theme=theme,
                                  orientation=orientation,
                                  x_title=x_title, y_title=y_title, graph_title=graph_title, figsize=figsize, name=name,
                                  rotate=True, show=show, long=long)
    else:
        return {}, "img"


def dotplot(xs, ys, ax, fig, ylim, step, show=True, **args):
    """
    Creates dot plots on a given axis.

    Args:
        xs (array-like): The x-values for the dot plot.
        ys (array-like): The y-values for the dot plot.
        ax (matplotlib.axis): The axis on which to plot.
        fig (matplotlib.figure): The figure of the plot.
        ylim (list): The y-axis limits in the format [ymin, ymax].
        step (int): The step size for the y-values.
        show (bool, optional): Whether to display the plot. Defaults to True.
        **args: Additional keyword arguments to be passed to the scatter function.

    Returns:
        list: The y-values of the scatter plot.
    """
    fig_size = fig.get_size_inches()
    scatter_x = []  # x values
    scatter_y = []  # corresponding y values
    for x, y in zip(xs, ys):
        for z in range(ylim[0], y, step):
            scatter_x.append(x)
            scatter_y.append(z + step / 2)
    factor = (ylim[1] - ylim[0]) / 10 / step / (fig_size[1] / 6)
    ax.scatter(scatter_x, scatter_y, s=fig.dpi * 10 / factor ** 2, zorder=3, **args)

    ax.set_xticks(xs)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(step))

    ax.set_xmargin(np.random.uniform(0.04, 0.2))
    ax.set_ylim(ylim)
    if show:
        plt.show()
    return scatter_y


def generate_dot_plot(x, y, color='blue', grid_style="both", theme="white", figsize=(6, 4), ylim=[0, 10], step=1,
                      x_title=None, y_title=None, graph_title=None, name=None, show=True):
    set_style(theme)
    fig, ax = plt.subplots(figsize=figsize)
    set_grid(grid_style, ax)
    set_title(ax, x_title, y_title, graph_title)
    dotplot(x, y, ax, fig, ylim, step, show, color=color, marker='o', edgecolors='black')
    data_dict = {
        'chart-type': 'dot_plot',
        **extract_ax_data(ax, fig, x, y, name, 'visual-elements.dot points'),
        'visual-elements.bars': [],
        'visual-elements.boxplots': [],
        'visual-elements.lines': [],
        'visual-elements.scatter points': []
    }
    final_name = save_file(name, fig, data_dict)
    if show:
        plt.show()
    plt.close(fig)  # close the plot to avoid displaying
    return data_dict, final_name


def random_generate_dot_plot(x, y, x_title=None, y_title=None, graph_title=None, name=None, show=True):
    """
    Generates a dot plot with random visual parameters and returns the data dictionary and final name.

    Args:
        x (array-like): The x-values for the dot plot.
        y (array-like): The y-values for the dot plot.
        x_title (str, optional): Title for the x-axis. Defaults to None.
        y_title (str, optional): Title for the y-axis. Defaults to None.
        graph_title (str, optional): Main title for the graph. Defaults to None.
        name (str, optional): Name for saving the graph. Defaults to None.
        show (bool, optional): Whether to display the plot. Defaults to True.

    Returns:
        dict: Data dictionary containing information about the plot.
        str: The final name of the saved plot.
    """
    color_palette = ['blue', 'red', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'pink', "black"]
    color = random.choice(color_palette)
    grid_style = random.choice(["y", "none"])
    theme = 'default'
    figsize = (random.randint(6, 12), 6)
    ylim = [0, random.randint(max(y), max(y) + 5)]
    step = 1
    data_dict, final_name = generate_dot_plot(x, y, color=color, grid_style=grid_style, theme=theme, figsize=figsize,
                                              ylim=ylim, step=step,
                                              x_title=x_title, y_title=y_title, graph_title=graph_title, name=name,
                                              show=show)

    if not check_text_overlap(data_dict, figsize[1]):
        return data_dict, final_name
    os.remove(f"{final_name}.jpg")
    os.remove(f"{final_name}.json")
    if figsize[0] < 10:
        figsize = (random.randint(figsize[0] + 1, 13), 6)
        data_dict, final_name = generate_dot_plot(x, y, color=color, grid_style=grid_style, theme=theme,
                                                  figsize=figsize, ylim=ylim, step=step,
                                                  x_title=x_title, y_title=y_title, graph_title=graph_title, name=name,
                                                  show=show)
        if not check_text_overlap(data_dict, figsize[1]):
            return data_dict, final_name
        os.remove(f"{final_name}.jpg")
        os.remove(f"{final_name}.json")
    return {}, ""


def gen_all_4(save_loc=r"D:\mga_outputs"):
    """
    Generates the four types of charts and saves them to the specified location.

    Args:
        save_loc (str, optional): The location where the generated charts will be saved. Defaults to "D:\mga_outputs".

    Returns:
        None
    """
    data_dict, final_name = random_generate_line_chart(np.linspace(0, 100, 10).astype(int), np.sin(np.linspace(0, 10, 10)),
                                                       name=os.path.join(save_loc, "line"),
                                                       x_title="x_title", y_title="y_title", graph_title="graph_title")
    img_name = os.path.join(save_loc, f"{final_name}.jpg")
    boxes = get_bboxes(data_dict, gen=True)
    plot_image_with_boxes(img_name, boxes, jupyter=False)

    data_dict, final_name = random_generate_scatter_chart(np.random.rand(10) * 10, np.random.rand(10) * 10,
                                                          name=os.path.join(save_loc, "scat"))
    img_name = os.path.join(save_loc, f"{final_name}.jpg")
    boxes = get_bboxes(data_dict, gen=True)
    plot_image_with_boxes(img_name, boxes, jupyter=False)

    data_dict, final_name = random_generate_bar_chart(['A', 'B', 'C', 'D'], [10, 15, 7, 12],
                                                      name=os.path.join(save_loc, "bar"))
    img_name = os.path.join(save_loc, f"{final_name}.jpg")
    boxes = get_bboxes(data_dict, gen=True)
    plot_image_with_boxes(img_name, boxes, jupyter=False)

    data_dict, final_name = random_generate_dot_plot(np.arange(1, 6), [1, 5, 0, 6, 6],
                                                     name=os.path.join(save_loc, "dot"))
    img_name = os.path.join(save_loc, f"{final_name}.jpg")
    boxes = get_bboxes(data_dict, gen=True)
    plot_image_with_boxes(img_name, boxes, jupyter=False)


def choose_randomly(val1, val2):
    """
    Chooses a value randomly from the two provided values.

    Args:
        val1 (Any): The first value.
        val2 (Any): The second value.

    Returns:
        Any: One of the two values, chosen randomly.
    """
    if val1 is not None and val2 is not None:
        return random.choice([val1, val2])
    return val1 or val2


def merge_rows(row_x, row_y):
    """
    Merges two rows and returns a new dictionary with merged graph titles, x titles, and y titles.

    Args:
        row_x (dict): The first row containing title information.
        row_y (dict): The second row containing title information.

    Returns:
        dict: A new dictionary with merged data.
    """
    new_dict = {
        'graph_title': choose_randomly(row_x['text'].get('graph_title'), row_y['text'].get('graph_title')),
        'x_title': choose_randomly(row_x['text'].get('x_title'), row_y['text'].get('x_title')),
        'y_title': choose_randomly(row_x['text'].get('y_title'), row_y['text'].get('y_title'))
    }

    return new_dict


def generate_dynamic_data_point(df, rand_subset=5):
    """
    Generates dynamic data points from a given dataframe by sampling two rows and combining them.

    Args:
        df (pd.DataFrame): The dataframe from which to sample data points.
        rand_subset (int, optional): The maximum subset size for random sampling. Defaults to 5.

    Returns:
        list: List of x-values after sampling.
        list: List of y-values after sampling.
        dict: Dictionary containing merged titles for the graph, x-axis, and y-axis.
    """
    row_x = df.sample(n=1).iloc[0]
    row_y = df.sample(n=1).iloc[0]
    x_values = row_x['x']
    y_values = row_y['y']
    rand_start = random.randint(0, min(rand_subset, len(x_values)//3, len(y_values)//3))
    rand_end = random.randint(0, min(rand_subset, len(x_values)//3, len(y_values)//3))
    if rand_end:
        x_values = x_values[rand_start: -rand_end]
        y_values = y_values[rand_start: -rand_end]
    else:
        x_values = x_values[rand_start:]
        y_values = y_values[rand_start:]

    titels = merge_rows(row_x, row_y)
    if len(y_values) > len(x_values):
        start_index = random.randint(0, len(y_values) - len(x_values))
        y_values = y_values[start_index:start_index + len(x_values)]
    elif len(y_values) < len(x_values):
        start_index = random.randint(0, len(x_values) - len(y_values))
        x_values = x_values[start_index:start_index + len(y_values)]
    return x_values, y_values, titels


def preprocess_data_series(data_series):
    """
    Preprocesses a data series by extracting and processing the data series column of the competetion.

    Args:
        data_series (pd.DataFrame): The dataframe containing the data series.

    Returns:
        pd.DataFrame: The preprocessed dataframe with additional columns for x-values, y-values, and text.
    """
    data_series['data-series'] = data_series['data-series'].apply(safe_literal_eval)
    if "Unnamed: 2" in data_series.columns:
        data_series["Unnamed: 2"] = data_series["Unnamed: 2"].apply(safe_literal_eval)
        data_series['text'] = data_series["Unnamed: 2"].apply(extract_titles)
    else:
        data_series['text'] = data_series['text'].apply(safe_literal_eval)
    data_series['x'] = data_series['data-series'].apply(
        lambda lst: [d['x'] for d in lst if isinstance(d, dict) and 'x' in d and 'y' in d])
    data_series['y'] = data_series['data-series'].apply(
        lambda lst: [d['y'] for d in lst if isinstance(d, dict) and 'x' in d and 'y' in d])
    return data_series


def create_long_y_df(data_series, include_titles=True):
    """
      Creates a dataframe of long y-values from a given data series.

      Args:
          data_series (pd.DataFrame): The dataframe containing the data series.
          include_titles (bool, optional): Whether to include titles in the long y-values. Defaults to True.

      Returns:
          list: List of long strings combining x-values, y-values, and titles (if included).
      """
    all_xes = set(chain.from_iterable(data_series["x"].to_list()))
    all_xes_str = {x for x in all_xes if not is_numeric(x)}
    sorted_all_xes_str = sorted(all_xes_str, key=len)[-300:]

    all_ys = set(chain.from_iterable(data_series["y"].to_list()))
    all_ys_str = {y for y in all_ys if not is_numeric(y)}
    sorted_all_ys_str = sorted(all_ys_str, key=len)[-100:]

    sorted_all_titles = []
    if include_titles:
        titles_df = data_series["text"].apply(pd.Series)
        all_titles = {title.replace("\n", "") for title in titles_df["graph_title"] if title}
        sorted_all_titles = sorted(all_titles, key=len)[-5000:-1500]

    long_strings = sorted_all_xes_str + sorted_all_ys_str + sorted_all_titles
    return long_strings


def postprocess_data_gen(data_dict, final_name, data_list, only_plot_area=False):
    """
    Post-processes the generated data by appending it to a list and creating annotations.

    Args:
        data_dict (dict): The dictionary containing information about the plot.
        final_name (str): The final name of the saved plot.
        data_list (list): List of data dictionaries.
        only_plot_area (bool, optional): If set to True, only the plot area will be annotated. Defaults to False.

    Returns:
        dict: The updated data dictionary.
        list: The updated list of data dictionaries.
    """
    data_dict["name"] = os.path.basename(final_name)
    data_list = np.append(data_list, data_dict)
    annotation_to_labels(os.path.join(generated_imgs, f"{final_name}.jpg"),
                         data_dict, False, generated_imgs, gen=True, only_plot_area=only_plot_area)
    return data_dict, data_list


def generate_n_plots(data_series, generated_imgs, n=2, data_types=["line", "scat", "dot", "bar"], show=False,
                     clear_list=False):
    """
    Generate 'n' plots of specified data types and save them in the provided directory.

    Args:
        data_series (pd.DataFrame): The dataframe containing the data series.
        generated_imgs (str): Directory to save the generated images.
        n (int, optional): Number of plots to generate. Defaults to 2.
        data_types (list, optional): Types of data plots to generate. Defaults to ["line", "scat", "dot", "bar"].
        show (bool, optional): Whether to display the plot. Defaults to False.
        clear_list (bool, optional): Whether to clear the data list after processing. Defaults to False.

    Returns:
        np.array: Array containing dictionaries of generated data plots.
    """
    os.makedirs(generated_imgs, exist_ok=True)
    data_list = np.array([])
    for i in tqdm(range(n)):
        try:
            x_data_dynamic, y_data_dynamic, titels = generate_dynamic_data_point(data_series)
            if len(x_data_dynamic) < 7 or len(y_data_dynamic) < 7 or len(y_data_dynamic) > 75:
                continue
            if not isinstance(x_data_dynamic[0], str) and "scat" in data_types:
                data_dict, final_name = random_generate_scatter_chart(x_data_dynamic, y_data_dynamic,
                                                                      name=os.path.join(generated_imgs, "scat"),
                                                                      **titels, show=show)
                data_dict, data_list = postprocess_data_gen(data_dict, final_name, data_list)
                if show:
                    img_name = os.path.join(generated_imgs, f"{final_name}.jpg")
                    boxes = get_bboxes(data_dict, gen=True)
                    plot_image_with_boxes(img_name, boxes, jupyter=False)
            if len(y_data_dynamic) > 30:
                continue
            if not isinstance(x_data_dynamic[0], str):
                x_data_dynamic = [int(val) for val in x_data_dynamic]
            x_data_dynamic_arr = np.array([str(x) for x in x_data_dynamic])
            if "bar" in data_types:
                data_dict, final_name = random_generate_bar_chart(x_data_dynamic_arr, np.array(y_data_dynamic),
                                                                  name=os.path.join(generated_imgs, "bar"), **titels,
                                                                  show=show)
                data_dict, data_list = postprocess_data_gen(data_dict, final_name, data_list)
                if show:
                    img_name = os.path.join(generated_imgs, f"{final_name}.jpg")
                    boxes = get_bboxes(data_dict, gen=True)
                    plot_image_with_boxes(img_name, boxes, jupyter=False)

            if len(y_data_dynamic) > 20:
                continue

            if not isinstance(x_data_dynamic[0], str):
                differences = np.diff(x_data_dynamic)
                if len(set(differences)) > 1:
                    continue
            if "line" in data_types:
                data_dict, final_name = random_generate_line_chart(x_data_dynamic, y_data_dynamic,
                                                                   name=os.path.join(generated_imgs, "line"), **titels,
                                                                   show=show)
                data_dict, data_list = postprocess_data_gen(data_dict, final_name, data_list)
                if show:
                    img_name = os.path.join(generated_imgs, f"{final_name}.jpg")
                    boxes = get_bboxes(data_dict, gen=True)
                    plot_image_with_boxes(img_name, boxes, jupyter=False)

            if len(x_data_dynamic_arr) > 15:
                start_index = random.randint(0, len(x_data_dynamic_arr) - 10)
                x_data_dynamic_arr = x_data_dynamic_arr[start_index:start_index + len(x_data_dynamic_arr)]
            if "dot" in data_types:
                data_dict, final_name = random_generate_dot_plot(x_data_dynamic_arr, np.random.randint(1, 13, 10),
                                                                 name=os.path.join(generated_imgs, "dot"), **titels,
                                                                 show=show)
                if data_dict:
                    data_dict, data_list = postprocess_data_gen(data_dict, final_name, data_list)
                    if show:
                        img_name = os.path.join(generated_imgs, f"{final_name}.jpg")
                        boxes = get_bboxes(data_dict, gen=True)
                        plot_image_with_boxes(img_name, boxes, jupyter=False)
            if clear_list:
                data_list = np.array([])
        except Exception as err:
            print(err)
            print(x_data_dynamic)
            print(y_data_dynamic)
    return data_list


def generate_cont_lines(data_series, generated_imgs, n=2, data_types=["line"], show=False,
                     clear_list=False):
    """
    Generate 'n' continuous line plots and save them in the provided directory.

    Args:
        data_series (pd.DataFrame): The dataframe containing the data series.
        generated_imgs (str): Directory to save the generated images.
        n (int, optional): Number of plots to generate. Defaults to 2.
        data_types (list, optional): Types of data plots to generate. Defaults to ["line"].
        show (bool, optional): Whether to display the plot. Defaults to False.
        clear_list (bool, optional): Whether to clear the data list after processing. Defaults to False.

    Returns:
        np.array: Array containing dictionaries of generated data plots.
    """
    os.makedirs(generated_imgs, exist_ok=True)
    data_list = np.array([])
    for i in tqdm(range(n)):
        try:
            x_data_dynamic, y_data_dynamic, titels = generate_dynamic_data_point(data_series)
            if len(x_data_dynamic) < 5 or len(y_data_dynamic) < 5 or len(y_data_dynamic) > 20:
                continue

            if "line" in data_types:
                if not isinstance(x_data_dynamic[0], str):
                    x_data_dynamic = [str(int(x)) for x in x_data_dynamic]
                elif random.random() < 1:
                    x_data_dynamic = [insert_newline_at_random_space(x) for x in x_data_dynamic]
                data_dict, final_name = random_generate_line_chart(x_data_dynamic, y_data_dynamic,
                                                                   name=os.path.join(generated_imgs, "line_cont"),
                                                                   **titels, show=show, cont=True)
                data_dict, data_list = postprocess_data_gen(data_dict, final_name, data_list)
                if show:
                    img_name = os.path.join(generated_imgs, f"{final_name}.jpg")
                    boxes = get_bboxes(data_dict, gen=True)
                    plot_image_with_boxes(img_name, boxes, jupyter=False)

            if clear_list:
                data_list = np.array([])
        except Exception as err:
            print(err)
            print(x_data_dynamic)
            print(y_data_dynamic)
    return data_list


def generate_n_long_plots(data_series, generated_imgs, n=2, data_types=["bar"], show=False):
    """
    Generate 'n' plots with long x-axis values and save them in the provided directory.

    Args:
        data_series (pd.DataFrame): The dataframe containing the data series.
        generated_imgs (str): Directory to save the generated images.
        n (int, optional): Number of plots to generate. Defaults to 2.
        data_types (list, optional): Types of data plots to generate. Defaults to ["bar"].
        show (bool, optional): Whether to display the plot. Defaults to False.

    Returns:
        np.array: Array containing dictionaries of generated data plots.
    """
    os.makedirs(generated_imgs, exist_ok=True)
    data_list = np.array([])
    long_strings = list(create_long_y_df(data_series))
    for i in tqdm(range(n)):
        try:
            x_data_dynamic, y_data_dynamic, titels = generate_dynamic_data_point(data_series)
            if len(y_data_dynamic) < 4 or len(y_data_dynamic) > 25 or isinstance(y_data_dynamic[0], str):
                continue
            x_data_dynamic = random.sample(long_strings, len(y_data_dynamic))
            if isinstance(x_data_dynamic[0], str) and isinstance(y_data_dynamic[0], str):
                continue
            if not isinstance(x_data_dynamic[0], str):
                x_data_dynamic = [int(val) for val in x_data_dynamic]
            x_data_dynamic_arr = np.array([break_string(str(x), random.randint(35, 75)) for x in x_data_dynamic])

            if "bar" in data_types:
                data_dict, final_name = random_generate_bar_chart(x_data_dynamic_arr, np.array(y_data_dynamic),
                                                                  name=os.path.join(generated_imgs, "bar_long"), **titels,
                                                                  show=show, horizontal_prob=1, long=True)
                data_dict, data_list = postprocess_data_gen(data_dict, final_name, data_list)
                if show:
                    img_name = os.path.join(generated_imgs, f"{final_name}.jpg")
                    boxes = get_bboxes(data_dict, gen=True)
                    plot_image_with_boxes(img_name, boxes, jupyter=False)
            data_list = []
        except Exception as err:
            print(err)
            print(x_data_dynamic)
            print(y_data_dynamic)
    return data_list


def generate_random_bg_plot(x_title=None, y_title=None, graph_title=None, name="", show=False, folder=""):
    """
    Generate a random background plot with optional titles and save in the provided directory.

    Args:
        x_title (str, optional): Title for the x-axis. Defaults to None.
        y_title (str, optional): Title for the y-axis. Defaults to None.
        graph_title (str, optional): Main title for the graph. Defaults to None.
        name (str, optional): Name for the saved plot. Defaults to "".
        show (bool, optional): Whether to display the plot. Defaults to False.
        folder (str, optional): Directory to save the generated image. Defaults to "".
    """
    if folder != "":
        os.makedirs(folder, exist_ok=True)
    figsize_w = random.randint(4, 13)
    figsize_h = random.randint(max(4, int(figsize_w / 3.7)), min(int(figsize_w * 1.5), 9))

    theme_choices = ['dark_background', 'default', 'grayscale', 'dark_gray']
    theme = np.random.choice(theme_choices, p=[0.1, 0.6, 0.2, 0.1])
    grid_style = np.random.choice(["both", "x", "y", "none"], p=[0.1, 0.1, 0.1, 0.7])

    set_style(theme)
    fig, ax = plt.subplots(figsize=(figsize_w, figsize_h))
    set_grid(grid_style, ax)

    ax.tick_params(left=False, right=False, labelleft=False,
                   labelbottom=False, bottom=False)
    set_title(ax, x_title, y_title, graph_title)
    set_ax_loc_rotate(ax, random.choice([True, False]))
    data_dict = extract_bg_data(ax, fig)
    final_name = save_file(os.path.join(folder, name), fig, {})
    postprocess_data_gen(data_dict, final_name, [], True)
    if show:
        plt.show()


if __name__ == "__main__": # example for runs
    data_series_path = r"D:\MGA\data_series.csv"
    data_series = preprocess_data_series(pd.read_csv(data_series_path))
    generated_imgs = r"D:\MGA\gen_charts_new_line_cont"
    generated_imgs_bg = r"D:\MGA\bg_gen_charts"


    # data_types = ["bar"]
    # data_list = generate_n_long_plots(data_series, generated_imgs, n=5000, data_types=data_types,
    #                              show=False)
    # df = pd.DataFrame.from_records(data_list)
    # df.to_csv(os.path.join(generated_imgs, "generated_data.csv"))

    data_types = ["line"]
    data_list = generate_cont_lines(data_series, generated_imgs, n=8000, data_types=data_types,
                                 show=False, clear_list=True)
    df = pd.DataFrame.from_records(data_list)
    df.to_csv(os.path.join(generated_imgs, "generated_data.csv"))



    # data_types = ["line"]
    # data_list = generate_n_plots(data_series, generated_imgs, n=5000, data_types=data_types,
    #                              show=False, clear_list=True)
    # df = pd.DataFrame.from_records(data_list)
    # df.to_csv(os.path.join(generated_imgs, "generated_data.csv"))

    # x_data_dynamic, y_data_dynamic, titels = generate_dynamic_data_point(data_series)
    #
    # data_list = generate_n_plots(data_series, generated_imgs, n=500, data_types=["line", "scat", "bar"],
    #                              show=False, clear_list=True)
    # df = pd.DataFrame.from_records(data_list)
    # df.to_csv(os.path.join(generated_imgs, "generated_data.csv"))

    # for i in tqdm(range(4000)):
    #     x_data_dynamic, y_data_dynamic, titels = generate_dynamic_data_point(data_series)
    #     generate_random_bg_plot(**titels, name="bg", show=False, folder=generated_imgs_bg)

    # data_dict, final_name = random_generate_bar_chart(["A", "B", "C", "D", "E", "F"], [1,2,3,4,5,6],
    #                                                   name=os.path.join(generated_imgs, "bar"))
    # img_name = os.path.join(generated_imgs, f"{final_name}.jpg")
    # boxes = get_bboxes(data_dict)
    # plot_image_with_boxes(img_name, boxes, jupyter=False)

    # data_dict, final_name = random_generate_line_chart(np.linspace(0, 100, 10).astype(int), np.sin(np.linspace(0, 10, 10)),
    #                                                    name=os.path.join(generated_imgs, "line"),
    #                                                    x_title="x_title", y_title="y_title", graph_title="graph_title")
    # img_name = os.path.join(generated_imgs, f"{final_name}.jpg")
    # boxes = get_bboxes(data_dict, gen=True)
    # plot_image_with_boxes(img_name, boxes, jupyter=False)
