import matplotlib.ticker as ticker
from sklearn.linear_model import LinearRegression
import pandas as pd
from tqdm import tqdm
import random
from data_generation.gen_ax_fig import *
from plot_functions.mga_plt import plot_image_with_boxes
from utils.util_funcs import get_bboxes, safe_literal_eval, annotation_to_labels


def generate_line_chart(x, y, line_color='blue', grid_style="both", x_title=None, y_title=None, graph_title=None,
                        theme="default",
                        line_style='-', marker_style=None, figsize=(6, 4), name=None, rotate=False, show=True):
    set_style(theme)
    fig, ax = plt.subplots(figsize=figsize)
    set_grid(grid_style, ax)
    set_title(ax, x_title, y_title, graph_title)
    ax.plot(x, y, color=line_color, linestyle=line_style, marker=marker_style)
    ax.set_xticks(x)
    ax.set_xticklabels(x, rotation=0 if not rotate else 45, ha='right')  # Rotate for better visibility if needed
    set_ax_loc_rotate(ax, rotate)
    data_dict = {
        'chart-type': 'line',
        **extract_ax_data(ax, fig, x, y, name, data_type="visual-elements.lines"),
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


def random_generate_line_chart(x, y, x_title=None, y_title=None, graph_title=None, name=None, show=True):
    color_palette = ['blue', 'red', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'pink', "black"]
    grid_style_choices = ["both", "x", "y", "none"]
    theme_choices = ['dark_background', 'default', 'grayscale', 'dark_gray']
    line_color = random.choice(color_palette)
    grid_style = np.random.choice(grid_style_choices, p=[0.2, 0.2, 0.2, 0.4])
    theme = np.random.choice(theme_choices, p=[0.2, 0.5, 0.2, 0.1])
    if theme == 'dark_background' and line_color == "black":
        theme = 'default'
    line_style = "-"
    marker_style = np.random.choice(['o', 'x', None], p=[0.25, 0.25, 0.5])
    figsize_w = random.randint(4, 13)
    figsize_h = random.randint(max(4, int(figsize_w / 3.7)), min(int(figsize_w * 1.5), 9))
    figsize = (figsize_w, figsize_h)
    data_dict, final_name = generate_line_chart(x, y, line_color=line_color, grid_style=grid_style, x_title=x_title,
                                                y_title=y_title, graph_title=graph_title,
                                                theme=theme, line_style=line_style, marker_style=marker_style,
                                                figsize=figsize, name=name, show=show)
    if not check_text_overlap(data_dict):
        return data_dict, final_name
    os.remove(f"{final_name}.jpg")
    os.remove(f"{final_name}.json")
    if figsize_w < 10:
        figsize_w = random.randint(figsize_w + 2, 13)
        figsize_h = random.randint(max(3, int(figsize_w / 3.7)), min(int(figsize_w * 1.5), 9))
        figsize = (figsize_w, figsize_h)
    if random.random() < 0.5:
        data_dict, final_name = generate_line_chart(x, y, line_color=line_color, grid_style=grid_style, x_title=x_title,
                                                    y_title=y_title, graph_title=graph_title,
                                                    theme=theme, line_style=line_style, marker_style=marker_style,
                                                    figsize=figsize, name=name, show=show)
        if not check_text_overlap(data_dict):
            return data_dict, final_name
        os.remove(f"{final_name}.jpg")
        os.remove(f"{final_name}.json")
    return generate_line_chart(x, y, line_color=line_color, grid_style=grid_style, x_title=x_title, y_title=y_title,
                               graph_title=graph_title,
                               theme=theme, line_style=line_style, marker_style=marker_style, figsize=figsize,
                               name=name, rotate=45, show=show)


def generate_scatter_chart(x, y, color='blue', grid_style="both", theme="white", show_regression_line=False,
                           x_title=None, y_title=None, graph_title=None, figsize=(6, 4), name=None, show=True):
    set_style(theme)
    fig, ax = plt.subplots()
    set_grid(grid_style, ax)
    set_title(ax, x_title, y_title, graph_title)
    ax.scatter(x, y, label='Sample Scatter', color=color, marker='o')
    set_ax_loc_rotate(ax, False)

    if show_regression_line:
        model = LinearRegression().fit(np.array(x).reshape(-1, 1), y)
        y_pred = model.predict(np.array(x).reshape(-1, 1))
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
    color_palette = ['blue', 'red', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'pink', "black"]
    color = random.choice(color_palette)
    grid_style = np.random.choice(["both", "x", "y", "none"], p=[0.2, 0.2, 0.2, 0.4])
    theme = np.random.choice(['dark_background', 'default', 'grayscale', 'dark_gray'], p=[0.2, 0.5, 0.2, 0.1])
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
                       show=True):
    set_style(theme)
    fig, ax = plt.subplots()
    set_grid(grid_style, ax)
    set_title(ax, x_title, y_title, graph_title)
    set_ax_loc_rotate(ax, rotate)
    if rotate:
        ax.set_xticklabels(categories, rotation=45)  # Rotate for better visibility if needed

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
    plt.close(fig)  # Close the plot to avoid displaying
    return data_dict, final_name


def random_generate_bar_chart(categories, values, x_title=None, y_title=None, graph_title=None, name=None, show=True):
    color_palette = ['blue', 'red', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'pink', "black"]
    color = random.choice(color_palette)
    grid_style = np.random.choice(["both", "x", "y", "none"], p=[0.2, 0.2, 0.2, 0.4])
    theme = np.random.choice(['dark_background', 'default', 'grayscale', 'dark_gray'], p=[0.2, 0.5, 0.2, 0.1])
    if theme == 'dark_background' and color == "black":
        theme = 'default'
    orientation = "horizontal" if random.random() < 0.1 else "vertical"  # 10% chance for horizontal
    if os.sep in name:
        name = os.path.join(os.path.dirname(name), f"{orientation}_{os.path.basename(name)}")
    else:
        name = f"{orientation}_{name}"
    figsize_w = random.randint(4, 13)
    figsize_h = random.randint(max(4, int(figsize_w / 3.7)), min(int(figsize_w * 1.5), 9))
    figsize = (figsize_w, figsize_h)
    data_dict, final_name = generate_bar_chart(categories, values, color=color, grid_style=grid_style, theme=theme,
                                               orientation=orientation,
                                               x_title=x_title, y_title=y_title, graph_title=graph_title,
                                               figsize=figsize, name=name, show=show)
    if not check_text_overlap(data_dict):
        return data_dict, final_name
    os.remove(f"{final_name}.jpg")
    os.remove(f"{final_name}.json")
    figsize_w = random.randint(figsize_w + 2, 14)
    figsize_h = random.randint(max(5, int(figsize_w / 3.7)), min(int(figsize_w * 1.5), 9))
    figsize = (figsize_w, figsize_h)
    if random.random() < 0.5 and figsize_w < 10:
        data_dict, final_name = generate_bar_chart(categories, values, color=color, grid_style=grid_style, theme=theme,
                                                   orientation=orientation,
                                                   x_title=x_title, y_title=y_title, graph_title=graph_title,
                                                   figsize=figsize, name=name, show=show)
        if not check_text_overlap(data_dict):
            return data_dict, final_name
        os.remove(f"{final_name}.jpg")
        os.remove(f"{final_name}.json")
    return generate_bar_chart(categories, values, color=color, grid_style=grid_style, theme=theme,
                              orientation=orientation,
                              x_title=x_title, y_title=y_title, graph_title=graph_title, figsize=figsize, name=name,
                              rotate=True, show=show)


def dotplot(xs, ys, ax, fig, ylim, step, show=True, **args):
    """
    Function that creates dot plots.
    """
    fig_size = fig.get_size_inches()
    scatter_x = []  # x values
    scatter_y = []  # corresponding y values
    for x, y in zip(xs, ys):
        for z in range(ylim[0], y, step):
            scatter_x.append(x)
            scatter_y.append(z + step / 2)

    # draw dot plot using scatter()
    factor = (ylim[1] - ylim[0]) / 10 / step / (fig_size[1] / 6)
    ax.scatter(scatter_x, scatter_y, s=fig.dpi * 10 / factor ** 2, zorder=3, **args)

    # Show all unique x-values
    ax.set_xticks(xs)

    # Change major ticks to show every 1 value
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(step))

    # Margin so dots fit on screen
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
    plt.close(fig)  # Close the plot to avoid displaying
    return data_dict, final_name


def random_generate_dot_plot(x, y, x_title=None, y_title=None, graph_title=None, name=None, show=True):
    # Standard color palette
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

    if not check_text_overlap(data_dict):
        return data_dict, final_name
    os.remove(f"{final_name}.jpg")
    os.remove(f"{final_name}.json")
    if figsize[0] < 10:
        figsize = (random.randint(figsize[0] + 1, 13), 6)
        data_dict, final_name = generate_dot_plot(x, y, color=color, grid_style=grid_style, theme=theme,
                                                  figsize=figsize, ylim=ylim, step=step,
                                                  x_title=x_title, y_title=y_title, graph_title=graph_title, name=name,
                                                  show=show)
        if not check_text_overlap(data_dict):
            return data_dict, final_name
        os.remove(f"{final_name}.jpg")
        os.remove(f"{final_name}.json")
    return {}, ""


def gen_all_4(save_loc = r"D:\mga_outputs"):
    data_dict, final_name = random_generate_line_chart(np.linspace(0, 100, 10).astype(int), np.sin(np.linspace(0, 10, 10)),
                                                       name=os.path.join(save_loc, "line"),
                                                       x_title="x_title", y_title="y_title", graph_title="graph_title")
    img_name = os.path.join(save_loc, f"{final_name}.jpg")
    boxes = get_bboxes(data_dict)
    plot_image_with_boxes(img_name, boxes, jupyter=False)

    data_dict, final_name = random_generate_scatter_chart(np.random.rand(10) * 10, np.random.rand(10) * 10,
                                                          name=os.path.join(save_loc, "scat"))
    img_name = os.path.join(save_loc, f"{final_name}.jpg")
    boxes = get_bboxes(data_dict)
    plot_image_with_boxes(img_name, boxes, jupyter=False)

    data_dict, final_name = random_generate_bar_chart(['A', 'B', 'C', 'D'], [10, 15, 7, 12],
                                                      name=os.path.join(save_loc, "bar"))
    img_name = os.path.join(save_loc, f"{final_name}.jpg")
    boxes = get_bboxes(data_dict)
    plot_image_with_boxes(img_name, boxes, jupyter=False)

    data_dict, final_name = random_generate_dot_plot(np.arange(1, 6), [1, 5, 0, 6, 6],
                                                     name=os.path.join(save_loc, "dot"))
    img_name = os.path.join(save_loc, f"{final_name}.jpg")
    boxes = get_bboxes(data_dict)
    plot_image_with_boxes(img_name, boxes, jupyter=False)


def choose_randomly(val1, val2):
    if val1 is not None and val2 is not None:
        return random.choice([val1, val2])
    return val1 or val2


def merge_rows(row_x, row_y):
    new_dict = {
        'graph_title': choose_randomly(row_x['text'].get('graph_title'), row_y['text'].get('graph_title')),
        'x_title': choose_randomly(row_x['text'].get('x_title'), row_y['text'].get('x_title')),
        'y_title': choose_randomly(row_x['text'].get('y_title'), row_y['text'].get('y_title'))
    }

    return new_dict


def generate_dynamic_data_point(df):
    row_x = df.sample(n=1).iloc[0]
    row_y = df.sample(n=1).iloc[0]
    x_values = row_x['x']
    y_values = row_y['y']
    titels = merge_rows(row_x, row_y)
    if len(y_values) > len(x_values):
        start_index = random.randint(0, len(y_values) - len(x_values))
        y_values = y_values[start_index:start_index + len(x_values)]
    elif len(y_values) < len(x_values):
        x_values = x_values[:len(y_values)]
    return x_values, y_values, titels


def preprocess_data_series(data_series):
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


def postprocess_data_gen(data_dict, final_name, data_list):
    data_dict["name"] = os.path.basename(final_name)
    data_list = np.append(data_list, data_dict)
    annotation_to_labels(os.path.join(generated_imgs, f"{final_name}.jpg"),
                         data_dict, False, generated_imgs)
    return data_dict, data_list


def generate_n_plots(data_series, generated_imgs, n=2, data_types=["line", "scat", "dot", "bar"], show=False):
    os.makedirs(generated_imgs, exist_ok=True)
    data_list = np.array([])
    for i in tqdm(range(n)):
        try:
            x_data_dynamic, y_data_dynamic, titels = generate_dynamic_data_point(data_series)
            if len(x_data_dynamic) < 3 or len(y_data_dynamic) < 3 or len(y_data_dynamic) > 30:
                continue
            if not isinstance(x_data_dynamic[0], str) and "scat" in data_types:
                data_dict, final_name = random_generate_scatter_chart(x_data_dynamic, y_data_dynamic,
                                                                      name=os.path.join(generated_imgs, "scat"),
                                                                      **titels, show=show)
                data_dict, data_list = postprocess_data_gen(data_dict, final_name, data_list)
            if len(y_data_dynamic) > 20:
                continue
            if not isinstance(x_data_dynamic[0], str):
                x_data_dynamic = [int(val) for val in x_data_dynamic]
            x_data_dynamic_arr = np.array([str(x) for x in x_data_dynamic])
            if "bar" in data_types:
                data_dict, final_name = random_generate_bar_chart(x_data_dynamic_arr, np.array(y_data_dynamic),
                                                                  name=os.path.join(generated_imgs, "bar"), **titels,
                                                                  show=show)
                data_dict, data_list = postprocess_data_gen(data_dict, final_name, data_list)

            if len(y_data_dynamic) > 15:
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

            if len(x_data_dynamic_arr) > 10:
                start_index = random.randint(0, len(x_data_dynamic_arr) - 10)
                x_data_dynamic_arr = x_data_dynamic_arr[start_index:start_index + len(x_data_dynamic_arr)]
            if "dot" in data_types:
                data_dict, final_name = random_generate_dot_plot(x_data_dynamic_arr, np.random.randint(1, 13, 10),
                                                                 name=os.path.join(generated_imgs, "dot"), **titels,
                                                                 show=show)
                data_dict, data_list = postprocess_data_gen(data_dict, final_name, data_list)
        except:
            print(x_data_dynamic)
            print(y_data_dynamic)
    return data_list


if __name__ == "__main__":
    data_series_path = r"D:\MGA\data_series.csv"
    data_series = preprocess_data_series(pd.read_csv(data_series_path))
    generated_imgs = r"D:\MGA\gen"
    data_list = generate_n_plots(data_series, generated_imgs, n=2, data_types=["line", "scat", "dot", "bar"], show=False)
    df = pd.DataFrame.from_records(data_list)
    df.to_csv(os.path.join(generated_imgs, "generated_data.csv"))
