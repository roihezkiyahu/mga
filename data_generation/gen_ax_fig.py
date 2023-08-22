import matplotlib.pyplot as plt
import numpy as np
import json
import os
import re
from utils.util_funcs import safe_literal_eval


def filter_ticks(ticks, lower_bound, upper_bound):
    """Filter out ticks that are outside the given bounds."""
    return [tick for tick in ticks if lower_bound <= tick <= upper_bound]


def filter_texts(texts, plot_bbox, pixel_dims):
    """
    Filters out texts whose polygons are not entirely within the plot bounding box.
    """
    def is_inside_bbox(polygon, bbox, pixel_dims):
        """Check if a polygon is inside a bounding box."""
        return (((polygon['x0']+polygon['x1'])/2 <= bbox.x1*pixel_dims[1] and (polygon['x0']+polygon['x1'])/2 >= bbox.x0*pixel_dims[1]-4) or
                ((polygon['y2']+polygon['y0'])/2 <= bbox.y1*pixel_dims[0]+8 and (polygon['y2']+polygon['y0'])/2 >= bbox.y0*pixel_dims[0]+1))
    return [text for text in texts if is_inside_bbox(text['polygon'], plot_bbox, pixel_dims)]


def bbox_to_polygon(bbox, pixel_dims, fig_inch_size, shift = 2):
    return {
        'x0': bbox.x0*pixel_dims[0]-shift,
        'x1': bbox.x1*pixel_dims[0],
        'x2': bbox.x1*pixel_dims[0],
        'x3': bbox.x0*pixel_dims[0]-shift,
        'y0': (fig_inch_size[1] - bbox.y1)*pixel_dims[1]-shift,
        'y1': (fig_inch_size[1] - bbox.y1)*pixel_dims[1]-shift,
        'y2': (fig_inch_size[1] - bbox.y0)*pixel_dims[1],
        'y3': (fig_inch_size[1] - bbox.y0)*pixel_dims[1]
    }


def extract_texts(ax, fig):
    ax.figure.canvas.draw()
    pixel_dims = [fig.dpi, fig.dpi]
    fig_inch_size = fig.get_size_inches()
    texts = []
    for label, role in zip(ax.get_xticklabels() + ax.get_yticklabels(),
                           ['tick_label'] * (len(ax.get_xticklabels()) + len(ax.get_yticklabels()))):
        bbox = label.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
        texts.append({
            'id': len(texts),
            'polygon': bbox_to_polygon(bbox, pixel_dims, fig_inch_size),
            'text': label.get_text(),
            'role': role
        })

    for axis, label in [('x', ax.xaxis.label), ('y', ax.yaxis.label)]:
        if label.get_text() != '':
            bbox = label.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
            texts.append({
                'id': len(texts),
                'polygon': bbox_to_polygon(bbox, pixel_dims, fig_inch_size),
                'text': label.get_text(),
                'role': f'{axis}_axis_title'
            })

    for text in ax.texts:
        bbox = text.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
        texts.append({
            'id': len(texts),
            'polygon': bbox_to_polygon(bbox, pixel_dims, fig_inch_size),
            'text': text.get_text(),
            'role': 'annotation'
        })

    return texts


def extract_ax_data(ax, fig, x, y, name=None, data_type="visual-elements.lines", bar_chart_type=""):
    fig_inch_size = fig.get_size_inches()
    pixel_dims = fig_inch_size * fig.dpi
    is_categorical = isinstance(x[0], str)
    if is_categorical:
        x_numeric = list(range(len(x)))
    else:
        x_numeric = x
    bbox = ax.get_position()
    x_ticks_location = [ax.transData.transform((tick, 0))[0] for tick in ax.get_xticks()]
    y_ticks_location = []
    for tick, label in zip(ax.get_yticks(), ax.get_yticklabels()):
        width = label.get_window_extent().width
        y_ticks_location.append(ax.transData.transform((bbox.x0 - width, tick))[1])
    image_height = pixel_dims[1]
    y_ticks_location = [image_height - loc for loc in y_ticks_location]
    off_set = 0.5 if data_type == "visual-elements.dot points" else 0
    data_series_locations = [{"x": ax.transData.transform((x_val, y_val))[0],
                              "y": image_height - ax.transData.transform((x_val, y_val - off_set))[1]}
                             for x_val, y_val in zip(x_numeric, y)]
    if is_categorical and bar_chart_type == "vertical_bar":
        x_ticks_location = [point["x"] for point in data_series_locations]
    plot_bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

    x_ticks_location = filter_ticks(x_ticks_location, bbox.x0 * pixel_dims[0], bbox.x1 * pixel_dims[0])
    y_ticks_location = filter_ticks(y_ticks_location, bbox.y0 * pixel_dims[1], bbox.y1 * pixel_dims[1])
    ar = fig_inch_size[0] / fig_inch_size[1] if data_type == 'visual-elements.dot points' else 3
    return {
        'text': filter_texts(extract_texts(ax, fig), plot_bbox, [fig.dpi, fig.dpi]),
        data_type: data_series_locations,
        'data-series': [{"x": x_val, "y": y_val} for x_val, y_val in zip(x, y)],
        'plot-bb.height': bbox.height * pixel_dims[1],
        'plot-bb.width': bbox.width * pixel_dims[0],
        'plot-bb.x0': bbox.x0 * pixel_dims[0],
        'plot-bb.y0': bbox.y0 * pixel_dims[1] + (3 - ar) * 3,
        'axes.x-axis.ticks': [
            {"id": i, "tick_pt": {"x": loc, "y": bbox.y0 * pixel_dims[1] + bbox.height * pixel_dims[1] + (3 - ar) * 3}}
            for i, loc in enumerate(x_ticks_location)],
        'axes.x-axis.tick-type': "markers",
        'axes.x-axis.values-type': "numerical",
        'axes.y-axis.ticks': [{"id": i, "tick_pt": {"x": bbox.x0 * pixel_dims[0], "y": loc}}
                              for i, loc in enumerate(y_ticks_location)],
        'name': name
    }


def set_style(theme):
    if theme == "dark_gray":
        plt.rcParams['figure.facecolor'] = '#494b47'
        plt.rcParams['axes.facecolor'] = '#494b47'
        plt.rcParams['axes.edgecolor'] = '#000000'
        plt.rcParams['axes.labelcolor'] = '#000000'
        plt.rcParams['xtick.color'] = '#000000'
        plt.rcParams['ytick.color'] = '#000000'
        plt.rcParams['grid.color'] = '#4A4A4A'
    else:
        plt.style.use(theme)


def set_grid(grid_style, ax):
    if grid_style == "both":
        ax.grid(True)
    elif grid_style == "x":
        ax.xaxis.grid(True)
    elif grid_style == "y":
        ax.yaxis.grid(True)
    else:
        ax.grid(False)


def set_title(ax, x_title, y_title, graph_title):
    if x_title:
        ax.set_xlabel(x_title)
    if y_title:
        ax.set_ylabel(y_title)
    if graph_title:
        ax.set_title(graph_title)


def set_x_ticks(ax, x, rotate):
    ax.set_xticks(x)
    rotation_angle = 45 if rotate else 0
    ax.set_xticklabels(x, rotation=rotation_angle)


def set_ax_loc_rotate(ax, rotate):
    rand_noise = np.random.uniform(0.075, 0.2)
    ax.set_position([rand_noise, rand_noise, 1 - rand_noise * 2, 1 - rand_noise * 2])
    if rotate:
        ax.margins(0.05)
        ax.set_position([0.1 + rand_noise, 0.1 + rand_noise, 0.8 - rand_noise * 2, 0.8 - rand_noise * 2])


def default_serialize(obj):
    """Default JSON serializer."""
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
        np.int16, np.int32, np.int64, np.uint8,
        np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32,
        np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):  # Handle numpy arrays
        return obj.tolist()
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')


def save_file(name, fig, data_dict):
    if not name:
        return
    base_name = name
    counter = 0

    # Extract the potential number from the end of the name
    match = re.search(r'(\d+)$', base_name)
    if match:
        counter = int(match.group(1))
        base_name = name[:-len(match.group(1))]
    final_name = f"{base_name}{counter}"
    # Construct new name by incrementing counter until a unique filename is found
    while os.path.exists(f"{final_name}.jpg") or os.path.exists(f"{final_name}data.json"):
        counter += 1
        final_name = f"{base_name}{counter}"

    # Save the figure and JSON with custom serialization for NumPy types
    fig.savefig(f"{final_name}.jpg")
    with open(f'{final_name}.json', 'w') as file:
        json.dump(data_dict, file, indent=4, default=default_serialize)
    return final_name


def is_overlapping(rect1, rect2):
    return not (rect1['x0'] > rect2['x1'] or
                rect1['x1'] < rect2['x0'] or
                rect1['y0'] > rect2['y1'] or
                rect1['y1'] < rect2['y0'])


def check_text_overlap(data_dict):
    texts = data_dict['text']
    overlaps = []
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            if is_overlapping(texts[i]['polygon'], texts[j]['polygon']):
                overlaps.append((texts[i]['id'], texts[j]['id']))
    return len(overlaps) > 0


def extract_titles(data):
    # Initial placeholders
    titles = {
        'x_title': None,
        'y_title': None,
        'graph_title': None
    }

    # Extract roles and texts for 'axis_title' and 'chart_title'
    axis_titles = []
    chart_title = None
    if data == "#VALUE!":
        return titles

    for item in safe_literal_eval(data):
        if item['role'] == 'axis_title':
            axis_titles.append((item['polygon']['x0'], item['text']))
        elif item['role'] == 'chart_title':
            chart_title = (item['polygon']['x0'], item['text'])

    # If there's only one 'axis_title'
    if len(axis_titles) == 1:
        if not chart_title:  # If there's no 'chart_title'
            titles['x_title'] = axis_titles[0][1]
        else:
            # If 'axis_title' x0 is lower than 'chart_title' x0, it's a y_title
            if axis_titles[0][0] < chart_title[0]:
                titles['y_title'] = axis_titles[0][1]
            else:
                titles['x_title'] = axis_titles[0][1]
    # If there are two 'axis_titles'
    elif len(axis_titles) == 2:
        # Sort by x0 value
        axis_titles = sorted(axis_titles, key=lambda x: x[0])
        titles['y_title'] = axis_titles[0][1]
        titles['x_title'] = axis_titles[1][1]

    # Add chart title if it exists
    if chart_title:
        titles['graph_title'] = chart_title[1]

    return titles


def determine_data_type(x_values):
    try:
        [float(x) for x in x_values]  # Check if we can convert all x values to float
        return 'numeric'
    except:
        return 'categorical'


# Function to determine if the data is categorical or numerical based on x and y values
def determine_overall_data_type(x_values, y_values):
    x_type = determine_data_type(x_values)
    y_type = determine_data_type(y_values)

    if x_type == 'categorical' or y_type == 'categorical':
        return 'categorical'
    else:
        return 'numerical'
