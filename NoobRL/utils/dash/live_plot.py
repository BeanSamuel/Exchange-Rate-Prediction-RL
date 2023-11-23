import os
import socket
import logging
import threading
import numpy as np
import time
import torch
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
from dash import Dash, html, dcc, Output, Input
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
pio.renderers.default = "browser"


callback_dict = dict()


def callback(*args, **kwargs):
    def wrapped(func):
        global callback_dict
        callback_dict[func.__name__] = (args, kwargs)
        return func

    return wrapped


class LivePlot:
    def __init__(self, name, titles, steps):
        self.name = name
        self.titles = titles
        self.dim_names = list(titles.keys())
        self.dim_labels = list(titles.values())
        self.num_dims = len(self.dim_names)

        for i, labels in enumerate(self.dim_labels):
            if isinstance(labels, list):
                self.dim_labels[i] = ['All'] + labels
            if isinstance(labels, int):
                self.dim_labels[i] = ['All'] + list(map(str, range(labels)))

        self.steps = 0
        self.size = steps
        self.time_axis = np.arange(steps)
        self.free_dim = -1
        self.datas = np.full([steps] + [len(x) - 1 for x in self.dim_labels], np.nan)

        self._build_app()
        self._create_thread()

    def _build_app(self):
        dropdowns = []
        for name, labels in zip(self.dim_names, self.dim_labels):
            dropdowns.append(name)
            options = {str(i): label for i, label in enumerate(labels)}
            dropdowns.append(dcc.Dropdown(id=name, options=options, value='0'))

        app = Dash(__name__)
        app.layout = html.Div([
            html.H1(children=self.name, style={'textAlign': 'center'}),
            html.Div(dropdowns),
            html.Div([
                dcc.Graph(id='live-graph'),
                dcc.Interval(
                    id='interval-component',
                    interval=16,
                    n_intervals=0
                )
            ])
        ])

        for func_name, (args, kwargs) in callback_dict.items():
            func = getattr(self, func_name)
            app.callback(*args, **kwargs)(func)

        app.callback(
            [Output(i, 'value') for i in self.dim_names],
            [Input(i, 'value') for i in self.dim_names]
        )(self._update_figure)

        self._update_figure(*(['1'] * self.num_dims))

        self._app = app

    def _create_thread(self):
        port = 8050
        while True:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('localhost', port)):
                    break
                else:
                    port += 1

        run_server = lambda: self._app.run(host='0.0.0.0', port=port)
        thread = threading.Thread(target=run_server)
        thread.daemon = True
        thread.start()
        time.sleep(0.1)

        print('live plot:', self.name, f'http://localhost:{port}')

        self._thread = thread

    def _update_figure(self, *values, save_path=None):
        values = [str(v) for v in values]
        idx = [slice(None)]
        titles = [' ']

        # print('free dim', self.free_dim)

        free_dim = -1
        for i, v in enumerate(values):
            if v == '0':
                if free_dim == -1:
                    free_dim = i
                else:
                    values[i] = '1'

        if free_dim != self.free_dim and self.free_dim != -1:
            values[self.free_dim] = '1'

        self.free_dim = free_dim

        for i in range(self.num_dims):
            if values[i] == '0':
                titles = self.dim_labels[i][1:]
                idx.append(slice(None))
            else:
                idx.append(int(values[i]) - 1)

        self.idx = tuple(idx)
        # print(self.idx)
        # print(titles)

        self._updating = True
        self.fig = go.FigureWidget(make_subplots(rows=len(titles), cols=1, subplot_titles=titles))
        for i, data in enumerate(self._get_plot_data()):
            self.fig.add_trace(go.Scatter(name='', x=self.time_axis, y=data), row=i+1, col=1)
        self.fig.update_layout(height=200*len(titles)+100, template='plotly')
        self._updating = False

        if save_path is not None:
            self.fig.write_html(save_path)

        # print(values)
        return values

    def _get_plot_data(self):
        datas = self.datas[self.idx]
        return np.expand_dims(datas, 0) if datas.ndim == 1 else np.swapaxes(datas, 0, 1)

    def update(self, datas):
        if isinstance(datas, torch.Tensor):
            datas = datas.detach().cpu().numpy()
        if self.steps >= self.size:
            self.time_axis += 1
            self.datas[:-1] = self.datas[1:]
            self.datas[-1] = datas
        else:
            self.datas[self.steps] = datas

        self.steps += 1
        while self._updating:
            time.sleep(0.01)
        for i, data in enumerate(self._get_plot_data()):
            self.fig.data[i]['x'] = self.time_axis
            self.fig.data[i]['y'] = data

    @callback(
        Output('live-graph', 'figure'),
        Input('interval-component', 'n_intervals')
    )
    def _update_graph(self, n):
        return self.fig

    def select_labels(self, *labels):
        # ToDo update selector label
        self._update_figure(*labels)

    def snapshot(self, dir_path, free_dim=0):

        def export(labels, names):
            dim = len(labels)
            if dim == self.num_dims:
                name = self.name + ': ' + ' '.join(names) if names else self.name
                save_path = os.path.join(dir_path, name) + '.html'
                self._update_figure(*labels, save_path=save_path)
            else:
                if dim == free_dim:
                    export(labels + [0], names)
                else:
                    for i, s in enumerate(self.dim_labels[dim][1:]):
                        export(labels + [i+1], names + [s])

        export([], [])

    def save(self, dir_path):
        state = self.__dict__.copy()
        state.pop('_app')
        state.pop('_thread')
        torch.save(state, os.path.join(dir_path, self.name + '.liveplot'))

    @staticmethod
    def load(path):
        plot = LivePlot.__new__(LivePlot)
        plot.__dict__ = torch.load(path)
        plot._build_app()
        plot._create_thread()
        return plot


if __name__ == '__main__':
    plot = LivePlot('1', {'1': ['a', 'b'], '2': 5}, 30)
    plot2 = LivePlot('2', {'1': ['a', 'b'], '2': 5}, 30)
    import time
    for i in range(10):
        plot.update(np.random.random([2, 5]))
        plot2.update(np.random.random([2, 5]))
        time.sleep(0.1)
