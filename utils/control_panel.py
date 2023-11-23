import omni.ui as ui


def _preproc_kwargs(kwargs):
    for k in kwargs.keys():
        if k in ['width', 'height']:
            kwargs[k] = ui.Length(kwargs[k])
    return kwargs


class ControlPanel:
    def __init__(self, name):
        self._window = ui.Window(name, auto_resize=True)
        self._components = dict()

    def __getitem__(self, name):
        if isinstance(name, (list, tuple)):
            return [self.__getitem__(x) for x in name]

        item = self._components.get(name)
        if isinstance(item, ui.FloatSlider):
            return item.model.get_value_as_float()
        elif isinstance(item, ui.CheckBox):
            return item.model.get_value_as_bool()
        else:
            raise IndexError

    def __setitem__(self, key, value):
        if isinstance(key, (list, tuple)):
            for k, v in zip(key, value):
                self.__setitem__(k, v)
            return

        item = self._components.get(key)
        if isinstance(item, ui.FloatField):
            item.model.set_value(value)
        else:
            raise IndexError

    def add_slider(self, name, **kwargs):
        self._components[name] = lambda: ui.FloatSlider(**_preproc_kwargs(kwargs))

    def add_float(self, name, **kwargs):
        self._components[name] = lambda: ui.FloatField(**_preproc_kwargs(kwargs))

    def add_check_box(self, name, **kwargs):
        self._components[name] = lambda: ui.CheckBox(**_preproc_kwargs(kwargs))

    def build(self):
        with self._window.frame:
            with ui.VStack(width=150):
                for k, v in self._components.items():
                    ui.Label(k)
                    self._components[k] = v()
