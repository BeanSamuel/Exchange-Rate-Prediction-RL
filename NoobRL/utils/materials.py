import numpy as np
from pxr import Sdf
from omni.isaac.core.materials import omni_pbr


class OmniPBR(omni_pbr.OmniPBR):
    def __init__(self, name, prim_path=None, color: list = None, opacity=None, reflection=None):
        if prim_path is None:
            prim_path = '/World/Looks/' + name
        super().__init__(prim_path, name, color=color)
        if reflection is not None:
            self.set_reflection_roughness(1 - reflection)
        if opacity is not None:
            self.set_opacity(opacity)

    def set_opacity(self, value: float):
        enable_opacity = value < 1
        if self.shaders_list[0].GetInput("enable_opacity").Get() is None:
            self.shaders_list[0].CreateInput("enable_opacity", Sdf.ValueTypeNames.Bool).Set(enable_opacity)
        else:
            self.shaders_list[0].GetInput("enable_opacity").Set(enable_opacity)

        if self.shaders_list[0].GetInput("opacity_constant").Get() is None:
            self.shaders_list[0].CreateInput("opacity_constant", Sdf.ValueTypeNames.Float).Set(value)
        else:
            self.shaders_list[0].GetInput("opacity_constant").Set(value)

    def set_color(self, color) -> None:
        super().set_color(np.array(color))
