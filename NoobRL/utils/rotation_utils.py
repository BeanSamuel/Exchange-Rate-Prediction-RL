import sys
from torch.autograd import Variable
import torch.distributed.algorithms

sys.path.append('/home/hardy/.local/share/ov/pkg/isaac_sim-2022.2.1/exts/omni.isaac.core')

import numpy as np
from numpy import pi, sin, cos
import plotly.express as px
import plotly.io as pio
from utils.torch_utils import *
pio.renderers.default = "browser"


# auto-shaping
def ash(func, x, in_size):
    shape = x.shape[:-1]
    return func(x.view(shape + (-1, in_size))).view(shape + (-1,))


@torch.jit.script
def normalize(x, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


@torch.jit.script
def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


def rad2deg(radian_value, device=None):
    return torch.rad2deg(radian_value).float().to(device)


def deg2rad(degree_value, device=None):
    return torch.deg2rad(degree_value).float().to(device)


def zero_pos(shape, device=None):
    return torch.zeros(to_torch_size(shape) + (3,), device=device)


def zero_pos_like(x):
    return zero_pos(x.shape[:-1], x.device)


def full_pos(shape, value, device=None):
    x = torch.zeros(to_torch_size(shape) + (3,), device=device)
    x[:] = torch.tensor(value, device=device)
    return x


def full_pos_like(x, value):
    return full_pos(x.shape[:-1], value, x.device)


def identity_quat(shape, device=None):
    q = torch.zeros(to_torch_size(shape) + (4,), device=device)
    q[..., 0] = 1
    return q


def identity_quat_like(x):
    return identity_quat(x.shape[:-1], x.device)


@torch.jit.script
def quat_unit(a):
    return normalize(a)


# @torch.jit.script
# def quat_mul_unnorm(a, b):
#     shape = a.shape
#     a = a.reshape(-1, 4)
#     b = b.reshape(-1, 4)
#
#     w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
#     w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
#     ww = (z1 + x1) * (x2 + y2)
#     yy = (w1 - y1) * (w2 + z2)
#     zz = (w1 + y1) * (w2 - z2)
#     xx = ww + yy + zz
#     qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
#     w = qq - ww + (z1 - y1) * (y2 - z2)
#     x = qq - xx + (x1 + w1) * (x2 + w2)
#     y = qq - yy + (w1 - x1) * (y2 + z2)
#     z = qq - zz + (z1 + y1) * (w2 - x2)
#     quat = torch.stack([w, x, y, z], dim=-1).view(shape)
#
#     return quat


# @torch.jit.script
# def quat_inverse(a):
#     shape = a.shape
#     a = a.reshape(-1, 4)
#     return torch.cat((a[..., 0:1], -a[..., 1:]), dim=-1).view(shape)


@torch.jit.script
def quat_mul_unnorm(a, b):
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    quat = torch.stack([w, x, y, z], dim=-1)

    return quat


@torch.jit.script
def quat_inverse(a):
    a = a.clone()
    a[..., 1:] *= -1
    return a


@torch.jit.script
def quat_rotate(q, v):
    q_w = q[..., 0:1]
    q_vec = q[..., 1:]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
    c = q_vec * torch.sum(q_vec * v, dim=-1, keepdim=True) * 2.0
    return a + b + c


@torch.jit.script
def quat_rotate_inverse(q, v):
    q_w = q[..., 0].unsqueeze(-1)
    q_vec = q[..., 1:]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
    c = q_vec * torch.sum(q_vec * v, dim=-1, keepdim=True) * 2.0
    return a - b + c


@torch.jit.script
def quat_mul(q0, q1):
    return quat_unit(quat_mul_unnorm(q0, q1))


@torch.jit.script
def quat_div(x, y):
    return quat_mul(quat_inverse(y), x)


@torch.jit.script
def quat_diff_rad(a, b):
    eps = 1e-5
    b_conj = quat_inverse(b)
    mul = quat_mul_unnorm(a, b_conj)
    # 2 * torch.acos(torch.abs(mul[..., -1]))
    return 2.0 * torch.asin(torch.clamp(torch.norm(mul[..., 1:], p=2, dim=-1), max=1-eps, min=eps-1))


@torch.jit.script
def quat_to_angle_axis(q):
    # computes axis-angle representation from quaternion q
    # q must be normalized
    min_theta = 1e-5
    qw, qx, qy, qz = 0, 1, 2, 3

    sin_theta = torch.sqrt(1 - q[..., qw] * q[..., qw])
    angle = 2 * torch.acos(q[..., qw])
    angle = normalize_angle(angle)
    sin_theta_expand = sin_theta.unsqueeze(-1)
    axis = q[..., qx:] / sin_theta_expand

    mask = sin_theta > min_theta
    default_axis = torch.zeros_like(axis)
    default_axis[..., qw] = 1

    angle = torch.where(mask, angle, torch.zeros_like(angle))
    mask_expand = mask.unsqueeze(-1)
    axis = torch.where(mask_expand, axis, default_axis)
    return angle, axis


@torch.jit.script
def quat_from_angle_axis(angle, axis):
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    return quat_unit(torch.cat([w, xyz], dim=-1))


@torch.jit.script
def angle_axis_to_exp_map(angle, axis):
    # compute exponential map from axis-angle
    angle_expand = angle.unsqueeze(-1)
    exp_map = angle_expand * axis
    return exp_map


@torch.jit.script
def quat_to_exp_map(q):
    eps = 1e-5
    qw = q[..., 0, None].clamp(-1+eps, 1-eps)
    q_axis = q[..., 1:]
    angle = normalize_angle(2 * qw.acos())
    axis = q_axis / torch.sqrt(1 - qw ** 2)
    return angle * axis


# @torch.jit.script
# def quat_to_exp_map(q):
#     # compute exponential map from quaternion
#     # q must be normalized
#     angle, axis = quat_to_angle_axis(q)
#     exp_map = angle_axis_to_exp_map(angle, axis)
#     return exp_map


# @torch.jit.script
# def exp_map_to_angle_axis(exp_map):
#     min_theta = 1e-5
#
#     angle = torch.norm(exp_map, dim=-1)
#     angle_exp = torch.unsqueeze(angle, dim=-1)
#     axis = exp_map / angle_exp
#     angle = normalize_angle(angle)
#
#     default_axis = torch.zeros_like(exp_map)
#     default_axis[..., -1] = 1
#
#     mask = angle > min_theta
#     angle = torch.where(mask, angle, torch.zeros_like(angle))
#     mask_expand = mask.unsqueeze(-1)
#     axis = torch.where(mask_expand, axis, default_axis)
#
#     return angle, axis


# @torch.jit.script
# def exp_map_to_quat(exp_map):
#     angle, axis = exp_map_to_angle_axis(exp_map)
#     q = quat_from_angle_axis(angle, axis)
#     return q


@torch.jit.script
def exp_map_to_quat(exp_map):
    eps = 1e-12
    angle = torch.norm(exp_map, dim=-1, keepdim=True)
    axis = exp_map / (angle + eps)

    theta = normalize_angle(angle) / 2
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    return quat_unit(torch.cat([w, xyz], dim=-1))


@torch.jit.script
def quat_to_tan_norm(q):
    # represents a rotation using the tangent and normal vectors
    ref_tan = torch.zeros_like(q[..., 0:3])
    ref_tan[..., 0] = 1
    tan = quat_rotate(q, ref_tan)

    ref_norm = torch.zeros_like(q[..., 0:3])
    ref_norm[..., -1] = 1
    norm = quat_rotate(q, ref_norm)

    norm_tan = torch.cat([tan, norm], dim=len(tan.shape) - 1)
    return norm_tan


@torch.jit.script
def quat_from_rotation_matrix(m):
    m = m.unsqueeze(0)
    diag0 = m[..., 0, 0]
    diag1 = m[..., 1, 1]
    diag2 = m[..., 2, 2]

    # Math stuff.
    w = (((diag0 + diag1 + diag2 + 1.0) / 4.0).clamp(0.0, None)) ** 0.5
    x = (((diag0 - diag1 - diag2 + 1.0) / 4.0).clamp(0.0, None)) ** 0.5
    y = (((-diag0 + diag1 - diag2 + 1.0) / 4.0).clamp(0.0, None)) ** 0.5
    z = (((-diag0 - diag1 + diag2 + 1.0) / 4.0).clamp(0.0, None)) ** 0.5

    # Only modify quaternions where w > x, y, z.
    c0 = (w >= x) & (w >= y) & (w >= z)
    x[c0] *= (m[..., 2, 1][c0] - m[..., 1, 2][c0]).sign()
    y[c0] *= (m[..., 0, 2][c0] - m[..., 2, 0][c0]).sign()
    z[c0] *= (m[..., 1, 0][c0] - m[..., 0, 1][c0]).sign()

    # Only modify quaternions where x > w, y, z
    c1 = (x >= w) & (x >= y) & (x >= z)
    w[c1] *= (m[..., 2, 1][c1] - m[..., 1, 2][c1]).sign()
    y[c1] *= (m[..., 1, 0][c1] + m[..., 0, 1][c1]).sign()
    z[c1] *= (m[..., 0, 2][c1] + m[..., 2, 0][c1]).sign()

    # Only modify quaternions where y > w, x, z.
    c2 = (y >= w) & (y >= x) & (y >= z)
    w[c2] *= (m[..., 0, 2][c2] - m[..., 2, 0][c2]).sign()
    x[c2] *= (m[..., 1, 0][c2] + m[..., 0, 1][c2]).sign()
    z[c2] *= (m[..., 2, 1][c2] + m[..., 1, 2][c2]).sign()

    # Only modify quaternions where z > w, x, y.
    c3 = (z >= w) & (z >= x) & (z >= y)
    w[c3] *= (m[..., 1, 0][c3] - m[..., 0, 1][c3]).sign()
    x[c3] *= (m[..., 2, 0][c3] + m[..., 0, 2][c3]).sign()
    y[c3] *= (m[..., 2, 1][c3] + m[..., 1, 2][c3]).sign()

    return quat_unit(torch.stack([w, x, y, z], dim=-1)).squeeze(0)


@torch.jit.script
def quat_from_dir(v):
    u = torch.zeros_like(v)
    u[..., 2] = 1
    xyz = torch.cross(u, v, dim=-1)
    w = torch.sqrt((u ** 2).sum(-1) * (v ** 2).sum(-1)) + (u * v).sum(-1)
    q = quat_unit(torch.cat([w.unsqueeze(-1), xyz], dim=-1))
    q[q.abs().sum(-1) < 1e-6, [1]] = 1
    return q


@torch.jit.script
def quat_to_tan_norm(q):
    ref_tan = torch.zeros_like(q[..., 0:3])
    ref_tan[..., 0] = 1
    tan = quat_rotate(q, ref_tan)

    ref_norm = torch.zeros_like(q[..., 0:3])
    ref_norm[..., -1] = 1
    norm = quat_rotate(q, ref_norm)

    norm_tan = torch.cat([tan, norm], dim=len(tan.shape) - 1)
    return norm_tan


@torch.jit.script
def exp_map_mul(e0, e1):
    shape = e0.shape[:-1] + (-1,)
    q0 = exp_map_to_quat(e0.reshape(-1, 3))
    q1 = exp_map_to_quat(e1.reshape(-1, 3))
    return quat_to_exp_map(quat_mul(q0, q1)).view(shape)


@torch.jit.script
def exp_map_div(e0, e1):
    shape = e0.shape[:-1] + (-1,)
    q0 = exp_map_to_quat(e0.reshape(-1, 3))
    q1 = exp_map_to_quat(e1.reshape(-1, 3))
    return quat_to_exp_map(quat_div(q0, q1)).view(shape)


@torch.jit.script
def exp_map_diff_rad(e0, e1):
    return quat_diff_rad(exp_map_to_quat(e0), exp_map_to_quat(e1))


@torch.jit.script
def lerp(p0, p1, t):
    return (1 - t) * p0 + t * p1


# @torch.jit.script
def slerp(q0, q1, t):
    qw, qx, qy, qz = 0, 1, 2, 3

    cos_half_theta = q0[..., qw] * q1[..., qw] \
                     + q0[..., qx] * q1[..., qx] \
                     + q0[..., qy] * q1[..., qy] \
                     + q0[..., qz] * q1[..., qz]

    neg_mask = cos_half_theta < 0
    q1 = q1.clone()
    q1[neg_mask] = -q1[neg_mask]
    cos_half_theta = torch.abs(cos_half_theta)
    cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

    half_theta = torch.acos(cos_half_theta)
    sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)

    ratioA = torch.sin((1 - t) * half_theta) / sin_half_theta
    ratioB = torch.sin(t * half_theta) / sin_half_theta

    new_q_w = ratioA * q0[..., qw:qw + 1] + ratioB * q1[..., qw:qw + 1]
    new_q_x = ratioA * q0[..., qx:qx + 1] + ratioB * q1[..., qx:qx + 1]
    new_q_y = ratioA * q0[..., qy:qy + 1] + ratioB * q1[..., qy:qy + 1]
    new_q_z = ratioA * q0[..., qz:qz + 1] + ratioB * q1[..., qz:qz + 1]

    cat_dim = len(new_q_w.shape) - 1
    new_q = torch.cat([new_q_w, new_q_x, new_q_y, new_q_z], dim=cat_dim)

    new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
    new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)

    return new_q


@torch.jit.script
def calc_heading(q):
    # calculate heading direction from quaternion
    # the heading is the direction on the xy plane
    # q must be normalized
    ref_dir = torch.zeros_like(q[..., 0:3])
    ref_dir[..., 0] = 1
    rot_dir = quat_rotate(q, ref_dir)

    heading = torch.atan2(rot_dir[..., 1], rot_dir[..., 0])
    return heading

@torch.jit.script
def calc_heading_quat(q):
    # calculate heading rotation from quaternion
    # the heading is the direction on the xy plane
    # q must be normalized
    heading = calc_heading(q)
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 2] = 1

    heading_q = quat_from_angle_axis(heading, axis)
    return heading_q


@torch.jit.script
def calc_heading_quat_inv(q):
    # calculate heading rotation from quaternion
    # the heading is the direction on the xy plane
    # q must be normalized
    heading = calc_heading(q)
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 2] = 1

    heading_q = quat_from_angle_axis(-heading, axis)
    return heading_q


@torch.jit.script
def normalize_pos(pos):
    z = torch.zeros_like(pos)
    z[..., 2] = 1
    return z * pos.norm(p=2, dim=-1, keepdim=True)


def draw_exp_map(e):
    draw_quaternion(exp_map_to_quat(e))


def draw_quaternion(q):
    v = torch.Tensor([0, 0, 1]).repeat(len(q), 1)
    v = quat_rotate(q, v)
    fig = px.scatter_3d(x=v[:, 0], y=v[:, 1], z=v[:, 2])
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-1, 1]),
            yaxis=dict(range=[-1, 1]),
            zaxis=dict(range=[-1, 1]),
        )
    )
    fig.update_scenes(aspectmode='cube')
    fig_add_sphere(fig)
    fig.show()


def random_quaternion(size):
    return exp_map_to_quat((torch.rand([size, 3]) - 0.5) * 2 * torch.pi)


def fig_add_sphere(fig):
    theta = np.linspace(0, 2 * pi, 120)
    phi = np.linspace(0, pi, 60)
    u, v = np.meshgrid(theta, phi)
    xs = cos(u) * sin(v)
    ys = sin(u) * sin(v)
    zs = cos(v)

    x, y, z = [], [], []
    for t in [theta[10 * k] for k in range(12)]:  # meridians:
        x.extend(list(cos(t) * sin(phi)) + [None])  # None is inserted to mark the end of a meridian line
        y.extend(list(sin(t) * sin(phi)) + [None])
        z.extend(list(cos(phi)) + [None])

    for s in [phi[6 * k] for k in range(10)]:  # parallels
        x.extend(list(cos(theta) * sin(s)) + [None])  # None is inserted to mark the end of a parallel line
        y.extend(list(sin(theta) * sin(s)) + [None])
        z.extend([cos(s)] * 120 + [None])

    fig.add_surface(x=xs, y=ys, z=zs,
                    colorscale=[[0, '#ffffff'], [1, '#ffffff']],
                    showscale=False, opacity=0.5)  # or opacity=1
    fig.add_scatter3d(x=x, y=y, z=z, mode='lines', line_width=3, line_color='rgb(10,10,10)')


def _test_exp_map_diff_rad_grad():
    n = 10000
    print('testing...')
    for _ in range(1000):
        x = Variable(torch.rand([n, 3]) * 1000, requires_grad=True)
        y = exp_map_diff_rad(x, torch.rand([n, 3])).mean()
        y.backward()
        if x.grad.isnan().any():
            print(y)
    print('finish')


def _test_exp_map_to_quat_grad():
    n = 10000
    print('testing...')
    for _ in range(1):
        x = Variable(torch.rand([n, 3]) * 1000, requires_grad=True)
        y = exp_map_to_quat(x).mean()
        y.backward()
        print(x.grad)
        # if x.grad.isnan().any():
        #     print(y)
    print('finish')


def _test_quat_to_exp_map_grad():
    n = 10000
    print('testing...')
    for _ in range(1):
        x = Variable(torch.rand([n, 3]), requires_grad=True)
        y = exp_map_to_quat(x)
        y = quat_to_exp_map(y)
        y.mean().backward()
        print((y - x).sum())
        print(x.grad)
        # if x.grad.isnan().any():
        #     print(y)
    print('finish')


def _test_slerp():
    n = 15
    q0 = random_quaternion(1).repeat(n, 1)
    q1 = random_quaternion(1).repeat(n, 1)
    t = torch.arange(n).float() / n
    q = slerp(q0, q1, t.unsqueeze(-1))
    draw_quaternion(q)


if __name__ == '__main__':
    _test_quat_to_exp_map_grad()

