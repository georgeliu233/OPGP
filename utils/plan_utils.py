import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math
import bisect

import matplotlib.pyplot as plt


def ref_line_grids(ref_line, widths=10, pixels_per_meter=3.2, left=True):
    '''
    generate the mapping of Cartisan coordinates (x, y)
    according to the Frenet grids (s, d)
    inputs: ref_lines (b, length, 2), width
    outputs: refline_grids (b, length, width*2, 2)
    '''
    width_d = (torch.arange(-widths, widths) + 0.5) / pixels_per_meter
    # print(ref_line.shape)
    b, l, c = ref_line.shape
    width_d = width_d.unsqueeze(0).unsqueeze(1).expand(b, l, -1).to(ref_line.device)
    angle = ref_line[:, :, 2]
    angle = (angle  + np.pi) % (2*np.pi) - np.pi #- 

    ref_x = ref_line[:, :, 0:1]
    ref_y = ref_line[:, :, 1:2]

    # output coords always conincide with ogm's coords settings
    x = -torch.sin(angle).unsqueeze(-1) * width_d + ref_x
    y = torch.cos(angle).unsqueeze(-1) * width_d + ref_y

    cart_grids = torch.stack([x, y], dim=-1)
    return cart_grids

def ref_line_ogm_sample(ogm, rl_grids, config):
    """
    scatter the ogm fields to ref_line fields
    according to the ref_line_grids
    inputs: ogm [b, h, w]
    grids: [b, l_s, l_d, 2] 
    outputs: ref_line fields: [b, l_s, l_d]
    """
    points_x, points_y = rl_grids[..., 0], rl_grids[..., 1]
    pixels_per_meter = config.pixels_per_meter
    points_x = torch.round(-points_y * pixels_per_meter) + config.sdc_x_in_grid
    points_y = torch.round(-points_x * pixels_per_meter) + config.sdc_y_in_grid

    # Filter out points that are located outside the FOV of topdown map.
    point_is_in_fov = torch.logical_and(
        torch.logical_and(
            torch.greater_equal(points_x, 0), torch.greater_equal(points_y, 0)),
        torch.logical_and(
            torch.less(points_x, config.grid_width_cells),
            torch.less(points_y, config.grid_height_cells))).float()
    
    w_axis_in = points_x * point_is_in_fov
    h_axis_in = points_y * point_is_in_fov

    w_axis_in = w_axis_in.long()
    h_axis_in = h_axis_in.long()
    
    b, h, w = w_axis_in.shape
    B = torch.arange(b).long()
    refline_fields = ogm[B[:, None], h_axis_in.view(b, -1), w_axis_in.view(b, -1)]
    refline_fields = refline_fields.view(b, h, w)

    # mask refline_fields not in fovs:
    refline_fields = refline_fields * point_is_in_fov
    return refline_fields

def generate_ego_pos_at_field(ego_pos, ref_lines, angle):
    '''
    transfrom the ego occupancy into Frenet for refline_field:
    inputs: ego_pos: [B, 2] (x, y) angle: ego angles
    ref_lines : [B, L, 3] (a, y, angle)
    outputs 3 quantile points (s, d, theta) and safe-distance ||1/4 h, w||_2
    '''
    # 1. Transform ego pos (x, y, angle) into Frenet Coords (s, l, the):
    dist = torch.norm(ego_pos.unsqueeze(1) - ref_lines[..., :2], dim=-1, p=2.0)
    s = torch.argmin(dist, dim=1)
    b = ref_lines.shape[0]
    B = torch.arange(b).long()
    sel_ref = ref_lines[B, s, :]
    s = (s - 200) *0.1

    x_r, y_r, theta_r = sel_ref[:, 0], sel_ref[:, 1], sel_ref[:, 2]
    x, y = ego_pos[:, 0], ego_pos[:, 1]

    sgn = (y - y_r) * torch.cos(theta_r) - (x - x_r) * torch.sin(theta_r)
    dis = torch.sqrt(torch.square(x - x_r) + torch.square(y - y_r))
    l = torch.sign(sgn) * dis
    
    the = angle - theta_r
    # the += np.pi/2
    the = (the  + np.pi) % (2* np.pi) - np.pi

    ego = torch.stack([s, l, the], dim=-1)

    return ego


def refline_meshgrids(ref_line_field, pixels_per_meter=3.2):
    '''
    build the (s,l) meshgrids for ref_line field
    '''
    device = ref_line_field.device
    b, s, l, _ = ref_line_field.shape
    widths =  int(l/2)
    mesh_l = (torch.arange(-widths, widths) + 0.5) / pixels_per_meter
    mesh_s = (torch.arange(s).float() + 0.5) * 0.1 #/ pixels_per_meter
    mesh_s, mesh_l = torch.meshgrid(mesh_s, mesh_l)
    mesh_sl = torch.stack([mesh_s, mesh_l], dim=-1)
    mesh_sl = mesh_sl.unsqueeze(0).expand(b, -1, -1, -1)
    mesh_sl = mesh_sl.to(device)
    return mesh_sl

    
def gather_nd_slow(ogm, h_axis_in, w_axis_in):
    b, h, w = w_axis_in.shape
    output = torch.zeros((b, h, w)).to(w_axis_in.device)
    for i in range(b):
        for j in range(h):
            for k in range(w):
                output[i, j, k] = ogm[i, h_axis_in[i, j, k], w_axis_in[i, j, k]]
    return output



class Spline(object):
    """
    Cubic Spline class
    """

    def __init__(self, x, y):
        self.b, self.c, self.d, self.w = [], [], [], []

        self.x = x
        self.y = y

        self.nx = len(x)  # dimension of x
        h = np.diff(x) + 1e-3

        # calc coefficient c
        self.a = [iy for iy in y]

        # calc coefficient c
        A = self.__calc_A(h)
        B = self.__calc_B(h)
        self.c = np.linalg.solve(A, B)
        #  print(self.c1)

        # calc spline coefficient b and d
        for i in range(self.nx - 1):
            self.d.append((self.c[i + 1] - self.c[i]) / (3.0 * h[i]))
            tb = (self.a[i + 1] - self.a[i]) / h[i] - h[i] * (self.c[i + 1] + 2.0 * self.c[i]) / 3.0
            self.b.append(tb)

    def calc(self, t):
        """
        Calc position
        if t is outside of the input x, return None
        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = self.a[i] + self.b[i] * dx + self.c[i] * dx ** 2.0 + self.d[i] * dx ** 3.0

        return result

    def calcd(self, t):
        """
        Calc first derivative
        if t is outside of the input x, return None
        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx ** 2.0

        return result

    def calcdd(self, t):
        """
        Calc second derivative
        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = 2.0 * self.c[i] + 6.0 * self.d[i] * dx
        
        return result

    def __search_index(self, x):
        """
        search data segment index
        """
        return bisect.bisect(self.x, x) - 1
    
    def search_index(self, x):
        """
        search data segment index
        """
        return bisect.bisect(self.x, x) - 1

    def __calc_A(self, h):
        """
        calc matrix A for spline coefficient c
        """
        A = np.zeros((self.nx, self.nx))
        A[0, 0] = 1.0

        for i in range(self.nx - 1):
            if i != (self.nx - 2):
                A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            A[i + 1, i] = h[i]
            A[i, i + 1] = h[i]

        A[0, 1] = 0.0
        A[self.nx - 1, self.nx - 2] = 0.0
        A[self.nx - 1, self.nx - 1] = 1.0
    
        return A

    def __calc_B(self, h):
        """
        calc matrix B for spline coefficient c
        """
        B = np.zeros(self.nx)

        for i in range(self.nx - 2):
            B[i + 1] = 3.0 * (self.a[i + 2] - self.a[i + 1]) / h[i + 1] - 3.0 * (self.a[i + 1] - self.a[i]) / h[i]

        return B


class Spline2D:
    """
    2D Cubic Spline class
    """

    def __init__(self, x, y):
        self.s = self.__calc_s(x, y)
        self.sx = Spline(self.s, x)
        self.sy = Spline(self.s, y)

    def __calc_s(self, x, y):
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = np.hypot(dx, dy)
        s = [0]
        s.extend(np.cumsum(self.ds))

        return s

    def calc_position(self, s):
        """
        calc position
        """
        x = self.sx.calc(s)
        y = self.sy.calc(s)

        return x, y

    def calc_curvature(self, s):
        """
        calc curvature
        """
        dx = self.sx.calcd(s)
        ddx = self.sx.calcdd(s)
        dy = self.sy.calcd(s)
        ddy = self.sy.calcdd(s)
        k = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2)**(3 / 2))

        return k

    def calc_yaw(self, s):
        """
        calc yaw
        """
        dx = self.sx.calcd(s)
        dy = self.sy.calcd(s)
        yaw = math.atan2(dy, dx)

        return yaw
    
    def search_index(self,s):
        i = self.sx.search_index(s)
        j = self.sy.search_index(s)
        return i,j

def generate_target_course(x, y):
    csp = Spline2D(x, y)
    s = np.arange(0, csp.s[-1], 0.1)
    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = csp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(csp.calc_yaw(i_s))
        rk.append(csp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, csp


