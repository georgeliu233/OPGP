import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append('/theseus')

import theseus as th
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import numpy as np

from utils.plan_utils import *
import casadi as cs
import math
import scipy

class Planner(object):
    def __init__(self,
        device, 
        horizon=4,
        g_length=50,
        g_width=40,
        test_iters=50,
        test_step=0.3,
        ):
        super(Planner, self).__init__()

        self.g_width = g_width
        self.horizon = horizon
        control_variables = th.Vector(dof=horizon *10 * 2, name="control_variables")
        ref_line_fields = th.Variable(torch.empty(1, horizon, g_length, g_width, 2), name="ref_line_field")
        ref_line_costs = th.Variable(torch.empty(1, horizon, g_length, g_width), name="ref_line_costs")
        lwt = th.Variable(torch.empty(1, horizon*10, 3), name="lwt")
        current_state = th.Variable(torch.empty(1, 2, 4), name="current_state")
        spl = th.Variable(torch.empty(1, 1), name="speed_limit")
        stp = th.Variable(torch.empty(1, 1), name="stop_point")
        il_lane = th.Variable(torch.empty(1, horizon*10, 2), name="il_lane")

        objective = th.Objective()
        objective = self.cost_function(objective, control_variables, ref_line_fields, 
                ref_line_costs, lwt, current_state, spl, stp, il_lane)
        self.optimizer = th.GaussNewton(objective, th.CholmodSparseSolver, vectorize=False,
             max_iterations=test_iters, step_size=test_step, abs_err_tolerance=1e-2)
        
        self.layer = th.TheseusLayer(self.optimizer, vectorize=False)
        self.layer.to(device=device)

        self.max_acc = 5
        self.max_delta = math.pi / 6
        self.ts = 0.1
        self.sigma = 1.0
        self.num = 0

    def preprocess(self, ego_state, ego_plan, ref_lines, ogm_prediction, type_mask, config, left=True):
        #computing the angle:
        diff_traj = torch.diff(ego_plan, axis=1)
        diff_traj = torch.cat([diff_traj, diff_traj[:, -1, :].unsqueeze(1)], dim=1)
        angle = torch.nan_to_num(torch.atan2(diff_traj[:,:,1], diff_traj[:,:,0]), 0).clamp(-0.67,0.67)

        #reshape time axis: l to the batch axis
        frenet_plan, angle = ego_plan[:, :50, :2], angle[:, :50]
        length, width = ego_state[:, -1, 6], ego_state[:, -1, 7]
        b, l, d = frenet_plan.shape
        length, width = length.unsqueeze(1).expand(-1, l), width.unsqueeze(1).expand(-1, l)
        frenet_plan = frenet_plan.reshape(b*l, d)
        speed = ego_state[:, -2:, 3]
        # print(ego_state[:, -2:])
        angle = angle.reshape(b*l)

        # ref_lines = ref_lines[:, ::2, :]
        speed_limit = ref_lines[:, :, 4]
        b, t, d = ref_lines.shape
        orf = ref_lines
        ref_lines = ref_lines.unsqueeze(1).expand(-1, l, -1, -1).reshape(b*l, t, d)
        
        #generate the ego mask
        ego = generate_ego_pos_at_field(frenet_plan, ref_lines, angle)
        spl = torch.max(speed_limit, dim=1, keepdim=True)[0]
        stp = torch.max(speed_limit==0, dim=1, keepdim=True)[1] * 0.1
        theta = ego[:, -1].reshape(b, l)

        ref_lines = ref_lines.reshape(b, l, t, d)[:, ::10].reshape(b*l//10, t, d)

        refline_fields = ref_line_grids(ref_lines, widths=int(self.g_width/2), left=left ,pixels_per_meter=3.2)
        
        ogm_prediction = ogm_prediction[:, 0] * (ogm_prediction[:, 0] > 0.3) + \
            10*ogm_prediction[:, 1] * type_mask[:, 1,None,None,None] + 
            10*ogm_prediction[:, 2] * type_mask[:, 2,None,None,None]
        ogm_prediction = ogm_prediction.clamp(0, 1)
  
        ogm_prediction = ogm_prediction * (ogm_prediction > 0.1)

        b, lp, h, w = ogm_prediction.shape
        og = ogm_prediction
        ogm_prediction = ogm_prediction.reshape(b*l//10, h, w)

        ogm_refline_fields = ref_line_ogm_sample(ogm_prediction, refline_fields, config)
        _, h, w = ogm_refline_fields.shape

        current_state = generate_ego_pos_at_field(ego_state[:, -1, [0, 1]],orf,ego_state[:, -1, 2])
        lcurrent_state = generate_ego_pos_at_field(ego_state[:, -2, [0, 1]],orf,ego_state[:, -2, 2])

        all_ego = torch.cat([current_state[:, None, :2], ego[:, :2].reshape(b, l, 2)], dim=1)
        d_ego = torch.diff(all_ego, dim=1) / 0.1

        c_state = torch.stack([lcurrent_state, current_state], dim=-2)
        c_state = torch.concat([c_state, speed.unsqueeze(-1)], dim=-1)
        il_lane = ego[:, :2].reshape(b, l, 2)
        # print(il_lane[0, 9::10])
      
        return {
            'control_variables': d_ego.reshape(b, l*2).detach(),
            'il_lane': il_lane.detach(),
            'lwt': torch.stack([length, width, theta], dim=-1).detach(),
            'ref_line_field': refline_fields.reshape(b, lp, h, w, 2).detach(),
            'ref_line_costs': ogm_refline_fields.reshape(b, lp, h, w).detach(),
            'current_state': c_state.detach(),
            'speed_limit': spl,
            'stop_point':stp,
        }
    
    def il_cost(self, optim_vars, aux_vars):
        ego = optim_vars[0].tensor.view(-1, self.horizon*10, 2) 
        current_state = aux_vars[0].tensor
        ds = ego[:, :, 0].clamp(min=0) 
        dl = ego[:, :, 1]
        s = current_state[:, -1, 0][:,None] + torch.cumsum(ds * 0.1, dim=-1)
        L = current_state[:, -1, 1][:,None] + torch.cumsum(dl * 0.1, dim=-1)
        ego = torch.stack([s, L], dim=-1)
        il_lane = aux_vars[1].tensor
        cost = torch.abs(il_lane - ego).mean(-1)
        return 1 * cost

    def collision_cost(self, optim_vars, aux_vars):
        ego = optim_vars
        lwt, refline_fields, ref_line_costs, current_state = aux_vars
        lwt, refline_fields, ref_line_costs = lwt.tensor, refline_fields.tensor, ref_line_costs.tensor
        b, l, h, w = ref_line_costs.shape
        lwt = lwt[:, 9::10,:].reshape(b*l, 3)
        ego = ego[0].tensor.view(b, l*10, 2)
        ds = ego[:, :, 0].clamp(min=0) 
        dl = ego[:, :, 1]
        s = current_state[:, -1, 0][:,None] + torch.cumsum(ds * 0.1, dim=-1)
        L = current_state[:, -1, 1][:,None] + torch.cumsum(dl * 0.1, dim=-1)
        ego = torch.stack([s, L], dim=-1)[:, 4::5, :]
        ego = ego.view(b, l, 2, 2)
        ego = ego.view(b*l, 2, 2)
        
        refline_fields = refline_fields.view(b*l, h, w, 2)
        ref_line_costs = ref_line_costs.view(b*l, h, w)

        mesh_sl = refline_meshgrids(refline_fields, pixels_per_meter=1.6)

        safety_cost_mask = ref_line_costs
        safety_cost_mask = safety_cost_mask > 0.3

        diff =  mesh_sl.unsqueeze(1) - ego.unsqueeze(-2).unsqueeze(-2)

        interactive_mask = (diff[..., 0] > 0) * (torch.abs(diff[..., 1]) < 7.5)
        ego_dist = torch.sqrt(torch.square(diff[..., 0]) + torch.square(diff[..., 1]))
        ego_dist = (5 - ego_dist) * (ego_dist < 5) * interactive_mask

        ego_s = ego_dist * safety_cost_mask.unsqueeze(1) 
        safety_cost_s = ego_s.sum(-1).sum(-1)

        safety_cost = safety_cost_s

        safety_cost = safety_cost.view(b, l*2)

        return 10 * safety_cost
    
    def red_light(self, optim_vars, aux_vars):
        ego = optim_vars
        ego =  ego[0].tensor.view(-1, self.horizon*10, 2) 
        stop_point = aux_vars[0].tensor
        current_state = aux_vars[1].tensor
        s = current_state[:, -1, 0][:,None] + torch.cumsum(ego[..., 0] * 0.1, dim=-1)
        stop_distance = stop_point - 3
        red_light_error = (s - stop_distance) * (s > stop_distance) * (stop_point != 0)
        return 10 * red_light_error

    
    def cost_function(self, objective, control_variables, ref_line_fields,
             ref_line_costs, lwt, current_state, spl, stp, il_lane, vectorize=True):
        
        safe_cost = th.AutoDiffCostFunction([control_variables], self.collision_cost, self.horizon*2, 
         aux_vars=[lwt, ref_line_fields, ref_line_costs, current_state], autograd_vectorize=vectorize, name="safe_cost")
        objective.add(safe_cost)

        il_cost = th.AutoDiffCostFunction([control_variables], self.il_cost,  self.horizon*10, 
        aux_vars=[current_state, il_lane],autograd_vectorize=vectorize, name="il_cost")
        objective.add(il_cost)


        rl_cost = th.AutoDiffCostFunction([control_variables], self.red_light, self.horizon*10, 
        aux_vars=[stp, current_state],autograd_vectorize=vectorize, name="red_light")
        objective.add(rl_cost)

        return objective
    
    
    def plan(self, planning_inputs, selected_ref, current_state):
       
        final_values, info = self.layer.forward(planning_inputs, optimizer_kwargs={'track_best_solution': True})
        plan = info.best_solution["control_variables"].view(-1, self.horizon*10, 2)
        plan = plan.to(selected_ref.device)
        s = planning_inputs['current_state'][:, -1, 0][:,None] + torch.cumsum(plan[..., 0] * 0.1, dim=-1)
        l = planning_inputs['current_state'][:, -1, 1][:,None] + torch.cumsum(plan[..., 1] * 0.1, dim=-1)
        plan = torch.stack([s, l], dim=-1)
        xy_plan = self.frenet_to_cartiesan(plan, selected_ref)
        speed = torch.hypot(current_state[:, -1, 2], current_state[:, -1, 3])
        last_speed = torch.hypot(current_state[:, -2, 2], current_state[:,-2, 3])
        acc = (speed - last_speed)/ 0.1
        current_state = torch.stack([current_state[:,-1, 0], current_state[:, -1, 1], current_state[:, -1, 4], speed, acc], dim=-1)
        b = xy_plan.shape[0]
        res = []
        for i in range(b):
            pl = self.refine(current_state[i].cpu().numpy(), xy_plan[i].cpu().numpy())
            res.append(pl)
        return torch.tensor(np.stack(res, 0)).to(xy_plan.device).float()
    
    def refine(self, current_state, reference):
        opti = cs.Opti()

        # Define the optimization variables
        X = opti.variable(4, self.horizon*10 + 1)
        U = opti.variable(2, self.horizon*10)

        # Define the initial state and the reference trajectory
        x0 = current_state[:4] # (x, y, theta, v)
        xr = reference.T

        # Define the cost function for the MPC problem
        obj = 0            

        for i in range(self.horizon*10):
            obj += (i+1) / (self.horizon*10) * cs.sumsqr(X[:2, i+1] - xr[:2, i])
            obj += 100 * (X[2, i+1] - xr[2, i]) ** 2
            obj += 0.1 * (U[0, i]) ** 2
            obj += U[1, i] ** 2
        
            if i >= 1:
                obj += 0.1 * (U[0, i] - U[0, i-1]) ** 2
                obj += (U[1, i] - U[1, i-1]) ** 2

        opti.minimize(obj)

        # Define the constraints for the MPC problem
        opti.subject_to(X[:, 0] == x0)
        opti.subject_to(U[0, 0] == current_state[4])

        for i in range(self.horizon*10):
            opti.subject_to([X[0, i+1] == X[0, i] + X[3, i] * cs.cos(X[2, i]) * self.ts,
                             X[1, i+1] == X[1, i] + X[3, i] * cs.sin(X[2, i]) * self.ts,
                             X[2, i+1] == X[2, i] + X[3, i] / 4 * cs.tan(U[1, i]) * self.ts,
                             X[3, i+1] == X[3, i] + U[0, i] * self.ts])
        
        for i in range(self.horizon):
            int_step = (i+1)*10
            opti.subject_to(X[0, int_step] - xr[0, int_step-1] <= 2)
            opti.subject_to(X[0, int_step] - xr[0, int_step-1] >= -2)
            opti.subject_to(X[1, int_step] - xr[1, int_step-1] <= 0.5)
            opti.subject_to(X[1, int_step] - xr[1, int_step-1] >= -0.5)

        opti.subject_to(X[3, :] >= 0)

        # Create the MPC solver
        opts = {'ipopt.print_level': 0, 'print_time': False}
        opti.solver('ipopt', opts)
        try:
            sol = opti.solve()
            states = sol.value(X)
            # act = sol.value(U)
        except:
            print("Solver failed. Returning best solution found.")
            states = opti.debug.value(X)
    
        traj = states.T[1:, :3]
    
        return traj

    def frenet_to_cartiesan(self, sl, ref_line):
        s, l = (sl[:, :, 0]*10 + 200), sl[:, :, 1]
        s = s.clamp(0, 1199)
        b = ref_line.shape[0]
        ref_points = ref_line[torch.arange(b).long()[:,None], s.long(), :]
        cartesian_x = ref_points[:, :, 0] - l * torch.sin(ref_points[:, :, 2])
        cartesian_y = ref_points[:, :, 1] + l * torch.cos(ref_points[:, :, 2])
        angle = ref_points[:, :, 2]

        return torch.stack([cartesian_x, cartesian_y, angle], dim=-1)