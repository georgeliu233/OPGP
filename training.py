import csv
import argparse
import time
import sys

import torch
from torch import optim
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from waymo_open_dataset.protos import occupancy_flow_metrics_pb2
from google.protobuf import text_format

from model.InteractPlanner import InteractPlanner
from utils.net_utils import *
from metric import TrainingMetrics, ValidationMetrics


# define model training epoch
def training_epoch(train_data, optimizer, epoch, scheduler):
    
    model.train()
    current = 0
    start_time = time.time()
    size = len(train_data)
    epoch_loss = []
    train_metric = TrainingMetrics()
    i = 0
    for batch in train_data:
        # prepare data
        inputs, target = batch_to_dict(batch, local_rank , use_flow=use_flow)

        optimizer.zero_grad()
        # query the model
        bev_pred, traj, score = model(inputs)
        actor_loss, occ_loss, flow_loss = occupancy_loss(bev_pred, target, use_flow=use_flow)
        il_loss, _, gt_modes = imitation_loss(traj, score, target, args.use_planning)
    
        loss = il_loss + 100*(actor_loss + occ_loss)
        if use_flow:
            loss += flow_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        current += args.batch_size
        epoch_loss.append(loss.item())
        if isinstance(traj, list):
            traj, score = traj[-1], score[-1]
        ade, fde, l_il, l_ogm = train_metric.update(traj, score, gt_modes, target, il_loss, actor_loss + occ_loss, bev_pred)

        if dist.get_rank() == 0:
            sys.stdout.write(f"\rTrain: [{current:>6d}/{size*args.batch_size:>6d}]|Loss: {np.mean(epoch_loss):>.4f}-{l_il:>.4f}-{l_ogm:>.4f}|ADE:{ade:>.4f}-FDE:{fde:>.4f}|{(time.time()-start_time)/current:>.4f}s/sample")
            sys.stdout.flush()
        
        scheduler.step(epoch + i/size)
        i += 1

    results = train_metric.result()
    
    return np.mean(epoch_loss), results

# define model validation epoch
def validation_epoch(valid_data,epoch):
    epoch_metrics = ValidationMetrics()
    model.eval()
    current = 0
    start_time = time.time()
    size = len(valid_data)
    epoch_loss = []

    print(f'Validation...Epoch{epoch+1}')
    for batch in valid_data:
        # prepare data
        inputs, target = batch_to_dict(batch, local_rank, use_flow=use_flow)

        # query the model
        with torch.no_grad():
            bev_pred, traj, score = model(inputs)
            actor_loss, occ_loss, flow_loss = occupancy_loss(bev_pred, target, use_flow=use_flow)
            il_loss, _, gt_modes = imitation_loss(traj, score, target, args.use_planning)
            loss = il_loss + 100*(actor_loss + occ_loss) 
            if use_flow:
                loss += flow_loss
        # compute metrics
        epoch_loss.append(loss.item())
        if isinstance(traj, list):
            traj, score = traj[-1], score[-1]
        ade, fde, l_il, l_ogm, ogm_auc,_, occ_auc,_ = epoch_metrics.update(traj, score, bev_pred, 
                        gt_modes, target, il_loss, actor_loss + occ_loss)

        current += args.batch_size
        if dist.get_rank() == 0:
            sys.stdout.write(f"\r\Val: [{current:>6d}/{size*args.batch_size:>6d}]|Loss: {np.mean(epoch_loss):>.4f}-{l_il:>.4f}-{l_ogm:>.4f}|ADE:{ade:>.4f}-FDE:{fde:>.4f}{(time.time()-start_time)/current:>.4f}s/sample")
            sys.stdout.flush()
        
    # process metrics
    epoch_metrics = epoch_metrics.result()
    
    return epoch_metrics,np.mean(epoch_loss)

# Define model training process
def model_training(train_data, valid_data, epochs, save_dir):
    # define optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, eta_min=1e-6)

    for epoch in range(epochs):
        if dist.get_rank() == 0:
            print(f"Epoch {epoch+1}/{epochs}")
        
        if epoch<=continue_ep and continue_ep!=0:
            scheduler.step()
            continue

        train_data.sampler.set_epoch(epoch)
        valid_data.sampler.set_epoch(epoch)

        train_loss,train_res = training_epoch(train_data, optimizer, epoch, scheduler)
        valid_metrics,val_loss = validation_epoch(valid_data,epoch)

        # save to training log
        log = {'epoch': epoch+1, 'loss': train_loss, 'lr': optimizer.param_groups[0]['lr']}
        log.update(valid_metrics)

        if dist.get_rank() == 0:
            if epoch == 0:
                with open(save_dir + f'train_log.csv', 'a') as csv_file: 
                    writer = csv.writer(csv_file) 
                    writer.writerow(log.keys())
                    writer.writerow(log.values())
            else:
                with open(save_dir + f'train_log.csv', 'a') as csv_file: 
                    writer = csv.writer(csv_file)
                    writer.writerow(log.values())

        # adjust learning rate
        scheduler.step()

        # save model at the end of epoch    
        if dist.get_rank() == 0:
            torch.save(model.state_dict(), save_dir+f'model_{epoch+1}_{train_loss:4f}_{val_loss:4f}.pth')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--use_flow", type=bool, action='store_true', default=True, 
                    help='whether to use flow warp')

    parser.add_argument("--save_dir", type=str, default='',help='path to save logs')
    parser.add_argument("--data_dir", type=str, default='',
                    help='path to load preprocessed train & val sets')
    parser.add_argument("--model_dir", type=str, default='',
                    help='path to load models for continue training')

    args = parser.parse_args()
    local_rank = args.local_rank

    use_flow = args.use_flow

    config = occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig()
    config_text = f"""
        num_past_steps: {10}
        num_future_steps: {50}
        num_waypoints: {5}
        cumulative_waypoints: {'false'}
        normalize_sdc_yaw: true
        grid_height_cells: {128}
        grid_width_cells: {128}
        sdc_y_in_grid: {int(128*0.75)}
        sdc_x_in_grid: {64}
        pixels_per_meter: {1.6}
        agent_points_per_side_length: 48
        agent_points_per_side_width: 16
        """

    text_format.Parse(config_text, config)

    model = InteractPlanner(config, dim=args.dim, enc_layer=2, heads=8, dropout=0.1,
        timestep=5, decoder_dim=384, fpn_len=2, flow_pred=use_flow)

    save_dir = args.save_dir + f"models/"
    os.makedirs(save_dir,exist_ok=True)

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')

    model = model.to(local_rank)
    if args.model_dir!= '':
        kw_dict = {}
        for k,v in torch.load(save_dir + args.load_dir,map_location='cpu').items():
            kw_dict[k[7:]] = v
        model.load_state_dict(kw_dict)
        continue_ep = int(args.load_dir.split('_')[-3]) - 1
        print(f'model loaded!:epoch {continue_ep + 1}')
    else:
        continue_ep = 0

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    train_dataset = DrivingData(args.data_dir + f'train/*.npz',use_flow=use_flow)
    valid_dataset = DrivingData(args.data_dir + f'valid/*.npz',use_flow=use_flow)

    training_size = len(train_dataset)
    valid_size = len(valid_dataset)
    if dist.get_rank() == 0:
        print(f'Length train: {training_size}; Valid: {valid_size}')

    train_sampler = DistributedSampler(train_dataset)
    valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
    train_data = DataLoader(train_dataset, batch_size=args.batch_size, 
                    sampler=train_sampler, num_workers=16)
    valid_data = DataLoader(valid_dataset, batch_size=args.batch_size, 
                    sampler=valid_sampler, num_workers=4)

    model_training(train_data, valid_data, args.epochs, save_dir)
