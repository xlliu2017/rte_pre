# === Standard Libraries ===
import sys, os
import logging
import argparse

# === Third-Party Libraries ===
import torch
from torch.utils.data import DataLoader
import scipy.io
import sys
import time
import numpy as np

# === Local Modules ===
from RTE_mgmodel import MG_precond
from RTE_dataloader import UnsuperviseDataset
from utils import Timer, save_loss_history, save_model, greenprint, check_cuda_mem
from mgRTE.RTE_ATFPS_func import func_inflow_Dirichlet, restrict_basis, squeeze_basis
from gmres import mygmres2dtorch
from RTE_datagenerator import RTECoef

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# python main.py --data_type interface --num_coef 1000 --num_data 10000 --cuda_device cuda:1 --tol_exp 1000000

# default value for setting
parser = argparse.ArgumentParser(description="Train model with configurable parameters.")
parser.add_argument('--cuda_device',     type = str,   nargs='?', default = 'cuda:0')
parser.add_argument('--data_type',       type = str,   nargs='?', default = 'diffusion')
parser.add_argument('--num_coef',        type = int,   nargs='?', default = 1000)
parser.add_argument('--num_data',        type = int,   nargs='?', default = 10000)
# default value for discretized parameters
parser.add_argument('--mesh_size',       type = int,   nargs='?', default = 16)
parser.add_argument('--xl',              type = float, nargs='?', default = 0)
parser.add_argument('--xr',              type = float, nargs='?', default = 1)
parser.add_argument('--yl',              type = float, nargs='?', default = 0)
parser.add_argument('--yr',              type = float, nargs='?', default = 1)
parser.add_argument('--N',               type = int,   nargs='?', default = 1)
parser.add_argument('--tol_exp',         type = int, nargs='?', default = 5)
# default value for network structure
parser.add_argument('--u_channels_list', type=int, nargs='+', default=[16, 32, 64, 128])
parser.add_argument('--a_channels_list', type=int, nargs='+', default=[16, 32, 64, 128])
parser.add_argument('--mg_levels',       type=int, nargs='+', default=[2, 2, 2, 4])
# default value for training
parser.add_argument('--lr',              type = float, nargs='?', default = 0.001)
parser.add_argument('--batch_size',      type = int,   nargs='?', default = 256)
parser.add_argument('--num_epochs',      type = int,   nargs='?', default = 1000)
parser.add_argument('--scheduler_name',  type = str,   nargs='?', default = 'CosineAnnealingLR')
# for stepLR
parser.add_argument('--step_size',       type = int,   nargs='?', default = 10)
parser.add_argument('--gamma',           type = float, nargs='?', default = 0.5)
# for CosineAnnealingLR
parser.add_argument('--T_max',           type = int,   nargs='?', default = 100)
parser.add_argument('--eta_min',         type = float, nargs='?', default = 1e-6)
# default value for validation
parser.add_argument('--val_steps',       type = int,   nargs='?', default = 5)
args = parser.parse_args()

# some path
current_dir = os.path.dirname(__file__)
quadrature_path = current_dir + '/discretized_parameters/quadrature2DN'+str(args.N)+'.mat'
Kappa_path =  current_dir + '/discretized_parameters/KappaN'+str(args.N)+'g0.mat'
coef_path = current_dir + '/discretized_parameters/'+args.data_type+'N'+str(args.N)+'I'+str(args.mesh_size)+'d'+str(args.num_coef)+'tol'+str(args.tol_exp)+'.pt'
if args.scheduler_name == 'StepLR':
    sub_dir = args.scheduler_name+'_stepsize'+str(args.step_size)+'_gamma'+str(args.gamma)+'/lr'+str(args.lr)+'/'
elif args.scheduler_name == 'CosineAnnealingLR':
    sub_dir = args.scheduler_name+'_T_max'+str(args.T_max)+'/lr'+str(args.lr)+'/'
else: 
    raise ValueError('scheduler %s not supported' % args.scheduler_name)
output_train_loss_dir = current_dir + '/train_loss/' + sub_dir
output_val_loss_dir = current_dir + '/val_loss/' + sub_dir
output_model_dir = current_dir + '/model_precond/' + sub_dir
os.makedirs(output_train_loss_dir, exist_ok=True)
os.makedirs(output_val_loss_dir, exist_ok=True)
os.makedirs(output_model_dir, exist_ok=True)


# Determine compute device
device = torch.device(args.cuda_device if torch.cuda.is_available() else 'cpu')
greenprint(f"Using device: {device}")   
# angular domain discrtization by DOM
quadrature = scipy.io.loadmat(quadrature_path)
ct = torch.tensor(quadrature['ct']).squeeze(-1).to(torch.float32).to(device)
st = torch.tensor(quadrature['st']).squeeze(-1).to(torch.float32).to(device)
omega = torch.tensor(quadrature['omega']).squeeze(-1).to(torch.float32).to(device)
M = quadrature['M'].item()
Kappa = torch.tensor(scipy.io.loadmat(Kappa_path)['Kappa']).to(torch.float32).to(device)

# Initialize MG_precond with user-specified channels and levels
model = MG_precond(M, args.u_channels_list, args.a_channels_list, args.mg_levels)
model.to(device)
# number of parameters 
total_params = sum(p.numel() for p in model.parameters())
print(f"number of total parameters: {total_params:,}")
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"number of trainable paramters {trainable_params:,}")

"""
Train the MG_precond model in an unsupervised fashion:
  - Minimize residual loss ||A(u) - source||.
  - Save per-epoch loss history.
"""
# Parse training hyperparameters with proper types
# Configure optimizer, loss function, and learning rate scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
loss_fn = torch.nn.MSELoss().to(device)
if args.scheduler_name == 'StepLR':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = args.step_size, gamma = args.gamma)
elif args.scheduler_name == 'CosineAnnealingLR':
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min = args.eta_min)
else: 
    raise ValueError('scheduler %s not supported' % args.scheduler_name)


# Prepare dataset and dataloader for boundary-based unsupervised data
RTEData = torch.load(coef_path, weights_only=True)
dataset = UnsuperviseDataset(args.N, args.mesh_size, args.num_coef, args.num_data, RTEData)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=args.batch_size,shuffle=True)
timer = Timer('gpu')
epoch_loss_list_train = []
epoch_loss_list_eval=[]
try:
    for epoch in range(args.num_epochs):
        # train
        model.train()
        # Reset timer and epoch loss accumulator
        timer.reset()
        epoch_loss=0
        for batch, [Coef_ins, fsmLRBTC_ins, I2A_ins, MLRBT_ins, VecSize_ins, rhs_ins] in enumerate(train_loader):
            Coef_ins = Coef_ins.to(device)
            fsmLRBTC_ins = fsmLRBTC_ins.to(device)
            I2A_ins = I2A_ins.to(device)
            MLRBT_ins = MLRBT_ins.to(device)
            VecSize_ins = VecSize_ins.to(device)
            rhs_ins =  rhs_ins.to(device)
            
            # model.setup(Coef_ins)
            # pred = model(rhs_ins)
            [a_list, inv_a_list] = model.setup(Coef_ins)
            pred = model(rhs_ins, a_list, inv_a_list)
            Apred = func_inflow_Dirichlet(pred, I2A_ins, fsmLRBTC_ins, MLRBT_ins, VecSize_ins, M, args.mesh_size, args.mesh_size, device)
            
            rhs_ins_selected, _ = squeeze_basis(rhs_ins, VecSize_ins)
            Apred_selected, _ = squeeze_basis(Apred, VecSize_ins)
            # loss = loss_fn(Apred, rhs_ins)
            loss = loss_fn(Apred_selected, rhs_ins_selected)

            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_loss += loss.item()
        # Average loss over all batches
        epoch_loss /= len(train_loader)
        scheduler.step()
        check_cuda_mem(device)
        epoch_loss_list_train.append(epoch_loss)
        logging.info(
                    f"data type: {args.data_type}, "
                    f"Epoch: [{epoch+1}/{args.num_epochs}], "
                    f"Train Loss: {epoch_loss:.4e}, "
                    f"time: {timer.elapsed('s'):.2f}s, "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                    )
        # validation
        model.eval()
        if (epoch+1) % args.val_steps == 0:
            with torch.no_grad():
                timer.reset()
                epoch_loss=0
                for batch, [Coef_ins, fsmLRBTC_ins, I2A_ins, MLRBT_ins, VecSize_ins, rhs_ins] in enumerate(test_loader):
                    Coef_ins = Coef_ins.to(device)
                    fsmLRBTC_ins = fsmLRBTC_ins.to(device)
                    I2A_ins = I2A_ins.to(device)
                    MLRBT_ins = MLRBT_ins.to(device)
                    VecSize_ins = VecSize_ins.to(device)
                    # rhs_ins, _ = restrict_basis(rhs_ins.to(device), VecSize_ins, M, size, size, device)
                    rhs_ins =  rhs_ins.to(device)
            
                    # model.setup(Coef_ins)
                    # pred = model(rhs_ins)
                    [a_list, inv_a_list] = model.setup(Coef_ins)
                    pred = model(rhs_ins, a_list, inv_a_list)
                    
                    Apred = func_inflow_Dirichlet(pred, I2A_ins, fsmLRBTC_ins, MLRBT_ins, VecSize_ins, M, args.mesh_size, args.mesh_size, device)
                    
                    rhs_ins_selected, _ = squeeze_basis(rhs_ins, VecSize_ins)
                    Apred_selected, _ = squeeze_basis(Apred, VecSize_ins)
                    # loss = loss_fn(Apred, rhs_ins)
                    loss = loss_fn(Apred_selected, rhs_ins_selected)
                    
                    epoch_loss += loss.item()
                    
                # Average loss over all batches
                epoch_loss /= len(test_loader)
                epoch_loss_list_eval.append(epoch_loss)
                logging.info(
                            f"Epoch: [{epoch+1}/{args.num_epochs}], "
                            f"Val Loss: {epoch_loss:.4e}, "
                            f"time: {timer.elapsed('s'):.2f}s"
                            )
    fn = args.data_type+'I'+str(args.mesh_size)+'N'+str(args.N)+'L'+str(len(args.u_channels_list))+'batch'+str(args.batch_size)+'epoch'+str(epoch+1)+'d'+str(args.num_coef)+str(args.num_data)+'tol'+str(args.tol_exp)+'_multilevel'
    # save_loss_history(output_train_loss_dir, epoch_loss_list_train, fn)
    # save_loss_history(output_val_loss_dir, epoch_loss_list_eval, fn)
    # save_model(output_model_dir, model, fn)
except KeyboardInterrupt:
    print("\nTraining interrupted by user (Ctrl+C)")
    
    user_choice = input("Save current model? (y/n): ").lower().strip()
    
    if user_choice == 'y':
        fn = args.data_type+'I'+str(args.mesh_size)+'N'+str(args.N)+'L'+str(len(args.u_channels_list))+'batch'+str(args.batch_size)+'epoch'+str(epoch+1)+'d'+str(args.num_coef)+str(args.num_data)+'tol'+str(args.tol_exp)+'_multilevel'
        save_loss_history(output_train_loss_dir, epoch_loss_list_train, fn)
        save_loss_history(output_val_loss_dir, epoch_loss_list_eval, fn)
        save_model(output_model_dir, model, fn)
    else:
        print("Model not saved.")
    
    print("Exiting.")
    

