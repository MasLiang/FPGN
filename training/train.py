import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from lut_layer import *
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import warnings
import resnet
warnings.filterwarnings('ignore')


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.quant_r = lut_quant(8, 3, kernel_size=1, stride=1, padding=[0,0])
        self.quant_g = lut_quant(8, 3, kernel_size=1, stride=1, padding=[0,0])
        self.quant_b = lut_quant(8, 3, kernel_size=1, stride=1, padding=[0,0])
        self.lut_conv0 = lut_conv(9, 16, kernel_size=3, stride=2, padding=[1,1])
        self.lut_resd1 = LambdaLayer(lambda x:
				     F.pad(F.avg_pool2d(x,kernel_size=2), (0, 0, 0, 0, 8, 8), "constant", 0))
        self.lut_conv7 = lut_conv(16, 32, kernel_size=3, stride=2, padding=[1,1])
        self.lut_resd2 = LambdaLayer(lambda x:
				     F.pad(F.avg_pool2d(x,kernel_size=2), (0, 0, 0, 0, 16, 16), "constant", 0))
        self.lut_conv13 = lut_conv(32, 64, kernel_size=3, stride=2, padding=[1,1])

        self.lut_fc3 = lut_fc(4*4*64, 2000)
        self.lut_fc4 = lut_fc(2000, 2000)
 
        self.bnquant = nn.BatchNorm2d(9, affine=False)
        self.bn0 = nn.BatchNorm2d(16, affine=False)
        self.bn7 = nn.BatchNorm2d(32, affine=False)
        self.bn13 = nn.BatchNorm2d(64, affine=False)
        self.w1 = nn.Parameter(torch.ones(18)*0.1)

        self.tau = 1
        self.q_y = True
        self.q_w = False
        self.hard = False
        self.res_flg = 1

    def forward(self, x):
        x = quant(x, self.tau, True)
        x_r = x[:,:8,...]
        x_g = x[:,8:16,...]
        x_b = x[:,16:,...]
        x_r = self.quant_r(x_r)
        x_g = self.quant_g(x_g)
        x_b = self.quant_b(x_b)
        x = torch.cat([x_r, x_g, x_b], dim=1)
        x = self.bnquant(x)
        x = binary_gumbel_softmax(x, tau=self.tau, hard=self.hard, w0y1=1)

        x = self.lut_conv0(x)
        x = self.bn0(x)

        x_r = self.lut_resd1(x)
        x = binary_gumbel_softmax(x, tau=self.tau, hard=self.hard, w0y1=1)
        x = self.lut_conv7(x)
        x = self.bn7(x)
        x = F.sigmoid(self.w1[6])*x+self.res_flg*x_r

        x_r = self.lut_resd2(x)
        x = binary_gumbel_softmax(x, tau=self.tau, hard=self.hard, w0y1=1)
        x = self.lut_conv13(x)
        x = self.bn13(x) 
        x = F.sigmoid(self.w1[12])*x+self.res_flg*x_r

        x = x.view(x.size(0), -1)
        x = x.unsqueeze(1)
        x = binary_gumbel_softmax(x, tau=self.tau, hard=self.hard, w0y1=1)
        x = self.lut_fc3(x)
        x = self.lut_fc4(x)
        x = x.squeeze(1)
        x = x.reshape(*x.shape[:-1], 10, x.shape[-1]//10).sum(-1)/10
        return x

def quant(data, tau, hard_q):
    bit_num=8
    batch_size,in_channel,H,W = data.size()
    #max_x = data.max()
    #min_x = data.min()
    #coeffs = []
    #data = (data-min_x)/(max_x-min_x)*((2**bit_num)-1)
    q = torch.round(data).int()
    dist = q-data
    bits = []
    for i in reversed(range(bit_num)):
        bit = ((q >> i) & 1).float()
        bits.append(bit)
    data = torch.stack(bits, dim=-1)
    if not hard_q:
        powers = 2 ** torch.arange(bit_num-1, -1, -1, device=data.device).float()  # [32, 16, ..., 1]
        powers = powers.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        dist = (dist/(2**bit_num-1)*tau).unsqueeze(-1)
        data = data+dist
    data = data.permute(0,1,4,2,3).contiguous().view(batch_size,in_channel*bit_num,H,W)
    
    return data

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12354'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(args, model, device, train_loader, optimizer, epoch, scaler):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            #data = quant(data, model.tau, True)
            output = model(data)
            loss = F.cross_entropy(output, target)
            #loss_ce = F.cross_entropy(output, target)
            #kl_loss = nn.KLDivLoss(reduction='batchmean')
            #loss_kd = kl_loss(F.log_softmax(output/4), F.softmax(target_soft/4))
            #loss = 0.8*loss_ce+0.2*loss_kd
            #for name, module in model.named_modules():
            #    if isinstance(module, lut_kernel):
            #        loss += module.lut_loss*0.01
        #loss.backward()
        #optimizer.step()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if batch_idx % args.log_interval == 0 and dist.get_rank()==0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            with torch.cuda.amp.autocast():
                #data = quant(data, model.tau, True)
                output = model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct, test_loss

def main_worker(rank, world_size, args):
    setup(rank, world_size)
    torch.manual_seed(args.seed)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.PILToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    transform_test = transforms.Compose([
        transforms.PILToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    dataset1 = datasets.CIFAR10('../data', train=True, download=True,
                       transform=transform_train)
    dataset2 = datasets.CIFAR10('../data', train=False,
                       transform=transform_test)
    train_sampler = DistributedSampler(dataset1, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(dataset1, batch_size=args.batch_size, sampler=train_sampler, num_workers=8, pin_memory=True)
    test_loader = DataLoader(dataset2, batch_size=args.test_batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model = Net()
    #model = resnet.__dict__['resnet20']().to(rank)
    checkpoint = torch.load('base_3layer_small_quantbn.pt', map_location='cpu')
    state_dict = checkpoint if isinstance(checkpoint, dict) and 'state_dict' not in checkpoint else checkpoint.get('state_dict', checkpoint)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(rank)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    max_acc = 0

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=20, gamma=args.gamma)
    scaler = torch.cuda.amp.GradScaler()
    tau = 1
    res_flg = 1
    hard = 0

    for name, module in model.named_modules():
        if isinstance(module, lut_quant):
            module.hard = True
            module.cfg()

    for epoch in range(1, args.epochs + 1):
        if res_flg>0:
            res_flg -= 0.4
        else:     
            res_flg = 0
        setattr(model.module, "res_flg", res_flg)
        if epoch>=1:
            train_sampler.set_epoch(epoch)
            train(args, model, device, train_loader, optimizer, epoch, scaler)
            if tau <2:
                acc, loss = test(model, device, test_loader)
            if rank==0:
                print(F.sigmoid(model.module.w1))
        else:
            acc = 0
        if epoch<30:
            scheduler.step()

        if epoch>30:
            if tau<2:
                tau += 0.1
            elif tau<30 and hard>=10:
                tau += 0.2
        if tau>=2:
            if hard>=0:
                model.module.lut_conv0.hard = True
                model.module.lut_conv0.cfg()
            if hard>=1:
                model.module.lut_conv7.hard = True
                model.module.lut_conv7.cfg()
            if hard>=2:
                model.module.lut_conv13.hard = True
                model.module.lut_conv13.cfg()
            if hard>=3:
                model.module.lut_fc3.hard = True
                model.module.lut_fc3.cfg()
            if hard>=4:
                model.module.lut_fc4.hard = True
                model.module.lut_fc4.cfg()
            if hard>=10:
                setattr(model.module, "hard", True)
                acc, loss = test(model, device, test_loader)
                if max_acc<acc:
                    max_acc = acc
                    if rank == 0:
                        print("max acc: ", max_acc)
                        torch.save(model.state_dict(), "base_3layer_small_quantbn_nores.pt")
            hard += 0.1
        setattr(model.module, "tau", tau)
        for name, module in model.named_modules():
            if isinstance(module, lut_fc):
                module.tau = tau
                module.cfg()
            if isinstance(module, lut_conv):
                module.tau = tau
                module.cfg()
        exit_flag = torch.zeros(1, dtype=torch.int, device=device)
        if rank == 0:
            exit_flag[0] = 1 if tau >= 20 else 0
        dist.broadcast(exit_flag, src=0)
        if exit_flag.item() == 1:
            dist.barrier()
            break
    cleanup()

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.5, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)

    
if __name__ == '__main__':
    main()
