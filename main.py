import fire
import numpy
import json
import torch
import pickle

from collections import OrderedDict
from easydict import EasyDict
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

import util as util

from model import get_network, DownSample2DMatlab, UpSample2DMatlab
from pruning import pruning_map
from src.config import TrainingConfig, PruningConfig
from src.dataset import SRDatasetFromDIV2K
from src.loss import get_loss


def train():
    print("[INFO] Set configuration")
    config = TrainingConfig()
    config.prepare()
    # FIXME: After this issue resolved
    # https://github.com/makinacorpus/easydict/issues/20
    config = config.cfg
    print("[INFO] Set random seed")
    numpy.random.seed(config.EXP.seed)
    torch.manual_seed(config.EXP.seed)
    torch.cuda.manual_seed(config.EXP.seed)
    print("[INFO] Get training dataset and data_loader")
    train_set = SRDatasetFromDIV2K(dir_path=config.DATA.div2k_dir,
                                   transform=transforms.Compose([
                                       transforms.RandomCrop(
                                           [config.DATA.hr_size, config.DATA.hr_size]),
                                       transforms.RandomRotation(180),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomVerticalFlip(),
                                       transforms.ToTensor()]),
                                   transform_lr=transforms.Compose([
                                       transforms.RandomCrop(
                                           [config.DATA.lr_size, config.DATA.lr_size]),
                                       transforms.RandomRotation(180),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomVerticalFlip(),
                                       transforms.ToTensor()
                                   ]))
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_set,
        num_workers=4,
        batch_size=config.TRAIN.batch_size,
        shuffle=True)
    print("[INFO] Prepare net, optimizer, loss for training")
    net = get_network(config.TRAIN.network).cuda()
    # Use multiple gpus if possible
    if torch.cuda.device_count() > 1:
        print(
            f"[INFO] Use multiple gpus with count {torch.cuda.device_count()}")
    optimizer = torch.optim.Adam(
        net.parameters(), lr=config.TRAIN.learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=config.TRAIN.lr_step_milestones,
        gamma=config.TRAIN.lr_step_gamma)
    criterion = get_loss(config.TRAIN.loss)
    # Re-load from checkpoint, this can be rewinding
    if config.TRAIN.resume:
        print(f"[INFO] Load checkpoint from {config.TRAIN.load_checkpoint_path}")
        checkpoint = torch.load(config.TRAIN.load_checkpoint_path)
        net.load_state_dict(checkpoint['net'])
        net = torch.nn.DataParallel(net)
        start_epoch = 0
        if config.TRAIN.network in ['VDSR']:
            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
        if config.TRAIN.network in ['RCAN']:
            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        start_epoch = 0
    # Set pruning mask
    if config.TRAIN.pruning:
        json_path = f"{config.TRAIN.pruning_dir}/pruning-report.json"
        print(f"[INFO] Load pruning report from {json_path}")
        with open(json_path, 'r') as f:
            pruning_report = json.load(f)
            pruned_index = int(pruning_report['statistics']['argmax']) + 1
        channel_mask_path = f"{config.TRAIN.pruning_dir}/channel_mask_{pruned_index}.pickle"
        print(f"[INFO] Load mask index from {channel_mask_path}")
        with open(channel_mask_path, 'rb') as f:
            channel_mask = pickle.load(f)
        pruning_method = pruning_report['meta']['config']['method']
        pruning_rate = pruning_report['meta']['config']['pruning_rate']
        if pruning_method in pruning_map.keys():
            pruning_method = pruning_map[pruning_method]
            pruning = pruning_method(net.parameters(), pruning_rate)
            pruning.update(channel_mask)
        else:
            raise Exception(
                f"Not proper config.PRUNE.method, cur var is: {config.PRUNE.method}")
    print("[INFO] Start training loop")
    net.train()
    writer = SummaryWriter(config.SAVE.summary_dir)
    print("[INFO] Save checkpoint before training")
    torch.save({
        'config': config,
        'epoch': 0,
        'net': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }, f"{config.SAVE.checkpoint_dir}/SRPruning_epoch_0.pth")
    net_parallel = torch.nn.DataParallel(net)
    for epoch in tqdm(range(start_epoch + 1,
                            config.TRAIN.end_epoch + 1), position=0, leave=True):
        for index, hr_image in enumerate(tqdm(train_dataloader, position=1, leave=False)):
            # Make low resolution input from high resolution image
            hr_image = hr_image.cuda()
            lr_image = DownSample2DMatlab(hr_image, 1 / 4, cuda=True)
            # Zero masked value with pruning
            if config.TRAIN.pruning:
                pruning.zero()
            # Forward
            out = net_parallel(lr_image)
            loss = criterion(out, hr_image)
            # Back-propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch == 1 or epoch % config.TRAIN.period_log == 0:
            # Add images to tensorboard
            writer.add_images('1 hr', hr_image.clamp(0, 1))
            writer.add_images('2 out', out.clamp(0, 1))
            #writer.add_images('3 bicubic', bicubic_image.clamp(0, 1))
            #writer.add_images('4 model_output', model_output)# Memory
            writer.add_images('5 lr', lr_image.clamp(0, 1))
            # Add values to tensorboard
            writer.add_scalar(
                '1 MSE', loss.item(), global_step=epoch)
            app = util.psnr_set5(net,
                                 set5_dir=config.DATA.set5_dir,
                                 save_dir=config.SAVE.save_dir)
            writer.add_scalar(
                '2 Set5 PSNR out', app, global_step=epoch)
            #writer.add_scalar(
            #    '3 Set5 PSNR bicubic', apb, global_step=epoch)
            writer.add_scalar(
                '4 learning rate', optimizer.param_groups[0]['lr'],
                global_step=epoch)
            writer.flush()
        if epoch % config.TRAIN.period_save == 0:
            # Save checkpoint
            torch.save({
                'config': config,
                'epoch': epoch,
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, f"{config.SAVE.checkpoint_dir}/SRPruning_epoch_{epoch}.pth")
        scheduler.step()


def pruning():
    print("[INFO] Set configuration")
    config = PruningConfig()
    config.prepare()
    # FIXME: After this issue resolved
    # https://github.com/makinacorpus/easydict/issues/20
    config = config.cfg
    print("[INFO] Set random seed")
    numpy.random.seed(config.EXP.seed)
    torch.manual_seed(config.EXP.seed)
    torch.cuda.manual_seed(config.EXP.seed)
    print(
        f"[INFO] Load from checkpoint {config.PRUNE.trained_checkpoint_path}")
    checkpoint = torch.load(config.PRUNE.trained_checkpoint_path)
    print(f"[INFO] Get psnr set5 from randomly pruned network")
    result = EasyDict()
    psnrs = []
    for i in tqdm(range(1, config.PRUNE.random_prune_try_cnt + 1)):
        # Load net
        net = VDSR().cuda()
        net = torch.nn.DataParallel(net)
        net.load_state_dict(checkpoint['net'])
        # Prune
        if config.PRUNE.method in pruning_map.keys():
            pruning_method = pruning_map[config.PRUNE.method]
            pruning = pruning_method(
                net.parameters(), config.PRUNE.pruning_rate)
        else:
            raise Exception(
                f"Not proper config.PRUNE.method, cur var is: {config.PRUNE.method}")
        pruning.update()
        pruning.zero()
        # Calculate psnr5
        psnr, _ = util.psnr_set5(net,
                            set5_dir=config.DATA.set5_dir,
                            save_dir=config.SAVE.save_dir,
                            save=False)
        # Save results
        with open(f"{config.SAVE.pruning_dir}/channel_mask_{i}.pickle", 'wb') as f:
            pickle.dump(pruning.channel_mask, f)
        psnrs.append(psnr)
    result.psnrs = psnrs
    print(f"[INFO] Get meta and statistics of experiment")
    result.meta = EasyDict()
    result.meta.config = config.PRUNE
    result.statistics = EasyDict()
    psnrs = numpy.array(psnrs)
    result.statistics.min = psnrs.min()
    result.statistics.max = psnrs.max()
    result.statistics.mean = psnrs.mean()
    result.statistics.argmax = int(psnrs.argmax())
    json_path = f"{config.SAVE.pruning_dir}/pruning-report.json"
    print(f"[INFO] Save masks and psnr value to {json_path}")
    with open(json_path, 'w') as f:
        json_txt = json.dumps(result, indent=4)
        f.write(json_txt)


def test(model_path, data_dir='../dataset', save=False):
    # --------------------------------
    # basic settings
    # --------------------------------
    testsets = f'{data_dir}/DIV2K'
    testset_L = f'DIV2K_valid_LR_bicubic'

    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # --------------------------------
    # load model
    # --------------------------------
    # model = MSRResNet(in_nc=3, out_nc=3, nf=64, nb=16, upscale=4)
    model = get_network('CARN')
    checkpoint = torch.load(model_path)
    net_state_dict = checkpoint
    model.load_state_dict(net_state_dict)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    # number of parameters
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    print(f'Params number: {number_parameters}')

    # --------------------------------
    # read image
    # --------------------------------
    L_folder = Path(testsets) / testset_L / 'X4'
    E_folder = Path(testsets) / testset_L / '_results'
    E_folder.mkdir(parents=True, exist_ok=True)

    # record PSNR, runtime
    test_results = OrderedDict()
    test_results['runtime'] = []

    print(L_folder)
    print(E_folder)
    idx = 0

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for img in sorted(list(L_folder.glob('*.png'))):

        # --------------------------------
        # (1) img_L
        # --------------------------------
        idx += 1
        img_name = img.stem 
        ext = img.suffix
        print(f"Load image from {img}")
        img_L = util.imread_uint(img, n_channels=3)
        img_L = util.uint2tensor4(img_L)
        img_L = img_L.to(device)

        start.record()
        img_E = model(img_L)
        end.record()
        torch.cuda.synchronize()
        test_results['runtime'].append(start.elapsed_time(end))  # milliseconds


#        torch.cuda.synchronize()
#        start = time.time()
#        img_E = model(img_L)
#        torch.cuda.synchronize()
#        end = time.time()
#        test_results['runtime'].append(end-start)  # seconds

        # --------------------------------
        # (2) img_E
        # --------------------------------
        img_E = util.tensor2uint(img_E)

        if save:
            new_name = '{:3d}'.format(int(img_name.split('x')[0]))
            path = os.path.join(E_folder, new_name+ext)
            print('Save {:4d} to {:10s}'.format(idx, path))
            util.imsave(img_E, path)
    ave_runtime = sum(test_results['runtime']) / len(test_results['runtime']) / 1000.0
    print('------> Average runtime of ({}) is : {:.6f} seconds'.format(L_folder, ave_runtime))


def hello():
    print("Hello, World!")


if __name__ == '__main__':
    fire.Fire({
        'train': train,
        'pruning': pruning,
        'test': test,
        'hello': hello
    })
