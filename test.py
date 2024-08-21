import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
from net import Net
from dataset import *
from metrics import *
import os
from thop import profile
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD test")
parser.add_argument("--model_names", default=['GCLNet'], type=list, help="model_name")
parser.add_argument("--pth_dirs", default=['NUDT-SIRST/GCLNet_best.pth.tar'], type=list, help="")
parser.add_argument("--dataset_dir", default='./dataset', type=str, help="train_dataset_dir")
parser.add_argument("--dataset_names", default=['NUDT-SIRST'], type=list,
                    help="dataset_name: 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K', 'SIRST3', 'NUDT-SIRST-Sea'")
parser.add_argument("--img_norm_cfg", default=None, type=dict,
                    help="specific a img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")

parser.add_argument("--save_img", default=True, type=bool, help="save image of or not")
parser.add_argument("--save_img_dir", type=str, default='./results/', help="path of saved image")
parser.add_argument("--save_log", type=str, default='./log/', help="path of saved .pth")
parser.add_argument("--threshold", type=float, default=0.5)

global opt
opt = parser.parse_args()

def test(): 
    test_set = TestSetLoader(opt.dataset_dir, opt.train_dataset_name, opt.test_dataset_name, opt.img_norm_cfg)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    
    net = Net(model_name=opt.model_name, mode='test').cuda()
    net.load_state_dict(torch.load(opt.pth_dir)['state_dict'])
    net.eval()

    input = torch.randn(1, 1, 256, 256).cuda()
    flops, params = profile(net, inputs=(input,))
    print('flops: %.3f G, params: %.2f M' % (flops / 1000 ** 3, params / 1000000.0))

    eval_FM = F_Measure()
    eval_ROC = ROCMetric(100)  ###bin越高，平均取点越密集，画线越细致
    eval_mIoU = mIoU() 
    eval_PD_FA = PD_FA()
    alltime = 0
    with torch.no_grad():
        for idx_iter, (img, gt_mask, size, img_dir) in enumerate(test_loader):
            img = Variable(img).cuda()

            start_time = time.time()
            pred = net.forward(img)
            end_time = time.time()

            # ~~~~~~~~~~~~~~~~~~~~~~fps~~~~~~~~~~~~~~~~~~~~
            if idx_iter > 50:
                alltime = alltime + end_time - start_time
            if idx_iter == 200:
                print('cost %f second' % (alltime / 150))
                print('fps: %f' % (1 / (alltime / 150)))

            pred = pred[:,:,:size[0],:size[1]]
            gt_mask = gt_mask[:,:,:size[0],:size[1]]
            eval_mIoU.update((pred>opt.threshold).cpu(), gt_mask)
            eval_PD_FA.update((pred[0,0,:,:]>opt.threshold).cpu(), gt_mask[0,0,:,:], size)
            eval_ROC.update(pred.cpu(), gt_mask)
            eval_FM.update(pred.cpu(), gt_mask)

            ### save img
            if opt.save_img == True:
                img_save = transforms.ToPILImage()((pred[0,0,:,:]).cpu())
                if not os.path.exists(opt.save_img_dir + opt.test_dataset_name + '/' + opt.model_name):
                    os.makedirs(opt.save_img_dir + opt.test_dataset_name + '/' + opt.model_name)
                img_save.save(opt.save_img_dir + opt.test_dataset_name + '/' + opt.model_name + '/' + img_dir[0] + '.png')
    
    results1 = eval_mIoU.get()
    results2 = eval_PD_FA.get()
    results4 = eval_FM.get()
    print("pixAcc, IoU, nIoU:\t" + str(results1))
    print("PD, FA:\t" + str(results2))
    print("F-Measure:\t" + str(results4))
    opt.f.write("pixAcc, mIoU:\t" + str(results1) + '\n')
    opt.f.write("PD, FA:\t" + str(results2) + '\n')

if __name__ == '__main__':
    opt.f = open(opt.save_log + 'test_' + (time.ctime()).replace(' ', '_').replace(':', '_') + '.txt', 'w')
    if opt.pth_dirs == None:
        for i in range(len(opt.model_names)):
            opt.model_name = opt.model_names[i]
            print(opt.model_name)
            opt.f.write(opt.model_name + '_400.pth.tar' + '\n')
            for dataset_name in opt.dataset_names:
                opt.dataset_name = dataset_name
                opt.train_dataset_name = opt.dataset_name
                opt.test_dataset_name = opt.dataset_name
                print(dataset_name)
                opt.f.write(opt.dataset_name + '\n')
                opt.pth_dir = opt.save_log + opt.dataset_name + '/' + opt.model_name + '_400.pth.tar'
                test()
            print('\n')
            opt.f.write('\n')
        opt.f.close()
    else:
        for model_name in opt.model_names:
            for dataset_name in opt.dataset_names:
                for pth_dir in opt.pth_dirs:
                    if dataset_name in pth_dir and model_name in pth_dir:
                        opt.test_dataset_name = dataset_name
                        opt.model_name = model_name
                        opt.train_dataset_name = pth_dir.split('/')[0]
                        print(pth_dir)
                        opt.f.write(pth_dir)
                        print(opt.test_dataset_name)
                        opt.f.write(opt.test_dataset_name + '\n')
                        opt.pth_dir = opt.save_log + pth_dir
                        test()
                        print('\n')
                        opt.f.write('\n')
        opt.f.close()
        
