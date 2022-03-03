from torch import nn
from sklearn.metrics import roc_curve, auc
import numpy as np
import torch
from torch.autograd import Variable
from copy import deepcopy
from PIL import Image
import sys
import argparse
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

def detection_test(data_type="Kvasir",
                 data_path=None,
                 model_dir="models",
                 epochs=256,
                 pretrained=True,
                 test_epochs=20,
                 test_loader=None,
                 learninig_rate=1e-3,
                 optim_name="SGD",
                 batch_size=64,
                 device = "cuda",
                 workers=16):
    path_name = data_path.split("/")[-2]
    model_path = model_dir+'/'+data_type+'_'+path_name+'_epoch'+str(test_epochs)+'.pth'
    # print(model_path)
    model = torch.load(model_path)
    label_score = []
    model.eval()
    for data in test_loader:
        X, Y = data
        X, Y = X.to(device), Y.to(device)
        Y = Y.long()
        output_pred = model.forward(X)
        predicted = torch.argmax(output_pred,axis=1)
        label_score += list(zip(Y.cpu().data.numpy().tolist(), torch.softmax(output_pred, dim = 1)[:, 0].cpu().data.numpy().tolist(), predicted.cpu().data.numpy().tolist()))

    labels, scores, predicts = zip(*label_score)
    labels = np.array(labels)
    scores = np.array(scores)
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=0)
    roc_auc = auc(fpr, tpr)
    roc_auc = round(roc_auc, 4)
    acc = np.sum(predicts==labels)/len(predicts)
    return roc_auc, acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training defect detection as described in the CutPaste Paper.')
    parser.add_argument('--type', default="EyeQ", type=str,
                        help='number of epochs to train the model , (default: 256)')

    parser.add_argument('--epochs', default=50, type=int,
                        help='number of epochs to train the model , (default: 256)')
    
    parser.add_argument('--model_dir', default="models",
                        help='output folder of the models , (default: models)')
    
    parser.add_argument('--no-pretrained', dest='pretrained', default=True, action='store_false',
                        help='use pretrained values to initalize vgg16 , (default: True)')
    
    parser.add_argument('--test_epochs', default=150, type=int,
                        help='interval to calculate the auc during trainig, if -1 do not calculate test scores, (default: 10)')                  

    parser.add_argument('--freeze_vgg', default=5, type=int,
                        help='number of epochs to freeze vgg (default: 20)')
    
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate (default: 1e-3)')

    parser.add_argument('--optim', default="sgd",
                        help='optimizing algorithm values:[sgd, adam] (dafault: "sgd")')

    parser.add_argument('--batch_size', default=32, type=int,
                        help='batch size, real batchsize is depending on cut paste config normal cutaout has effective batchsize of 2x batchsize (dafault: "64")')   

    
    parser.add_argument('--cuda', default=True,
                    help='use cuda for training (default: False)')
    
    parser.add_argument('--workers', default=16, type=int, help="number of workers to use for data loading (default:8)")


    args = parser.parse_args()
    print(args)
    
    
    device = "cuda" if args.cuda else "cpu"
    print(f"using device: {device}")

    epoch_AUC=0
    epoch_ACC=0
    train_path = "/mnt/huangwk/Dataset/EyeQ/split_EyeQ/train_multi_3_no_aug/"
    orig_transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5312014818191528, 0.519341230392456, 0.5130304098129272],
        std=[0.11665639281272888, 0.11394210159778595, 0.07831837981939316])
    ])
    test_data_path_all = ["/mnt/huangwk/Dataset/EyeQ/split_EyeQ/test_all/", "/mnt/huangwk/Dataset/EyeQ/split_EyeQ/test1/",
    "/mnt/huangwk/Dataset/EyeQ/split_EyeQ/test2/", "/mnt/huangwk/Dataset/EyeQ/split_EyeQ/test3/", "/mnt/huangwk/Dataset/EyeQ/split_EyeQ/test4/"]
    for epoch in range(5,50,5):
        print("Epoch{}".format(epoch))
        for i in range(5):
            test_data_path = test_data_path_all[i]
            test_set = ImageFolder(root=test_data_path, transform=orig_transform)
            test_dataloader = torch.utils.data.DataLoader(
                test_set,
                batch_size=64,
                num_workers=16,
                shuffle=False,
            )
            roc_auc, acc = detection_test(data_type=args.type,
                            data_path=train_path,
                            model_dir=args.model_dir,
                            epochs=args.epochs,
                            pretrained=args.pretrained,
                            test_epochs=epoch,
                            test_loader=test_dataloader,
                            learninig_rate=args.lr,
                            optim_name=args.optim,
                            batch_size=args.batch_size,
                            device=device,
                            workers=args.workers)
            print("RocAUC of test{} in epoch{} : {:.4f}".format(i, epoch, roc_auc))
            # print("ACC of test{} in epoch{} : {:.4f}".format(i, epoch, acc))
            torch.cuda.empty_cache()

    