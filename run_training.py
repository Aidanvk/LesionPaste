# code is basicly:https://github.com/google-research/deep_representation_one_class
from pathlib import Path
from tqdm import tqdm
import datetime
import argparse
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import roc_curve, auc

from dataset import load_data
from model import ProjectionNet
# from eval import eval_model

def evaluate(model, dataloader, device):
        label_score = []
        model.eval()
        for data in dataloader:
            X, Y = data
            X, Y = X.to(device), Y.to(device)
            Y = Y.long()
            output_pred = model.forward(X)
            predicted = torch.argmax(output_pred,axis=1)
            label_score += list(zip(Y.cpu().data.numpy().tolist(), 
            torch.softmax(output_pred, dim = 1)[:, 0].cpu().data.numpy().tolist(), 
            predicted.cpu().data.numpy().tolist()))
        labels, scores, predicts = zip(*label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=0)
        roc_auc = auc(fpr, tpr)
        acc = np.sum(predicts==labels)/len(predicts)
        model.train()
        return roc_auc, acc

def run_training(data_type="EyeQ",
                 train_data_path=None,
                 model_dir="models",
                 epochs=256,
                 pretrained=True,
                 test_epochs=10,
                 freeze_vgg=20,
                 learninig_rate=1e-3,
                 optim_name="SGD",
                 batch_size=64,
                 device = "cuda",
                 workers=16):
    torch.multiprocessing.freeze_support()
    # Temperature Hyperparameter currently not used
    temperature = 0.2

    weight_decay = 0.00003
    momentum = 0.9
    model_name = f"model-{data_type}" + '-{date:%Y-%m-%d_%H_%M_%S}'.format(date=datetime.datetime.now() )
    # train_data_path = "/mnt/huangwk/Dataset/EyeQ/split_EyeQ/not_split_trainset/"
    test_data_path = "/mnt/huangwk/Dataset/EyeQ/split_EyeQ/test/"
    test_data_path_all = ["/mnt/huangwk/Dataset/EyeQ/split_EyeQ/test_all/", 
    "/mnt/huangwk/Dataset/EyeQ/split_EyeQ/test1/",
    "/mnt/huangwk/Dataset/EyeQ/split_EyeQ/test2/",
    "/mnt/huangwk/Dataset/EyeQ/split_EyeQ/test3/",
    "/mnt/huangwk/Dataset/EyeQ/split_EyeQ/test4/"]
    # data_path = "/mnt/huangwk/Dataset/MosMedData/studies/train/"
    # test_data_path = "/mnt/huangwk/Dataset/MosMedData/studies/test/"
    path_name = train_data_path.split("/")[-2]
    print(path_name)
    
    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5312014818191528, 0.519341230392456, 0.5130304098129272],
        std=[0.11665639281272888, 0.11394210159778595, 0.07831837981939316])
    ])
    train_loader, _ = load_data(batch_size, workers, train_data_path, test_data_path, transform)

    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter(Path("logdirs") / model_name)

    # create Model:
    num_classes = 2
    model = ProjectionNet(pretrained=pretrained, num_classes=num_classes)
    
    model.to(device)
    # print(model)

    if freeze_vgg > 0:
        model.freeze_vgg()

    loss_fn = torch.nn.CrossEntropyLoss()
    if optim_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learninig_rate, momentum=momentum,  weight_decay=weight_decay)
        scheduler = CosineAnnealingWarmRestarts(optimizer, epochs)
        #scheduler = None
    elif optim_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learninig_rate, weight_decay=weight_decay)
        scheduler = None
    else:
        print(f"ERROR unkown optimizer: {optim_name}")
    
    for epoch in range(1, epochs + 1):
        if scheduler is not None:
                scheduler.step()
        step = 0
        epoch_loss=0
        epoch_acc=0
        progress = tqdm(enumerate(train_loader))
        for step, train_data in progress:
            model.train()
            if epoch == freeze_vgg:
                model.unfreeze()
            # print(len(data[0]))
            X, y = train_data
            X, y = X.to(device), y.to(device)
            y = y.long()
            # zero the parameter gradients
            optimizer.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.requires_grad_(True)

            # regulize weights:
            loss.backward()
            optimizer.step()
    #         predicted = torch.argmax(ip,axis=0)
            predicted = torch.argmax(logits,axis=1)
            accuracy = torch.true_divide(torch.sum(predicted==y), predicted.size(0))
            epoch_loss += loss.item()
            epoch_acc += accuracy
            avg_loss = epoch_loss / (step + 1)
            avg_acc = epoch_acc / (step + 1)
            progress.set_description(
                'epoch: {}, loss: {:.6f}, acc: {:.4f}'
                .format(epoch, avg_loss, avg_acc)
            )    
        if epoch % 10 == 0:
            torch.save(model, model_dir / f"{data_type+'_'+path_name+'_epoch'+str(epoch)}.pth")
            for k in range(5):
                test_data_path = test_data_path_all[k]
                test_set = ImageFolder(root=test_data_path, transform=transform)
                test_dataloader = torch.utils.data.DataLoader(
                    test_set,
                    batch_size=64,
                    num_workers=16,
                    shuffle=False,
                )
                roc_auc, acc= evaluate(model, test_dataloader, device)
                print("Test{} in epoch{}, RocAUC: {:.4f}, Acc: {:.4f}".format(k, epoch, roc_auc,acc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training defect detection as described in the CutPaste Paper.')
    parser.add_argument('--type', default="EyeQ", type=str,
                        help='number of epochs to train the model , (default: 256)')

    parser.add_argument('--epochs', default=201, type=int,
                        help='number of epochs to train the model , (default: 256)')
    
    parser.add_argument('--model_dir', default="models",
                        help='output folder of the models , (default: models)')
    
    parser.add_argument('--no-pretrained', dest='pretrained', default=True, action='store_false',
                        help='use pretrained values to initalize vgg16 , (default: True)')
    
    parser.add_argument('--test_epochs', default=50, type=int,
                        help='interval to calculate the auc during trainig, if -1 do not calculate test scores, (default: 10)')                  

    parser.add_argument('--freeze_vgg', default=1, type=int,
                        help='number of epochs to freeze vgg (default: 20)')
    
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate (default: 1e-3)')

    parser.add_argument('--optim', default="sgd",
                        help='optimizing algorithm values:[sgd, adam] (dafault: "sgd")')

    parser.add_argument('--batch_size', default=64, type=int,
                        help='batch size, real batchsize is depending on cut paste config normal cutaout has effective batchsize of 2x batchsize (dafault: "64")')   

    
    parser.add_argument('--cuda', default=True,
                    help='use cuda for training (default: False)')
    
    parser.add_argument('--workers', default=16, type=int, help="number of workers to use for data loading (default:8)")


    args = parser.parse_args()
    print(args)
    
    
    device = "cuda" if args.cuda else "cpu"
    print(f"using device: {device}")
    
    # create modle dir
    Path(args.model_dir).mkdir(exist_ok=True, parents=True)
    # save config.
    with open(Path(args.model_dir) / "run_config.txt", "w") as f:
        f.write(str(args))
    # train_data_path_all = ["/mnt/huangwk/Dataset/EyeQ/split_EyeQ/train_single_48_95_aug/"]

    # train_data_path_all = ["/mnt/huangwk/Dataset/EyeQ/split_EyeQ/train_multi_2_no_aug/",
    # "/mnt/huangwk/Dataset/EyeQ/split_EyeQ/train_multi_5_no_aug/",
    # "/mnt/huangwk/Dataset/EyeQ/split_EyeQ/train_multi_10_no_aug/"]

    train_data_path_all = ["/mnt/huangwk/Dataset/EyeQ/split_EyeQ/train_single_48_95_aug_blend0.7/",
    "/mnt/huangwk/Dataset/EyeQ/split_EyeQ/train_single_48_95_aug_blend0.6/"]
    # train_data_path_all = ["/mnt/huangwk/Dataset/EyeQ/split_EyeQ/train_single_48_95_aug_randblend2/"]
    for train_data_path in train_data_path_all:
        run_training(data_type=args.type,
                        train_data_path=train_data_path,
                        model_dir=Path(args.model_dir),
                        epochs=args.epochs,
                        pretrained=args.pretrained,
                        test_epochs=args.test_epochs,
                        freeze_vgg=args.freeze_vgg,
                        learninig_rate=args.lr,
                        optim_name=args.optim,
                        batch_size=args.batch_size,
                        device=device,
                        workers=args.workers)
