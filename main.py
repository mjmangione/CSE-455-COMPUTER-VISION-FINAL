'''
CSE 455: Computer Vision
Matthew Mangione
Bird Classifier

This script uses transfer learning to adjust a pretrained convolutional neural
net to classify species of birds founded in the associated dataset.

(the dataset is private & therefore not present in this repository)

Copied from Kaggle notebook.

'''


import numpy as np
import matplotlib.pyplot as plt
import random
import h5py


import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# format & transform bird data
def get_bird_data(n, augmentation=0):
    transform_train = transforms.Compose([
        transforms.Resize(n),
        transforms.RandomCrop(n, padding=8, padding_mode='edge'), # Take n x n crops from padded images
        transforms.RandomHorizontalFlip(),    # 50% of time flip image along y-axis
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(n),
        transforms.CenterCrop(n),
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.ImageFolder(root='/kaggle/input/birds21wi/birds/train', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

    testset = torchvision.datasets.ImageFolder(root='/kaggle/input/birds21wi/birds/test', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=False, num_workers=2)

    classes = open("/kaggle/input/birds21wi/birds/names.txt").read().strip().split("\n")

    # Backward mapping to original class ids (from folder names) and species name (from names.txt)
    class_to_idx = trainset.class_to_idx
    idx_to_class = {int(v): int(k) for k, v in class_to_idx.items()}
    idx_to_name = {k: classes[v] for k,v in idx_to_class.items()}
    return {'train': trainloader, 'test': testloader, 'to_class': idx_to_class, 'to_name':idx_to_name}

# wrapper class to use compressed HDF5 data
class BirdH5Dataset(torch.utils.data.Dataset):
    def __init__(self, in_file):
        self.in_file = in_file

    def __len__(self):
        with h5py.File(self.in_file, 'r') as file:
            length = file['labels'].shape[0]
        return length

    def __getitem__(self, idx):
        with h5py.File(self.in_file, 'r') as file:
            image = file['images'][idx]
            label = file['labels'][idx]
        return (torch.tensor(image), torch.tensor(label))

# converts and saves bird data to lzf compression
def toh5(filename, dataloader):
    #https://stackoverflow.com/questions/47072859/how-to-append-data-to-one-specific-dataset-in-a-hdf5-file-with-h5py
    print('working on', filename)
    with h5py.File(filename, "w") as file:
        for i, (images, labels) in enumerate(dataloader, 0):
            images = images.numpy()
            labels = labels.numpy()
            for lb in range(labels.shape[0]):
                labels[lb] = data['to_class'][labels[lb]]
            if i % 10 == 0:
                print(i)
            if i == 0:
                shape = images.shape
                file.create_dataset("images", np.shape(images), np.float32, data=images, compression="lzf", chunks=True, maxshape=(None,shape[1],shape[2],shape[3]))
                file.create_dataset("labels", np.shape(labels), np.int64, data=labels, compression="lzf", chunks=True, maxshape=(None,))
            else:
                new_size = file["images"].shape[0] + images.shape[0]
                file["images"].resize(new_size, axis = 0)
                file["labels"].resize(new_size, axis = 0)
                file["images"][-images.shape[0]:] = images
                file["labels"][-labels.shape[0]:] = labels
        print(file["images"].shape)
        print(file["labels"].shape)
        im_torch = torch.tensor(file["images"][0])
        lb = file['labels'][0]
        classes = open("/kaggle/input/birds21wi/birds/names.txt").read().strip().split("\n")
        print(classes[lb], lb)


def get_bird_dataset_compressed(data_og):
    trainset = BirdH5Dataset('./birdsbirds21wi_train.h5') # train set fr
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

    testset = BirdH5Dataset('./birdsbirds21wi_test.h5') # actually test set
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

    classes = open("/kaggle/input/birds21wi/birds/names.txt").read().strip().split("\n")
    return {'train' : trainloader, 'test' : testloader, 'classes' : classes}


def train(net, dataloader, epochs=1, start_epoch=0, lr=0.01, momentum=0.9, decay=0.0005,
          verbose=1, print_every=10, state=None, schedule={}, checkpoint_path=None):
    net.to(device)
    net.train()
    losses = []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=decay)
    #optimizer = optim.AdamW(net.parameters(), lr=lr)#, momentum=momentum, weight_decay=decay)


    # Load previous training state
    if state:
        net.load_state_dict(state['net'])
        optimizer.load_state_dict(state['optimizer'])
        start_epoch = state['epoch']
        losses = state['losses']

    # Fast forward lr schedule through already trained epochs
    for epoch in range(start_epoch):
        if epoch in schedule:
            print ("Learning rate: %f"% schedule[epoch])
            for g in optimizer.param_groups:
                g['lr'] = schedule[epoch]

    for epoch in range(start_epoch, epochs):
        sum_loss = 0.0

        # Update learning rate when scheduled
        if epoch in schedule:
            print ("Learning rate: %f"% schedule[epoch])
            for g in optimizer.param_groups:
                g['lr'] = schedule[epoch]

        for i, batch in enumerate(dataloader, 0):
            inputs, labels = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # autograd magic, computes all the partial derivatives
            optimizer.step() # takes a step in gradient direction

            losses.append(loss.item())
            sum_loss += loss.item()

            if i % print_every == print_every-1:    # print every 10 mini-batches
                if verbose:
                  print('[%d, %5d] loss: %.3f' % (epoch, i + 1, sum_loss / print_every))
                sum_loss = 0.0
        if checkpoint_path:
            state = {'epoch': epoch+1, 'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'losses': losses}
            torch.save(state, checkpoint_path + 'checkpoint-%d.pkl'%(epoch+1))
    return losses

def predict(net, dataloader, ofname):
    out = open(ofname, 'w')
    out.write("path,class\n")
    net.to(device)
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (imagesi, labelsi) in enumerate(dataloader, 0):
            if i%100 == 0:
                print(i)
            images, labels = imagesi.to(device), labelsi.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            fname, _ = dataloader.dataset.samples[i]
            #out.write("test/{},{}\n".format(fname.split('/')[-1], data['to_class'][predicted.item()]))
            out.write("test/{},{}\n".format(fname.split('/')[-1], predicted.item()))
    out.close()
    return (correct, total)

#------------------------------------- MAIN ------------------------------------

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# augment data / compress it / set dataloaders to compressed data
data = get_bird_data(224, augmentation=1)
toh5('./birdsbirds21wi_train.h5', data_og['train'])
toh5('./birdsbirds21wi_test.h5', data['test'])
data = get_bird_dataset_compressed(data)

# Download pre-trained net % reset final layer to # of bird species
#resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnext101_32x8d', pretrained=True)
resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnext50_32x4d', pretrained=True)
resnet.fc = nn.Linear(2048, 555)

# train network & save predictions
losses = train(resnet, data['train'], epochs=12, schedule={0:.01, 5:.001, 8:.0005, 10: .0001},print_every=25, checkpoint_path='./')
predict(resnet, data['test'], "preds3.csv")
