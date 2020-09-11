#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import time
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


#https://github.com/facebookarchive/fb.resnet.torch/issues/180#issuecomment-433419706

train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)

# use np.concatenate to stick all the images together to form a 1600000 X 32 X 3 array
x = np.concatenate([np.asarray(train_data[i][0]) for i in range(len(train_data))])
# print(x)
print(x.shape)
# calculate the mean and std along the (0, 1) axes
mean = np.mean(x, axis=(0, 1))/255
std = np.std(x, axis=(0, 1))/255
# the the mean and std
print(mean, std)

# In[3]:


np.random.seed = 0
indices = np.arange(0,50000)
np.random.shuffle(indices) # shuffle the indicies

num_worker = 4

train_transform = transforms.Compose(
    [
     transforms.Resize(256),
     transforms.RandomHorizontalFlip(p=0.5),
     transforms.RandomRotation(45),
     transforms.ColorJitter(contrast=0.2,brightness=0.2,hue=.2, saturation=.2),
     transforms.RandomCrop(224),
     transforms.ToTensor(),
     transforms.Normalize(mean=mean, std=std)]

train_sampler=torch.utils.data.SubsetRandomSampler(indices[:45000])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform
                                        )

train_loader = torch.utils.data.DataLoader(train_set, batch_size=32,sampler=train_sampler,
                                          shuffle=False, num_workers=num_worker)


                                                                                            
val_transform = transforms.Compose(
    [transforms.Resize(224),
     transforms.ToTensor(),
     transforms.Normalize(mean=mean, std=std)]

val_sampler=torch.utils.data.SubsetRandomSampler(indices[-5000:])

val_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                      download=True, transform=val_transform
                                      )

val_loader = torch.utils.data.DataLoader(val_set, batch_size=32,sampler=val_sampler,
                                          shuffle=False, num_workers=num_worker)


# In[4]:


test_transform = transforms.Compose(
    [transforms.Resize(224),
     transforms.ToTensor(),
     transforms.Normalize(mean=mean, std=std)]


testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# In[5]:


# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

print(images.shape)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(8)))


# In[6]:

# The below MobileNet source code is from https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v2.py

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            Swish()
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self,
                 num_classes=10,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None,
                 norm_layer=None):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)

net = MobileNetV2()
print(net)


# In[7]:


step_size = 4
learning_rate = 0.1
momentum = 0.9

device = torch.device('cuda')
#net = model
if torch.cuda.device_count() > 1:
      print("Let's use", torch.cuda.device_count(), "GPUs!")
      # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
      net = nn.DataParallel(net)

net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.75, last_epoch=-1) 


# In[8]:


train_loss_history = []
val_loss_history = []
train_acc_history = []
val_acc_history = []
history = []
train_total = 0
train_correct = 0
val_total = 0
val_correct = 0


# In[9]:


for epoch in range(160):  # loop over the dataset multiple times
    
    training_loss = 0.0
    valid_loss = 0.0
    start_time = time.time()
    net.train()
    for data, target in train_loader:
        # get the inputs; data is a list of [inputs, labels]
        #inputs, labels = data
        
        inputs, labels = data.to(device), target.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        train_loss = criterion(outputs, labels)
        train_loss.backward()
        optimizer.step()

        # print statistics
        training_loss += train_loss.item()*data.size(0)
        #print(training_loss)
        
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        
    net.eval()
    for data, target in val_loader:
        inputs, labels = data.to(device), target.to(device)
        
        outputs = net(inputs)
        val_loss = criterion(outputs, labels)
        
        valid_loss += val_loss.item()*data.size(0)
        #print(valid_loss)
        
        _, predicted = torch.max(outputs.data, 1)
        val_total += labels.size(0)
        val_correct += (predicted == labels).sum().item()
        
    end_time = time.time()
    
    train_acc = train_correct/train_total
    val_acc = val_correct/val_total
    
    train_loss = training_loss / 45000
    val_loss = valid_loss/5000
    print(f'epoch:{epoch + 1}, training loss:{train_loss} ,val loss:{val_loss}, time:{end_time - start_time}')
    print(f'train acc:{train_acc}, val_acc:{val_acc}')
    
    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)
    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)

print('Finished Training')


# In[10]:


history = []
history.append(train_loss_history)
history.append(val_loss_history)
history.append(train_acc_history)
history.append(val_acc_history)


# In[11]:


def plot_history(history):
    # Plot the results (shifting validation curves appropriately)
    plt.figure(figsize=(8,5))
    n = len(history[0])
    plt.plot(np.arange(0,n),history[0], color='orange')
    plt.plot(np.arange(0,n),history[1],'b')
    plt.plot(np.arange(0,n)+0.5,history[2],'r')  # offset both validation curves
    plt.plot(np.arange(0,n)+0.5,history[3],'g')
    plt.legend(['Train Loss','Val Loss','Train Acc','Val Acc'])
    #plt.legend(['Train Loss','Val Loss'])
    
    plt.grid(True)
    plt.gca().set_ylim(0, 1) # set the vertical range to [0-1] 
    plt.show() 
    
plot_history(history)


# In[12]:


net.eval()


# In[13]:


dataiter = iter(testloader)
images, labels = dataiter.next()

images.to(device)
labels.to(device)
# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(16)))


# In[14]:


outputs = net(images.to(device))


# In[15]:


_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(16)))


# In[19]:


correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)

        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %f ' % (
    100 * correct / total))


# In[17]:


class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)

        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(16):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

total_acc = 0
for i in range(10):
    class_acc = class_correct[i] / class_total[i]
    total_acc += class_acc
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_acc ))
avg_acc = total_acc/10
print(avg_acc)


# In[18]:


PATH = './MoblieNet_v2_no_pretrain-M0.pth'
torch.save(net.state_dict(), PATH)



