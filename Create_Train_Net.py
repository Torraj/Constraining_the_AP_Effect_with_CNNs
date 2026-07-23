import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import date
from datetime import datetime
import math
from torch.utils.data import DataLoader

field_size = None
fields_num = None
test_frac = None
batch = None

def initialize(field_size_dum, fields_num_dum, test_frac_dum, batch_dum):
    global field_size, fields_num, test_frac, batch
    field_size = field_size_dum
    fields_num = fields_num_dum
    test_frac = test_frac_dum
    batch = batch_dum
    print("ctn initialized!")

def choose_model_architecture(choice, field_size):
    """ returns initialized CNN with the chosen architecture (choices ar 'OG' or 'Simon')
    """
    if choice == 'OG':
        # Original neural network architecture that Alex used for his thesis work
        class Net(nn.Module):
            
            def __init__(self):
                super().__init__()
                # 1 input channel, 6 output channels, and kernel of size 5x5x5 (cube since kernel size given by 1 num)
                self.conv11 = nn.Conv3d(1, 6, 5, padding = 2)
                self.conv12 = nn.Conv3d(6, 16, 5, padding = 2)
                self.conv13 = nn.Conv3d(16, 16, 5,  padding = 2)
        
                # "MaxPool3d" defines a 3d kernel of size 2x2x2 that will generate another array from our input array with 
                # values dictated by the max value of the input array within the region of the array covered by the kernel. 
                # A new array is created of 1 less unit height, width, and depth than our input array unless padding is used
                self.pool = nn.MaxPool3d(2, 2)
    
                # The input to this layer has 16 channels. Each channel has a 3D galaxy field whose size has been halved
                # two times with pooling layers. Thus the total number of values across all 16 channels will be 16 times
                # the number of values in each 3D array
                self.fc1 = nn.Linear(int(16 * (field_size/(2**2))**3) , 1000)
                self.fc2 = nn.Linear(1000, 200)
                self.fc3 = nn.Linear(200, 1)
        
            #Forward describes how the network passes information from the input forward through the network
            def forward(self, x):
                # Specify pass through three convolutional layers each with a ReLU activation function
                x = F.relu(self.conv11(x))
                # These final two convolutional layers are each followed by a max-pooling layer
                x = self.pool(F.relu(self.conv12(x)))
                x = self.pool(F.relu(self.conv13(x)))
        
                # The '1' parameter in the torch.flatten function basically tells to leave the network to leave the 
                # outermost distinction in our data intact. Here, that is the difference between different thrimages 
                # in our batch, so each thrimage is flattened but not mixed together.
                x = torch.flatten(x, 1)
        
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x
    
    if choice == 'Simon':
        depth = 4                            # Depth of fully connected network
        depth_to_width = 4/115               # Optimal depth:width ratio for my network according to Simon 
        width = int(depth / depth_to_width)  # Depth divided by optimal ratio
        kernel_size = 5
        
        class Net(nn.Module):
            
            def __init__(self):
                super().__init__()
                self.conv11 = nn.Conv3d(1, 6, 5, padding = 2)
                self.conv12 = nn.Conv3d(6, 16, 5, padding = 2)
                self.conv13 = nn.Conv3d(16, 16, 5,  padding = 2)
        
                self.pool = nn.MaxPool3d(2, 2)
        
                self.fc_in = nn.Linear(int(16 * (field_size/(2**2))**3) , width)
                self.fc_hid = nn.Linear(width, width)
                self.fc_out = nn.Linear(width, 1)
        
                
                # Draw initial weights and biases from specified probability distributions for each layer
                c_w = 2  # std for weights that will be modified for each layer according to number of nodes in previous layer
                c_b = 0  # std for biases that will be modified for each layer according to number of nodes in previous layer. 
                         # 0 implies a constant distribution (delta function)
        
                # Initial weights/biases for convolutional layers (can also do "kaiming_normal")
                nn.init.kaiming_uniform_(self.conv11.weight, mode='fan_in', nonlinearity='relu')
                if self.conv11.bias is not None:
                    nn.init.constant_(self.conv11.bias, 0)
                    
                nn.init.kaiming_uniform_(self.conv12.weight, mode='fan_in', nonlinearity='relu')
                if self.conv12.bias is not None:
                    nn.init.constant_(self.conv12.bias, 0)
                    
                nn.init.kaiming_uniform_(self.conv13.weight, mode='fan_in', nonlinearity='relu')
                if self.conv13.bias is not None:
                    nn.init.constant_(self.conv13.bias, 0)
        
                # Initial weights/biases for fully connected layers. Note how the standard deviation for each
                # distribution is divided by the number of nodes in the previous layer of the fully connected
                # network
                nn.init.normal_(self.fc_in.weight, mean=0.0, std= np.sqrt(c_w / (16*(field_size/(2**2))**3)))
                nn.init.constant_(self.fc_in.bias, 0)
        
                nn.init.normal_(self.fc_hid.weight, mean=0.0, std= np.sqrt(c_w/width))
                nn.init.constant_(self.fc_in.bias, 0)
                
                nn.init.normal_(self.fc_out.weight, mean=0.0, std= np.sqrt(c_w/width))
                nn.init.constant_(self.fc_in.bias, 0)
        
            def forward(self, x):
                x = F.relu(self.conv11(x))
                x = self.pool(F.relu(self.conv12(x)))
                x = self.pool(F.relu(self.conv13(x)))
        
                x = torch.flatten(x, 1)
        
                if depth == 3:
                    x = F.relu(self.fc_in(x))
                    x = F.relu(self.fc_hid(x))
                    x = self.fc_out(x)
                    
                elif depth == 4:
                    x = F.relu(self.fc_in(x))
                    x = F.relu(self.fc_hid(x))
                    x = F.relu(self.fc_hid(x))
                    x = self.fc_out(x)
                    
                return x
    
    # Set up the network to be able to be run across multiple GPUs
    net = nn.DataParallel(Net())
    
    return net

def choose_optimizer(choice, net, lr):
    """ returns chosen optimizer for CNN (choices are choices are 'SGD', 'Adam', 'SGD_opt_lr', 'Adam_opt_lr').
        'SGD' is the pytorch standard gradient descent optimizer. 'Adam' is the pytorch standard adaptive Adam
        optimizer. 'SGD_opt_lr' is 'SGD' with the learning rates for each weight and bias in the CNN specified
        to be their optimal value (according to Simon) so that learning occurs in a stable manner. 'Adam_opt_lr'
        is once again 'Adam' with learning rates for all CNN weights/biases set to their optimal values

        Input args:
            choice: as definef above
            net: network for which optimizer is being used during training
            lr: learning rate for the optimizer
    """
    '''
    Definition of 'momentum' for the SGD optimizer:
    
    Neural networks work by adjusting weights and biases in a direction of parameter space that lowers the 
    network's loss. As a hyperparameter, momentum describes how much previous parameter gradients contribute
    to the current direction of parameter adjustments. As in, weights and biases are adjusted not just 
    considering the current gradient of the loss in parameter space but also previous gradients. A larger
    value for the momentum hyperparameter means that previous gradients are more important.
    '''
    
    if choice == 'SGD':
        # Optimizer defined to be standard gradient descent (SGD) with the same learning rate applied to
        # ALL network parameters (weights and biases in every layer)
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        
    elif choice == 'Adam':
        # Optimizer defined to be the adpative optimizer Adam with the same learning rate applied to
        # ALL network parameters
        optimizer = optim.Adam(net.parameters(), lr = lr)
        
    elif choice == 'SGD_opt_lr':
        optimizer = optim.SGD([
            # We divide the learning rate for the weights by the number of output values for a layer plus 
            # the number of layers of that type. For convolutional layers, this means dividing by the 
            # number of output channels times the cubic kernel size plus 3. We divide the learning rate
            # for biases just by the depth of the fully connected or convolutional part of the network
            {'params': net.module.conv11.weight, 'lr': lr*lam_w / (6*kernel_size**3 + 3)},
            {'params': net.module.conv11.bias, 'lr': lr*lam_b/3},
        
            {'params': net.module.conv12.weight, 'lr': lr*lam_w / (16*kernel_size**3 + 3)},
            {'params': net.module.conv12.bias, 'lr': lr*lam_b/3},
        
            {'params': net.module.conv13.weight, 'lr': lr*lam_w / (16*kernel_size**3 + 3)},
            {'params': net.module.conv13.bias, 'lr': lr*lam_b/3},
            
            {'params': net.module.fc_in.weight, 'lr': lr*lam_w / (depth+width)},
            {'params': net.module.fc_in.bias, 'lr': lr*lam_b/depth},
            
            {'params': net.module.fc_hid.weight, 'lr': lr*lam_w / (depth+width)},
            {'params': net.module.fc_hid.bias, 'lr': lr*lam_b/depth},
            
            {'params': net.module.fc_out.weight, 'lr': lr*lam_w / (depth+width)},
            {'params': net.module.fc_out.bias, 'lr': lr*lam_b/depth}
        ], lr=lr, momentum=0.9)
        
    elif choice == 'Adam_opt_lr':
        optimizer = optim.Adam([
            {'params': net.module.conv11.weight, 'lr': lr*lam_w / (6*kernel_size**3 + 3)},
            {'params': net.module.conv11.bias, 'lr': lr*lam_b/3},
        
            {'params': net.module.conv12.weight, 'lr': lr*lam_w / (16*kernel_size**3 + 3)},
            {'params': net.module.conv12.bias, 'lr': lr*lam_b/3},
        
            {'params': net.module.conv13.weight, 'lr': lr*lam_w / (16*kernel_size**3 + 3)},
            {'params': net.module.conv13.bias, 'lr': lr*lam_b/3},
            
            {'params': net.module.fc_in.weight, 'lr': lr*lam_w / (depth+width)},
            {'params': net.module.fc_in.bias, 'lr': lr*lam_b/depth},
            
            {'params': net.module.fc_hid.weight, 'lr': lr*lam_w / (depth+width)},
            {'params': net.module.fc_hid.bias, 'lr': lr*lam_b/depth},
            
            {'params': net.module.fc_out.weight, 'lr': lr*lam_w / (depth+width)},
            {'params': net.module.fc_out.bias, 'lr': lr*lam_b/depth}
        ], lr=lr)
    
    else:
        print("Invalid optimizer specified. Please choose either Adam, SGD, Adam_opt_lr, or SGD_opt_lr\
        as an optimizer.")

    return optimizer

def train_network(train_dataloader, test_dataloader, epoch_num, device, net, optimizer, criterion):
    """ Starts training the network.

        Input args:
            train_dataloader: dataloader object such that the network can access the training data
            test_dataloader:  dataloader object such that the network can access the validation dataset
            epoch_num:        number of epochs to train network
            device:           device on which the network is being trained (CPU or GPU) which is specified in notebook
            net:              the network being trained
            optimizer:        choice of optimizer when training
            criterion:        loss function with which we are training the neural network    
    """
    today = date.today()
    
    # Initialize arrays that we will use to make loss plots later
    epochs = np.zeros(epoch_num)
    train_loss = np.zeros(epoch_num)
    val_loss = np.zeros(epoch_num)
    
    # We will use 'best_loss' so that we can save a copy of the network with the lowest loss
    best_loss = np.inf
    
    s = datetime.now()
    
    # Loop over the dataset an epoch_num of times
    for epoch in range(epoch_num):
        train_running_loss = 0.0
        test_running_loss = 0.0
        test_running_loss_avg = 0.0
        # Enumerate assigns each entry in train_dataloader, a single batch, to the variable 'data' 
        # 'i' is an index each batch in train_dataloader
        
        for i, data in enumerate(train_dataloader):
            # 'inputs' is a list of all the thrimages in the batch and 'labels' is a list of all 
            # the stretch factors in the batch
            inputs = data[0].to(device)
            labels = data[1].to(device)
            
            #zero the parameter gradients between batches ran through the network so that they don't blow up
            optimizer.zero_grad()
            
            ##This block contains our forward + backward + optimize steps##
            
            # Convert the thrimage tensors into the float32 tensor type
            inputs = inputs.float()
            
            # Run thrimages through network and get the network's guess for the stretch factor for each 
            # thrimage in the batch
            outputs = net(inputs)
            
            # "Squeeze" out any single value arrays (get rid of unneeded brackets)
            outputs = outputs.squeeze()
            
            # Ensure output is a float32 value
            outputs = outputs.float()
            
            # Ensure labels are float32
            labels = labels.float()
            
            # Compute loss, just the square of the difference between the actual stretch factor and what 
            # the network guess is
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            train_running_loss += loss.item()
            '''
            This segment of code effectively checks if we have ran through the entire image dataset 
            during training. It does this by checking if the number of batches loaded is equal to the 
            number of batches the dataset can be divided into rounded up. It then prints the ith iteration 
            through the dataset that we are currently on and the average loss per image for that iteration
            through the dataset
            '''
            if i + 1 == math.ceil((1-test_frac)*fields_num/batch):
                # Averages running_loss over all the batches in one iteration of the dataset
                train_avg_loss = np.sqrt(train_running_loss/(i+1))
                print(f'[{epoch + 1}] loss: {train_avg_loss:.6f}')
                train_loss[epoch] = train_avg_loss
                
        # This is where we determine the accuracy of the network on the validation dataset We save a 
        # copy of the neural network with the lowest loss 
        for j, data in enumerate(test_dataloader):
            inputs = data[0].to(device)
            labels = data[1].to(device)
            inputs = inputs.float()
            outputs = net(inputs)
            outputs = outputs.squeeze()
            outputs = outputs.float()
            labels = labels.float()
            loss = criterion(outputs, labels)
            test_running_loss += loss.item()
            
            if j + 1 == math.ceil(test_frac*fields_num/batch):
                test_avg_loss = test_running_loss/(j+1)
                print("Average deviation of network guess from actual stretch is approximately: " + str("{0:.3f}".format(np.sqrt(test_avg_loss))))
                epochs[epoch] = epoch
                val_loss[epoch] = np.sqrt(test_avg_loss)
    
                if test_avg_loss < best_loss:
                    best_loss = test_avg_loss
                    model_path = f'nnet_Simon_Sancho_{today}.pth'
                    # Uncomment to save the network
                    torch.save(net.state_dict(), model_path)
                    
    
    print('Finished Training')
    e = datetime.now()
    
    elapsed = (e - s).total_seconds() 
    print("Elapsed:", elapsed, "s")
    return 0
        