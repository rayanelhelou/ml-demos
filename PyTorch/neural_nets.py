# Copyright 2020, Rayan El Helou.
# All rights reserved.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from IPython import display
from tqdm import trange
import sys


def convert_torch_to_numpy(arg):
    if isinstance(arg, torch.Tensor) or isinstance(arg, torch.nn.parameter.Parameter):
        new_arg = arg.cpu().detach().numpy()
    else:
        new_arg = arg
    return new_arg


def torch_plot(*args, **kwargs):
    '''
    Replaces plt.plot when specifically plotting data of type tensor that may
    be stored on GPU or CPU, and you don't want to manually type:
            plt.plot(x.cpu().detach().numpy())
    Rather:
            torch_plot(x)

    Accepts any list of arguments, just like plt.plot    

    '''
    args = list(map(convert_torch_to_numpy, args))
    plt.plot(*args, **kwargs)


# More general than torch_plot not just replaces plt.plot, but also replaces plt.(anything)
def torch_plt(f_str, *args, **kwargs):
    '''
    Replaces plt, and accepts any set of arguments
    Instead of plt.some_plt_function(x), use: torch_plt('some_plt_function', x)

    Example:
    torch_plt('imshow', net.net[2].weight)

    '''
    args = list(map(convert_torch_to_numpy, args))
    getattr(plt, f_str)(*args, **kwargs)


def UpdateDisplay():
    '''
    Use in notebooks, not in standalone .py files.
    
    Call this function in a loop at the end of each iteration
    after plot(s) have been generated, to clear any existing
    plots and replace with the plots from the next iteration.
    This effectively enables a plot animation within a cell.

    EXAMPLE:
    
        for i in range(N):
            x = some_long_process(i)
            plt.plot(x)
            UpdateDisplay()

    '''
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.pause(0.01)


def plot_SISO(x, y, net):
    '''
    Plots single-input single-output actual data (x,y)
    along with net-based data (x,yhat).
    
    x: torch.tensor of size (batch_size, 1)
    y: torch.tensor of size (batch_size, 1)
    net: type nn.Module which maps 1D to 1D

    '''
    # Extend plot range
    # Previously:     |-------o-------|
    # Now:        |-----------o-----------|
    x_min, x_max = x.min(), x.max()
    x_center = (x_min + x_max) / 2
    x_min = x_center + 1.5 * (x_min - x_center)
    x_max = x_center + 1.5 * (x_max - x_center)
    
    # Plot (x,y) pairs
    gray = 0.7
    torch_plot(x, y, 'o', color=[gray, gray, gray])
    
    # Plot (x,yhat) pairs
    x_extended = torch.linspace(x_min, x_max, 1000).reshape(-1, 1)
    yhat = net(x_extended)
    torch_plot(x_extended, yhat, color='maroon', lw=3)
    
    plt.xlabel('x')
    plt.ylabel('y')


def plot_DISO(x, y, net, threshold=0.1):
    '''
    Plots double-input single-output actual data (x,y)
    along with net-based data (x,yhat).
    
    x: torch.tensor of size (batch_size, 2)
    y: torch.tensor of size (batch_size, 1)
    net: type nn.Module which maps 2D to 1D
    
    ---------------------------------------------------
    NOTE: This function is poorly implemented. There
          must be a neater way to do this, but it works
          for the purposes of our demo.
    ---------------------------------------------------

    '''
    # Locate indices of positive and negative labels
    idx_p = torch.where(y == +1)[0]
    idx_n = torch.where(y == -1)[0]

    # determine the 2D tile size (dX, dY) based on resolution
    resolution = 300 # number of pixels along each of
                     # horizontal & vertical directions (e.g. 300x300)
    dX, dY = (x.max(dim=0).values - x.min(dim=0).values) / resolution
    dX = float(dX)
    dY = float(dY)

    # generate 2D grid for the x & y bounds
    Y, X = np.mgrid[slice(float(torch.min(x[:,1])), float(torch.max(x[:,1])) + dY, dY),
                    slice(float(torch.min(x[:,0])), float(torch.max(x[:,0])) + dX, dX)]

    device = x.device
    N = Y.shape[0]*Y.shape[1]
    Y, X = torch.from_numpy(Y).float().to(device), \
                torch.from_numpy(X).float().to(device)
    Z = net(torch.cat((X.reshape((N,1)), Y.reshape((N,1))), 1))

    idx_zero = torch.where(torch.abs(Z) < threshold)[0]
    Z[idx_zero,:] = 0.0

    Z = Z.reshape_as(X)
    Z.shape

    Z = Z[:-1, :-1]
    Zmin, Zmax = -torch.abs(Z).max(), torch.abs(Z).max()

    top = cm.get_cmap('Reds', 128)
    bottom = cm.get_cmap('Greens', 128)
    newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                                                bottom(np.linspace(0, 1, 128))))
    newcmp = ListedColormap(newcolors)

    torch_plot(x[idx_p,0], x[idx_p,1], 'gs', alpha=0.2)
    torch_plot(x[idx_n,0], x[idx_n,1], 'rs', alpha=0.2)

    torch_plt('pcolor', X, Y, Z, cmap=newcmp, vmin=Zmin, vmax=Zmax,
                                    alpha=0.5, lw=0, antialiased=True)
    plt.axis('tight')


class BuildSequential:
    '''
    Builds a trainable fully connected neural network, with
    the following attributes:
    
        net:        nn.Sequential
        n:          list of number of neurons per layer
                    (input, hidden_1, ..., hidden_n, output)
        activation: nn.Module (activation function class)
    
    EXAMPLE:
    
        net = BuildSequential(1, 16, 1, act=nn.ReLU())
        # Features:
                - 1 input feature
                - 1 hidden layer with 16 neurons
                - 1 output feature
                - ReLU activation between layers (not at output)

    Methods available: (For details about each, see its own docstring)
    
        self.train(x, y, **kwargs)
            Updates self.net parameters to fit (x,y) training data
            
                x: input  data as torch.tensor of size (batch_size, self.n[0])
                y: output data as torch.tensor of size (batch_size, self.n[-1])
                
        self.draw(fig)
            Visualizes self.net's neurons and their connection to each other.
            Each connection is assigned a thickness according to its magnitude
            to signify which neurons are more relevant than others.
            Created by me (Rayan) for the purposes of a specific demo.
            
                fig: figure, or subplot, where the neural network is drawn
                
        self.plot_hidden_output(tol=0.1)
            Only works for single-input-single-output (1D to 1D nets)
            Plots the output of each hidden neuron with one subplot
            per layer to show how the final output has been created by
            the previous hidden layers.
            
                tol: tolerance level to ignore certain neurons

    '''
    def __init__(self, *n, activation=nn.Tanh()):
        
        layers = []
        for i in range(len(n)-2):
            layers.append(nn.Linear(n[i],n[i+1]))
            layers.append(activation)
        layers.append(nn.Linear(n[-2],n[-1])) # last layer has no activation function
        
        self.net = nn.Sequential(*layers)
        self.n = n
        self.activation = activation

        # Initialize record of losses
        self.epoch_loss = []

    def __repr__(self):
        
        str_act = str(self.activation)
        if 'torch.nn.modules.activation' in str(type(self.activation)):
            str_act = 'nn.' + str_act
            
        return f'{self.__class__.__name__}({self.n}, activation={str_act}) \n\n'

    def print_memory_size(self):
        '''
        Prints amount of memory consumed by the network's
        weights, in either KB or MB.
        
        '''
        memory = 0
        number_weights = 0
        for layer in self.net:
            if isinstance(layer, torch.nn.modules.linear.Linear):
                # memory in bytes
                memory += sys.getsizeof(layer.weight.storage())
                # number of weights
                number_weights += layer.weight.numel()
        KB = memory / 1024
        if KB > 1024:
            print(KB / 1024, 'MB and', number_weights, 'weights')
        else:
            print(KB, 'KB and', number_weights, 'weights')

    def train(self, x, y, criterion=nn.MSELoss(), optim_name='Adam',
              lr=0.01, weight_penalty=0.0, frac_data=1.0, N_epoch=1000, plot_during=False):
        '''
        Updates self.net parameters to fit (x,y) training data

        Feel free to edit all of this function to suit your needs!
        
        REQUIRED INPUTS:
        
            x: input  data as torch.tensor of size (batch_size, self.n[0])
            y: output data as torch.tensor of size (batch_size, self.n[-1])

        OPTIONAL INPUTS:
        
            criterion:      loss function to compare actual and estimated output.
                            For more values, other than default, refer to:
                            https://pytorch.org/docs/stable/nn.html#loss-functions

            optim_name:     name of optimizer used to update neural network parameters.
                            For more values, other than default, refer to:
                            https://pytorch.org/docs/stable/optim.html

            lr:             learning rate for optimizer

            weight_penalty: when value is positive, a penalty is added to the loss
                            to discourage non-zero neural network weights

            frac_data:      specifies fraction of the data to be used per epoch.

            N_epoch:        total number of epochs.

            plot_during:    If set to True, plots are displayed live to indicate
                            the training progress, but it takes more time.
                            If set to False, a tqdm progress bar is shown (which
                            provides an estimate of time remaining to finish training).
        
        '''
        if x.shape[-1] != self.n[0] or y.shape[-1] != self.n[-1]:
            assert False, "Network architecture doesn't match number of features in data"
        
        try:
            optimizer = getattr(optim, optim_name)(self.net.parameters(), lr=lr)
        except:
            assert False, "unknown value assigned to keyword optim_name, try 'Adam' or 'SGD'" + \
                            " ... for more optimizers: https://pytorch.org/docs/stable/optim.html"

        # Batch size
        batch_size = x.shape[0]
        assert batch_size == y.shape[0], "x & y don't share the same batch size: " + \
                                        f"x: {x.shape[0]}, y: {y.shape[0]}"

        # Specify range of epochs
        N_epoch += 1 # to end at N_epoch, not N_epoch-1
        if plot_during:
            range_epoch = range(N_epoch)
        else:
            range_epoch = trange(N_epoch) # tqdm progress bar

        # Select batch length 'L' based on 'frac_data'
        assert frac_data > 0.0 or frac_data <= 1.0, "frac_data should range in (0, 1]"
        L = int(1 + frac_data*(batch_size - 1))

        # Train
        tic = time.time()
        for epoch in range_epoch: # loop over the dataset multiple times

            # Pick random part of the dataset for each epoch (length 'L')
            perm = torch.randperm(x.size(0))
            idx = perm[:L]
            x_idx = x[idx] if len(x.shape) == 1 else x[idx,:]
            y_idx = y[idx] if len(y.shape) == 1 else y[idx,:]

            # Update network parameters by looping number of times
            # inversely proporinal to 'frac_data'. That is, the more
            # sub-batches there are, the more times we loop, to cover
            # more of the data.
            training_loss = 0.0
            for _ in range(int(batch_size/L)):
                # forward pass
                yhat = self.net(x_idx)
                loss = criterion(yhat, y_idx)
                
                # penalize weights #### OPTIONAL
                if weight_penalty > 0:
                    for layer in self.net:
                        if isinstance(layer, torch.nn.modules.linear.Linear):
                            # L1-based (Mean Absolute) penalty
                            loss += weight_penalty * layer.weight.abs().mean()

                # PyTorch magic!
                optimizer.zero_grad()   # Initialize gradients
                loss.backward()         # Backpropagation
                optimizer.step()        # Update weights & biases

                # Update loss record
                training_loss += loss.item()
            
            self.epoch_loss.append(training_loss)

            # To plot or not to plot
            if plot_during:
                N_plots = 10
                plot_now = epoch % int(N_epoch/N_plots) == 0
            else:
                # Only plot in the end
                plot_now = epoch == N_epoch-1

            # Plot progress
            if plot_now:
                print('\n Plotting ...') # In case code below took too long to run
                plt.figure(figsize=(20,5))
                
                plt.subplot(1,3,1)
                plt.semilogy(self.epoch_loss)
                plt.title('Loss')
                plt.xlabel('Epoch')

                plt.subplot(1, 3, 2)
                if   self.n[0] == 1 and self.n[-1] == 1: # Plot single-input, single-output only
                    plot_SISO(x, y, self.net)
                elif self.n[0] == 2 and self.n[-1] == 1: # Plot double-input, single-output only
                    plot_DISO(x, y, self.net)
                else:
                    plt.text(0.5, 0.5, 'Plotting for other than\n' + \
                                     '1D/2D input, 1D output\n' + \
                                     'not programmed here', fontsize=16, \
                                     horizontalalignment='center', \
                                     verticalalignment='center')
                    plt.axis('off')
                
                h = plt.subplot(1,3,3)
                self.draw(h)
                
                time_spent = np.round(time.time() - tic, 2)
                plt.suptitle('Epoch ' + str(epoch) + ' of ' + str(N_epoch-1) +
                             ',     time spent = ' + str(time_spent) + ' s\n',fontsize=20)
                
                # This was designed for notebooks, not for standalone .py files
                UpdateDisplay()

    def draw(self, fig=None):
        '''
        Sketch a neural network on top of figure or subplot (specified by 'fig')

        (SUB-OPTIMAL IMPLEMENTATION)
        ---------------------------------------------------------------------------
        By trial and error, I created the code below, and I agree it's not self-
        explanatory. If you have trouble understanding it, feel free to contact me.

        This was created as a proof of concept, and is not meant to be extended to
        all kinds of neural nets.

        NOTE:   takes time to plot larger nets (e.g. 512 neurons per layer)
        ---------------------------------------------------------------------------

        '''
        t = np.linspace(0, 2*np.pi, 100)
        sx = 1
        sy = 4/np.max(np.array(self.n)) if np.max(np.array(self.n)) > 1 else 1
        R = 0.2*sy
        dxa = 1.5/4*R
        x = (R - dxa/2)*np.cos(t) - dxa/2
        y = (R - dxa/2)*np.sin(t)
        xa = np.array([-dxa/2 + np.sqrt(R*(R - dxa)), R, R, -dxa/2 + np.sqrt(R*(R - dxa))])
        ya = dxa/2*np.array([1, 1, -1, -1])
        cymax = 0

        # Create or re-use existing figure
        # and flag 'plt_show' for later
        # (whether to plt.show() or not)
        if fig is None:
            plt.figure()
            plt_show = True
        else:
            plt_show = False
            fig
            
        # Plot network
        lw = 0.5 # line width
        for i in range(len(self.n)): # i^th layer
                mid_i = (self.n[i] + 1)/2
                cx = i - (len(self.n) - 1)/2
                cx = cx*sx
                wx = cx + np.array([-R, -sx+R])

                for j in range(self.n[i]):

                        # Plot neurons
                        cy = j+1 - mid_i
                        cy = cy*sy
                        if i == 0: # input layer
                                plt.plot(x + dxa + cx, y + cy, 'k', lw=lw)
                        elif i+1 == len(self.n): # output layer
                                plt.plot(x + cx, y + cy, 'k', lw=lw)
                        else:
                                plt.plot(x + cx, y + cy, 'k', lw=lw)
                                plt.plot(xa + cx, ya + cy, 'k', lw=lw)

                        # Plot weights
                        if i>0:
                                W = self.net[2*(i-1)].weight
                                W = torch.abs(W)
                                W = W/torch.max(W)
                                for k in range(self.n[i-1]):
                                    alpha = float(W[j, k])
                                    cy_prev = k+1 - mid_i_prev
                                    cy_prev = cy_prev*sy
                                    wy = np.array([cy, cy_prev])
                                    plt.plot(wx, wy, 'k', lw=2*alpha, alpha=alpha)
                                    
                        if cy > cymax:
                                cymax = cy

                mid_i_prev = mid_i

        plt.axis('equal')
        plt.axis('off')
        if plt_show:
            plt.show()

    def plot_hidden_output(self, x, tol=0.1):
        '''
        Creates a horizontal series of plots which show the output of each 'significant'
        neuron for each layer. A hidden layer neuron is considered significant if its
        output signal has a difference between max & min greater than user-defined 'tol'.

        '''
        assert self.n[0] == 1 and self.n[-1] == 1, 'Did not implement for non single input single output nets'
        
        minmax = lambda i, j: torch.abs(torch.max(self.net[0:i+1](x)[:,j]) - \
                                        torch.min(self.net[0:i+1](x)[:,j]))

        n_hidden = len(self.n) - 2

        if n_hidden > 0:
            plt.figure(figsize=(min(5*n_hidden,20),5))
            h = 0
            for i, layer in enumerate(self.net):
                # only show output of activation functions
                if not isinstance(layer, torch.nn.modules.linear.Linear):
                    h += 1
                    plt.subplot(1, n_hidden, h)
                    for j in range(self.n[h]):
                        if minmax(i, j) > tol:
                            y = self.net[0:i+1](x)[:, j]
                            # Scale it by the weight of the link that's pulling it strongest
                            y *= self.net[i+1].weight.abs().max(dim=0).values[j]
                            torch_plot(x, y)
                    plt.title(f'Hidden Layer {h}')
                    plt.axis('off')
        else:
            print('There are no hidden layers.')
