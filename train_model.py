from tqdm.auto import tqdm
#import time

import torch
from torch import nn, optim

from Models.tinyvgg import model
model.cpu()

from dataset import train_data_loader, test_data_loader

loss_func = nn.MSELoss()  # Mean Squared Error loss for regression tasks
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 5

for epoch in range(num_epochs):
    # train it
    model.train()
    current_loss = 0.
    for data_in, data_out in tqdm(train_data_loader):
        data_in.cpu(); data_out.cpu()
        optimizer.zero_grad()                   # clear grads
        current_out = model(data_in)            # pass input data
        loss = loss_func(current_out, data_out) # calc loss
        loss.backward()             # calc grads wrt loss & params
        optimizer.step()            # update params
        current_loss += loss.item() # accumulate loss

    print(f'Training Loss: {current_loss/len(train_data_loader)}')

    # test it
    model.eval()
    test_loss = 0.
    with torch.inference_mode():
        for data_in, data_out in tqdm(test_data_loader):
            test_out = model(data_in)
            loss = loss_func(test_out, data_out)
            test_loss += loss.item()
    
    print(f'Finished epoch [{epoch+1}/{num_epochs}], Test Loss: {test_loss/len(test_data_loader)}')
