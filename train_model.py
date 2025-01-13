from tqdm.auto import tqdm
#import time

import torch
from torch import nn, optim
device = "cuda" if torch.cuda.is_available() else "cpu"

from Models.tinyvgg import model
model.to(device)

from dataset import train_data_loader, test_data_loader

def acc_func(real, test):
    return (torch.eq(real, test).sum().item() / len(test)) * 100

loss_func = nn.MSELoss()  # Mean Squared Error loss for regression tasks
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 5

for epoch in range(num_epochs):
    # train it
    model.train()
    current_loss = 0.
    for data_in, data_out in tqdm(train_data_loader):
        data_in.to(device); data_out.to(device)
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
    accuracy = 0.
    with torch.inference_mode():
        for data_in, data_out in tqdm(test_data_loader):
            test_out = model(data_in)
            loss = loss_func(test_out, data_out)
            test_loss += loss.item()
            accuracy += acc_func(data_out, test_out)

    print(f'Finished epoch [{epoch+1}/{num_epochs}], Test Loss: {test_loss/len(test_data_loader)}, Test Acc: {accuracy/len(test_data_loader)}')
