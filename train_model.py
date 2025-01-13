from tqdm.auto import tqdm
#import time

import torch
from torch import nn, optim
device = "cuda" if torch.cuda.is_available() else "cpu"

from Models.tinyvgg import model
model.to(device)
from dataset import train_data_loader, test_data_loader, BATCH_SIZE

## metric
from torchmetrics import IoU
insec_func = IoU(num_classes=4)

loss_func = nn.MSELoss()  # Mean Squared Error loss for regression tasks
optimizer = optim.Adam(model.parameters(), lr=0.0001)
num_epochs = 30
SAVE_PATH = 'Models/tinyvgg_state_dict.pkl'

for epoch in range(num_epochs):
    # train it
    model.train()
    current_loss = 0.
    for data_in, data_out in tqdm(train_data_loader):
        data_in, data_out = data_in.to(device), data_out.to(device)
        optimizer.zero_grad()                   # clear grads
        current_out = model(data_in)            # pass input data
        loss = loss_func(current_out, data_out) # calc loss
        loss.backward()             # calc grads wrt loss & params
        optimizer.step()            # update params
        current_loss += loss.item() # accumulate loss

    print(f'Train Loss/batch: {current_loss/len(train_data_loader)}')
    print(f'Train Loss/image: {current_loss/len(train_data_loader)/4}')

    # test it
    model.eval()
    test_loss = 0.
    intersection = 0.
    with torch.inference_mode():
        for data_in, data_out in tqdm(test_data_loader):
            data_in, data_out = data_in.to(device), data_out.to(device)
            test_out = model(data_in)
            loss = loss_func(test_out, data_out)
            test_loss += loss.item()
            intersection += insec_func(data_out, test_out)

    print(f'Test Loss/batch: {test_loss/len(test_data_loader)}, IoU/batch: {accuracy/len(test_data_loader)}')
    print(f'Test Loss/image: {test_loss/len(test_data_loader)/BATCH_SIZE}, IoU/image: {accuracy/len(test_data_loader)/BATCH_SIZE}')
    print(f"----------- Finished epoch [{epoch+1}/{num_epochs}]")


# save it
print(f"Saved model to: {SAVE_PATH}")
torch.save(obj=model.state_dict(), f=SAVE_PATH)
