import torch
from torch import nn
from torch import optim
from tqdm.notebook import tqdm


def train(train_loader, network, optimizer, loss_function,device_gpu):
    device_gpu = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Storing accuracy and loss values of each batch in lists
    loss_val_per_batch = []
    accuracy_val_per_batch = []

    print('Training the model')
    for img, lab in tqdm(train_loader):
        # sending images to GPU
        img, lab = img.to(device_gpu), lab.to(device_gpu)

        # Resetting optimizer gradients to zero
        optimizer.zero_grad()

        # classifying the images based on the network
        classification = network(img)

        # Caluculating loss value and appending it to a list
        loss = loss_function(classification, lab)
        loss_val_per_batch.append(loss.item())

        # Backward pass through the network
        loss.backward()

        # Updating the weights of the network
        optimizer.step()

        # Caluculating accurach for each bactch
        prediction = torch.argmax(classification, dim=1)
        accuracy = (prediction == lab).sum().item() / len(lab)
        accuracy_val_per_batch.append(accuracy)

    # Caluculating the accuracy value for the entire training data
    train_accuracy = sum(accuracy_val_per_batch) / len(accuracy_val_per_batch)
    return accuracy_val_per_batch, train_accuracy



def model_evaluate(validation_loader, model, device_gpu):
    model.eval()
    with torch.no_grad():
        correct_pred = 0
        total_pred = 0
        eval_acc_per_batch= []
        for inputs, lab in validation_loader:
            data, target = inputs.to(device_gpu), lab.to(device_gpu)
            output = model(data)
            x, predicted = torch.max(output.data, 1)
            total_pred += target.size(0)
            correct_pred += (predicted == target).sum().item()
            accuracy = (predicted == target).sum().item() / len(target)
            eval_acc_per_batch.append(accuracy)
            
        print(f"validation accuracy of the model is : {correct_pred/total_pred:.4f}")