import torchvision.models as models
from torch import nn
from torch import optim
import train_test

def vgg(img_dataset, train_loader, device_gpu):
    # Initializing the VGG model
    model2 = models.vgg16(pretrained=True)
    
    model2.to(device_gpu)
    
    targets_size = len(img_dataset.class_to_idx)
    
    # Update the last fully connected layer to match the number of target classes
    
    num_features = model2.classifier[6].in_features
    
    model2.classifier[6] = nn.Linear(num_features, targets_size)
    
    criterion = nn.CrossEntropyLoss()
    
    criterion.to(device_gpu)
    
    optimizer = optim.Adam(model2.parameters(), lr = 0.0001)
    
    model2.to(device_gpu)
    
    # Train the model
    
    num_epochs = 5
    acc_val = []
    
    for epoch in range(num_epochs):
        Train_acc_per_batch, accuracy = train_test.train(train_loader, model2, optimizer, criterion,device_gpu)
        acc_val.append(accuracy)
        print("Epoch accuracy: ",accuracy)
    print("Training accuracy of the model is ", accuracy)
    print("Completed")
    return model2



def resnet(img_dataset, train_loader, device_gpu):
    model1 = models.resnet18(pretrained=True)
    
    model1.to(device_gpu)
    
    target_size = len(img_dataset.class_to_idx)
    
    num_ftrs = model1.fc.in_features
    
    model1.fc = nn.Linear(num_ftrs, target_size)
    
    criterion = nn.CrossEntropyLoss()
    
    criterion.to(device_gpu)
    
    optimizer = optim.Adam(model1.parameters(), lr = 0.0001)
    
    model1.to(device_gpu)
    
    # Train the model
    
    num_epochs = 5
    
    acc_val = []
    
    for epoch in range(num_epochs):
        Train_acc_per_batch, accuracy = train_test.train(train_loader,model1, optimizer, criterion,device_gpu)
        acc_val.append(accuracy)
        print("Epoch accuracy: ",accuracy)
    print("Training accuracy is",accuracy)
    print("Completed")
    return model1