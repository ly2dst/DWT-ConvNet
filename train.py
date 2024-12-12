import torch
import torch.nn as nn
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from WLCNN import Model
import torch.optim as opt
transformer = transforms.Compose([
    transforms.Resize((512, 512)),  
    transforms.ToTensor(),        
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(0.2,0.2,0.2),
    transforms.Grayscale(num_output_channels=1),
    transforms.Normalize(mean=[0.5], std=[0.5]) 
])
train_dir="dataset/train"
test_dir="dataset/test"
Trainset=datasets.ImageFolder(train_dir,transform=transformer)
TraindataLoader=DataLoader(dataset=Trainset,batch_size=64,shuffle=True)

Testset=datasets.ImageFolder(test_dir,transform=transformer)
TestdataLoader=DataLoader(dataset=Testset,batch_size=64,shuffle=True)
print(f'trainset class:{Trainset.classes}')
print(f'length of trainset:{len(Trainset)}')
# model=Model(length=len(Trainset.classes),size=256)

model=Model()

criterion=nn.CrossEntropyLoss()
optim=opt.Adam(model.parameters(),lr=0.001,weight_decay=1e-5)

epochs=2
for epoch in range(0,epochs):
    print(f'{epoch+1} of epoch')
    for images,labels in TraindataLoader:
        optim.zero_grad()
        outputs=model(images)
        loss=criterion(outputs,labels)
        loss=loss
        loss.backward()
        optim.step()                         
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
   
model.eval()  

correct = 0
total = 0
val_loss = 0
TP = 0          
FP = 0          
FN = 0 
criterion = nn.CrossEntropyLoss()  
print('testing')
with torch.no_grad():  
    for images, labels in TestdataLoader:
        outputs = model(images)  
        loss = criterion(outputs, labels)  
        
        val_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)  
    for p, l in zip(predicted.view(-1), labels.view(-1)):  
            if p == l:
                if l == 1:
                    TP += 1
            else:
                if l == 1:
                    FN += 1
                else:
                    FP += 1  
    total += labels.size(0)  
    correct += (predicted == labels).sum().item()  


accuracy = 100 * correct / total
average_loss = val_loss / len(Testset)
recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
 

precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
 

f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

print(f'Accuracy: {accuracy:.2f}%')
print(f'Recall: {recall:.2f}')
print(f'Precision: {precision:.2f}')
print(f'F1 Score: {f1_score:.2f}')




