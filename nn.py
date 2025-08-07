import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report



try:
    X = np.load('data/X_processed.npy')  
    y = np.load('data/y_processed.npy')  
    print("dataset was loaded")
except Exception as e:
    print("failed to load data", e)


#split the data into training and testing data 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)


#using pytorch tensors 
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=64)


#the nueral network uses binary classification: attack or no attack 
class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),    #the number of hidden units is usually a power of 2 
            nn.ReLU(),                   #activation function used for hidden layers 
            nn.Dropout(0.3),             #used to randomly ignore data, prevetns overfitting 

            nn.Linear(64, 32),           #inpt to second layer is 64 units, outputs 32 units 
            nn.ReLU(),                  
            nn.Dropout(0.3),     

            nn.Linear(32, 1),            #output layer takes in 32 units, ouptuts 1 unit 
            nn.Sigmoid()                 #sigmoid activiation function ins used for the output/last layer
        )

    def forward(self, x):
        return self.net(x)

model = BinaryClassifier(input_dim=12)

#training the neural network 
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(20):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_X).squeeze()
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"epoch {epoch+1}/20 loss: {total_loss:.4f}")



#disable gradient calculation with no gradient to speed up the evaluation
#get the models predicted values for each batch 
#sort the predcitions by 0.5 
#then uses the classification report to calucate the precision score, and recall and F1 score

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        preds = model(batch_X).squeeze()
        preds = (preds >= 0.5).int()
        all_preds.extend(preds.numpy())
        all_labels.extend(batch_y.numpy())

print(classification_report(all_labels, all_preds, target_names=["Normal", "Attack"]))


#add a confusion matrix to visualize the precision and recall of the model
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Attack"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
