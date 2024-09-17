#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import torch
import pandas as pd
import torch.nn as nn
import seaborn as sns
import torch.optim as optim
import torch.nn.init as init
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns


# In[23]:



import pandas as pd
file_path = r"F:\deep learning\asignment1\diabetes.csv"
data_new= pd.read_csv(file_path)
data= data.dropna()
data.info()


# ### Data Visualization

# In[24]:


missing_values = data.isnull().sum()
print("Missing values:\n", missing_values)


# In[25]:



print("\nStatistical summary:\n", data.describe())


# In[26]:



correlation_matrix = data.corr()


plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.show()


# In[27]:



diabetic = data[data['Outcome'] == 1]
non_diabetic = data[data['Outcome'] == -1]
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
plt.figure(figsize=(14, 10))
for i, feature in enumerate(features):
    plt.subplot(3, 3, i+1)
    sns.histplot(diabetic[feature], color='orange', label='Diabetic', kde=True, stat="density", common_norm=False)
    sns.histplot(non_diabetic[feature], color='blue', label='Non-diabetic', kde=True, stat="density", common_norm=False)
    plt.title(feature)
    plt.legend()

plt.tight_layout()
plt.show()


# # Multi-layer Perceptron

# In[ ]:





# In[ ]:





# In[28]:



X = data.drop("Outcome",axis=1).values
y = data_new["Outcome"].values
y = np.where(y == -1, 0, y)
samples = X.shape[0]
train_proportion = 0.7   
val_proportion = 0.15   
train_size = int(train_proportion * samples)
val_size = int(val_proportion * samples)
test_size = samples - train_size - val_size
X_train = X[:train_size]
y_train = y[:train_size]
X_val = X[train_size:train_size + val_size]
y_val = y[train_size:train_size + val_size]
X_test = X[train_size + val_size:]
y_test = y[train_size + val_size:]
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)


# In[29]:



device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('device:', device)


# In[ ]:





# In[30]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

features = data.drop('Outcome', axis=1)
target = data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=16)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
perceptron_model = Perceptron()
perceptron_model.fit(X_train_scaled, y_train)
y_pred = perceptron_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")


# In[ ]:





# In[31]:


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
         
        self.layers = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim), 
         )
        
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight) 
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.layers(x)
    

torch.manual_seed(16)

input_dimension = X_train.shape[1]  
hidden_dimension = 35  
output_dimension = 2  
learning_rate = 0.01  
mlp_model = MLP(input_dimension, hidden_dimension, output_dimension)  
loss_criterion = nn.CrossEntropyLoss()  
weight_decay_coefficient = 0.004  
optimizer = optim.SGD(mlp_model.parameters(), lr=learning_rate, weight_decay=weight_decay_coefficient)

training_accuracies, validation_accuracies, training_losses = [], [], []


for epoch in range(100):  
    print(f'Starting epoch {epoch + 1}')

    
    outputs = mlp_model(X_train_tensor)
    loss = loss_criterion(outputs, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    training_losses.append(loss.item())
    with torch.no_grad():
        _, train_predicted = torch.max(outputs, 1)
        train_accuracy = (train_predicted == y_train_tensor).float().mean()
        training_accuracies.append(train_accuracy)
        val_outputs = mlp_model(X_val_tensor)
        _, val_predicted = torch.max(val_outputs, 1)
        val_accuracy = (val_predicted == y_val_tensor).float().mean()
        validation_accuracies.append(val_accuracy)
    print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}')
print('Training process has finished.')
best_validation_accuracy = max(validation_accuracies)
print(f"The best validation accuracy: {best_validation_accuracy:.4f}")


# In[32]:



epochs = range(1, len(training_accuracies) + 1)
plt.figure(figsize=(10, 6))
plt.plot(epochs, training_accuracies, label='Train Accuracy', marker='o', linestyle='-')
plt.plot(epochs, validation_accuracies, label='Validation Accuracy', marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()


# In[33]:


epochs = range(1, len(training_losses) + 1)
plt.figure(figsize=(10, 6))
plt.plot(epochs, training_losses, label='Training Loss', marker='o', linestyle='-')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.grid(True)
plt.show()


# ### Three-Layer Perceptron

# In[34]:


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(MLP, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),  
            nn.ReLU(), 
            nn.Linear(hidden_dim1, hidden_dim2), 
            nn.ReLU(), 
            nn.Linear(hidden_dim2, output_dim), 
        )
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight) 
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.layers(x)
    
torch.manual_seed(16)
input_dimension = X_train.shape[1] 
hidden_dimension1 = 50  
hidden_dimension2 = 35  
output_dimension = 2 
learning_rate = 0.01 
mlp_model = MLP(input_dimension, hidden_dimension1, hidden_dimension2, output_dimension)
loss_criterion = nn.CrossEntropyLoss()
weight_decay_coefficient = 0.004
optimizer = optim.SGD(mlp_model.parameters(), lr=learning_rate, weight_decay=weight_decay_coefficient)
training_accuracies, validation_accuracies, training_losses = [], [], []
for epoch in range(100):
    print(f'Starting epoch {epoch + 1}')

    outputs = mlp_model(X_train_tensor)
    loss = loss_criterion(outputs, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    training_losses.append(loss.item())
    with torch.no_grad():
        _, train_predicted = torch.max(outputs, 1)
        train_accuracy = (train_predicted == y_train_tensor).float().mean()
        training_accuracies.append(train_accuracy)
        val_outputs = mlp_model(X_val_tensor)
        _, val_predicted = torch.max(val_outputs, 1)
        val_accuracy = (val_predicted == y_val_tensor).float().mean()
        validation_accuracies.append(val_accuracy)

    print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}')

print('Training process has finished.')
best_validation_accuracy = max(validation_accuracies)
print(f"The best validation accuracy: {best_validation_accuracy:.4f}")


# ### dropout

# In[35]:



torch.manual_seed(16)

class MLPWithDropout(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob=0.5):
        super(MLPWithDropout, self).__init__()
       
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            nn.ReLU(),  
            nn.Dropout(p=dropout_prob), 
            nn.Linear(hidden_dim, hidden_dim), 
        )
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.layers(x)
input_dimension = X_train.shape[1] 
hidden_dimension = 35 
output_dimension = 2 
learning_rate = 0.01  
dropout_probability = 0.5 

mlp_model_with_dropout = MLPWithDropout(input_dimension, hidden_dimension, output_dimension, dropout_prob=dropout_probability)
loss_criterion = nn.CrossEntropyLoss()
weight_decay_coefficient = 0.004 
optimizer = optim.SGD(mlp_model_with_dropout.parameters(), lr=learning_rate, weight_decay=weight_decay_coefficient)

training_accuracies, validation_accuracies, training_losses = [], [], []

for epoch in range(100):
    print(f'Starting epoch {epoch + 1}')


    outputs = mlp_model_with_dropout(X_train_tensor)
    loss = loss_criterion(outputs, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    training_losses.append(loss.item())
    with torch.no_grad():
        _, train_predicted = torch.max(outputs, 1)
        train_accuracy = (train_predicted == y_train_tensor).float().mean()
        training_accuracies.append(train_accuracy)
        val_outputs = mlp_model_with_dropout(X_val_tensor)
        _, val_predicted = torch.max(val_outputs, 1)
        val_accuracy = (val_predicted == y_val_tensor).float().mean()
        validation_accuracies.append(val_accuracy)

    print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}')

print('Training process has finished.')
best_validation_accuracy = max(validation_accuracies)
print(f"The best validation accuracy: {best_validation_accuracy:.4f}")


# In[36]:


epochs = range(1, len(training_accuracies) + 1)
plt.figure(figsize=(10, 6))
plt.plot(epochs, training_accuracies, label='Train Accuracy', marker='o', linestyle='-')
plt.plot(epochs, validation_accuracies, label='Validation Accuracy', marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()


# In[37]:


epochs = range(1, len(training_losses) + 1)
plt.figure(figsize=(10, 6))
plt.plot(epochs, training_losses, label='Training Loss', marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:





# In[ ]:




