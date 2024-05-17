import boto3
import torch
import importlib
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import pickle
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
from io import StringIO
from sklearn.model_selection import train_test_split
from SLM import SLMProcessor as model

class SLMTrainLoad(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, data_type):
        super(SLMTrainLoad, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First layer: fully connected (fc)
        self.fc2 = nn.Linear(hidden_size, output_size)  # Second layer: fully connected (fc)

        # Dynamically import the appropriate preprocessing function
        preprocess_module = importlib.import_module(f'{data_type}Preprocess')
        self.preprocess = getattr(preprocess_module, f'preprocess_{data_type}')

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply ReLU activation function after first layer
        x = self.fc2(x)  # Pass through second layer
        return x
    
    def get_target(self, input):
        target = self.training_dataset.loc[input, 'target_column']
        return target

    def load_training_dataset(self):
        # Create a client for the S3 service
        s3 = boto3.client('s3', region_name='us-west-2')

        # Get the object from the S3 bucket
        obj = s3.get_object(Bucket='my-bucket', Key='training_dataset.csv')

        # Preprocess the data
        preprocessed_data = self.preprocess(obj['Body'].read())

        # Use preprocessed data instead of raw data
        self.training_dataset = pd.read_csv(StringIO(preprocessed_data.decode('utf-8')))

        return self.training_dataset

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

class EarlyStopping:
    # EarlyStopping class to monitor validation metric and stop training 
    # when it stops improving for a certain patience window.
    def __init__(self):
        self.early_stopping = EarlyStopping(
            patience=self.config['early_stopping_patience'],
            verbose=self.config['early_stopping_verbose'],
            monitor=self.config['early_stopping_monitor'],
            mode=self.config['early_stopping_mode']
        )        
        self.best_score = None
        self.wait = 0
        self.stopped_epoch = 0
        self.early_stop = False

    def __call__(self, epoch, logs):
        current_score = logs.get(self.monitor)
        if current_score is None:
            # If the monitored metric is not available, skip early stopping check.
            return False

        if self.best_score is None or \
            (self.mode == 'min' and current_score <= self.best_score) or \
            (self.mode == 'max' and current_score >= self.best_score):
            self.best_score = current_score
            self.wait = 0
        else:
            # If current score is not better (depending on mode), increment wait counter.
            self.wait += 1
            if self.wait >= self.patience:
                # If current score is better, update best score and reset wait counter.
                self.stopped_epoch = epoch
                self.early_stop = True
                if self.verbose:
                    print(f'EarlyStopping: Stop training at epoch {self.stopped_epoch}')

                # If wait counter reaches patience, stop training.
                return True
            else:
                return False

# Define vocab
vocab = {"word1": 1, "word2": 2, "word3": 3}  # replace with your actual vocabulary

# Define num_epochs
num_epochs = 100  # replace with the number of epochs you want to train for

# Save model checkpoint
torch.save(model.state_dict(), 'checkpoint.pth')

# Load model checkpoint
model.load_state_dict(torch.load('checkpoint.pth'))

# Save vocab
with open('vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)

# Load vocab
with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

# Create an instance of the model
model = SLMTrainLoad()

# Load the training dataset
model.load_training_dataset()

# Define a DataLoader if necessary
dataloader = DataLoader(TensorDataset, batch_size=32, shuffle=True)

# Define the optimizer and the loss function
optimizer = optim.Adam(model.parameters())
criterion = nn.MSELoss()  # or nn.CrossEntropyLoss()

# Define a learning rate scheduler for learning rate decay
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# Define early stopping
early_stopping = EarlyStopping(patience=10, verbose=True)

# Define the training loop
for epoch in range(num_epochs):
    # Unfreeze the model after 5 epochs
    if epoch == 5:
        model.unfreeze()

    for inputs, targets in dataloader:
        # Forward pass
        outputs = model(inputs)
        
        # Compute loss
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Assuming 'inputs' and 'targets' are your full dataset
train_inputs, validation_inputs, train_targets, validation_targets = train_test_split(
    inputs, targets, test_size=0.2, random_state=42)

# Now you can create your DataLoaders
train_dataset = TensorDataset(train_inputs, train_targets)
validation_dataset = TensorDataset(validation_inputs, validation_targets)

train_loader = DataLoader(train_dataset, batch_size=32)
val_loader = DataLoader(validation_dataset, batch_size=32)
# Print loss for this epoch
print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Update learning rate
scheduler.step()

# Validation
model.eval()  # Set the model to evaluation mode
val_loss = 0
with torch.no_grad():
    for inputs, targets in val_loader:  # Assuming you have a DataLoader for your validation data
        outputs = model(inputs)
        loss = criterion(outputs, targets)  # Assuming you have a loss function defined as 'criterion'
        val_loss += loss.item()

        val_loss /= len(val_loader)  # Average the validation loss

        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

# Load the last checkpoint with the best model
model.load_state_dict(torch.load('checkpoint.pt'))

# Save the model
torch.save(model.state_dict(), 'model.pth')

# Assuming you have some predicted output 'pred' and ground truth 'target'
pred = model(input)
target = model.get_target(input)

# Mean Squared Error loss
mse_loss = F.mse_loss(pred, target)

# Cross Entropy loss
# Note: target for cross_entropy should be class indices and not one-hot vectors
cross_entropy_loss = F.cross_entropy(pred, target)