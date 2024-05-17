import torch
import json
import os
import torch.nn as nn
from werkzeug.utils import secure_filename
from flask import request
from SLM import SLMOtak
from SLMTrainLoad import SLMTrainLoad, EarlyStopping, DataLoader
from SLMOtak import SLMOtak
from slmAdmin import MyFlaskApp

app = MyFlaskApp(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return 'File uploaded successfully'

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
#batch_size = 32
#learning_rate = 0.001
#num_epochs = 100
#patience = 5

# Load hyperparameters from config.json
with open('config.json', 'r') as f:
    config = json.load(f)

batch_size = config['batch_size']
learning_rate = config['learning_rate']
num_epochs = config['num_epochs']
patience = config['patience']

# Load data
dataloader = DataLoader()
train_loader, val_loader = dataloader(batch_size)

# Initialize model
model = SLMOtak().to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Early stopping
early_stopping = EarlyStopping(patience=patience, verbose=True)

# Training loop
for epoch in range(num_epochs):
    train_loss = SLMTrainLoad(model, train_loader, optimizer, criterion, device)
    val_loss = SLMTrainLoad(model, val_loader, optimizer, criterion, device, train=False)
    
    # Print progress
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Early stopping
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping triggered.")
        break
    
def save_checkpoint(epoch, model, optimizer, train_loss, val_loss):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss
    }
    torch.save(state, f'checkpoint_{epoch}.pth')

def load_checkpoint(model, optimizer, filename='checkpoint.pth'):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch'], checkpoint['train_loss'], checkpoint['val_loss']

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

    # Load best model
    model, optimizer, start_epoch, train_loss, val_loss = load_checkpoint(model, optimizer)

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        train_loss = SLMTrainLoad(model, train_loader, optimizer, criterion, device)
        val_loss = SLMTrainLoad(model, val_loader, optimizer, criterion, device, train=False)
    
        # Print progress
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
        
        # Save checkpoint
        save_checkpoint(epoch, model, optimizer, train_loss, val_loss)
