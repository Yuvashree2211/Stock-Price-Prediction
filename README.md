# Stock-Price-Prediction


## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement:
Predicting stock prices is a complex task due to market volatility. Using historical closing prices, we aim to develop a Recurrent Neural Network (RNN) model that can analyze time-series data and generate future price predictions.

## Dataset:
The dataset consists of historical stock closing prices from trainset.csv and testset.csv. The data is normalized using MinMax scaling, and sequences of 60 past values are used as input features. The model learns patterns from training data to predict upcoming prices, helping traders and investors make informed decisions.

## Design Steps

### Step 1:
Data Collection & Preprocessing: Load historical stock prices, normalize using MinMaxScaler, and create sequences for time-series input.

### Step 2:
Model Design: Build an RNN with two layers, define input/output sizes, and set activation functions.

### Step 3:
Training Process: Train the model using MSE loss and Adam optimizer for 20 epochs with batch-size optimization.

### Step 4:
Evaluation & Prediction: Test on unseen data, inverse transform predictions, and compare with actual prices.

### Step 5:
Visualization & Interpretation: Plot training loss and predictions to analyze performance and potential improvements.



## Program
#### Name: YUVASHREE R
#### Register Number: 212224040378

```Python 
# Define RNN Model
class RNNModel(nn.Module):
    def __init__(self,input_size=1,hidden_size=64,num_layers=2,output_size=1):
      super(RNNModel,self).__init__()
      self.rnn=nn.RNN(input_size,hidden_size,num_layers,batch_first=True)
      self.fc=nn.Linear(hidden_size,output_size)

    def forward(self, x):
      out, _ = self.rnn(x)
      out = self.fc(out[:, -1, :])
      return out


model = RNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# Train the Model

epochs = 20
model.train()
train_losses = []
for epoch in range(epochs):
  epoch_loss = 0
  for x_batch, y_batch in train_loader:
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    optimizer.zero_grad()
    outputs = model(x_batch)
    loss = criterion(outputs, y_batch)
    loss.backward()
    optimizer.step()
    epoch_loss += loss.item()
  train_losses.append(epoch_loss / len(train_loader))
  print(f"Epoch [{epoch+1}/{epochs}], Loss:{train_losses[-1]:.4f}")


```


### Datasets
# Training: 
<img width="594" height="708" alt="image" src="https://github.com/user-attachments/assets/435321c2-addf-449e-82ee-575e0cf8ef3c" />

# Testing: 

<img width="593" height="697" alt="image" src="https://github.com/user-attachments/assets/bb03b776-d7a4-46b0-8f6b-7fa3fd5df9f3" />


## Output
### True Stock Price, Predicted Stock Price vs time

<img width="859" height="582" alt="image" src="https://github.com/user-attachments/assets/bad298a0-7ddd-4f15-8c52-63750eb150fd" />




### Predictions 

<img width="223" height="42" alt="image" src="https://github.com/user-attachments/assets/48c0d669-9a99-4012-958e-aa1eff83da5b" />


## Result

Thus, a Recurrent Neural Network model for stock price prediction has successfully been devoloped successfully.
