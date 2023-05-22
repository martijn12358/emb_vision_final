import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Define the neural network architecture
class HandGestureClassifier(nn.Module):
    def __init__(self):
        super(HandGestureClassifier, self).__init__()
        self.fc1 = nn.Linear(63, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 8)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

# Define a custom dataset to load the hand gesture data from the CSV file
class HandGestureDataset(Dataset):
    def __init__(self, csv_file):
        self.data = []
        with open(csv_file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the header row
            for row in reader:
                landmark_data = [float(x) for x in row[:-1]]
                label = int(row[-1])
                self.data.append((landmark_data, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        landmark_data, label = self.data[idx]
        landmark_data = torch.tensor(landmark_data, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return landmark_data, label

# Initialize the dataset and data loader
dataset = HandGestureDataset('hand_gestures.csv')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize the neural network and loss function
net = HandGestureClassifier()
criterion = nn.CrossEntropyLoss()

# Initialize the optimizer
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Train the neural network
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

print('Finished training')

# Save the trained model
torch.save(net.state_dict(), 'hand_gesture_classifier.pth')
