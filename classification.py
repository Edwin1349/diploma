import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import seaborn as sns
from os import walk
from os.path import splitext
from os.path import join
import time
import random
from tqdm import tqdm
from classification_model import *
matplotlib.style.use('ggplot')

class ImageDataset(Dataset):
    def __init__(self, images, labels=None, transforms=None):
        self.X = images
        self.y = labels
        self.transforms = transforms

    def __len__(self):
        return (len(self.X))

    def __getitem__(self, i):
        data = self.X[i][:]

        if self.transforms:
            data = self.transforms(data)

        if self.y is not None:
            return (data, self.y[i])
        else:
            return data


def seed_everything(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True


def load(img_dir):
    image_paths = list()
    for root, dirs, files in walk(img_dir):
        for f in files:
            if splitext(f)[1].lower() == ".jpg":
                image_paths.append(join(root, f))
    data = []
    labels = []
    for image_path in image_paths:
        label = image_path.split(os.path.sep)[-2]
        image = cv2.imread(image_path)
        data.append(image)
        labels.append(label)
    data = np.array(data)
    labels = np.array(labels)
    return data, labels


def one_hot(labels):
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    print(f"Total number of classes: {len(lb.classes_)}")
    return labels

def create_confusion_matrix(y_true, y_pred, classes):
    y_true = np.argmax(y_true, axis=1)
    amount_classes = len(classes)

    confusion_matrix = np.zeros((amount_classes, amount_classes))
    print(len(y_true))
    for idx in range(len(y_true)):
        target = y_true[idx]

        output = y_pred[idx]

        confusion_matrix[target][output] += 1

    plt.figure(figsize=(20, 15))

    df_cm = pd.DataFrame(confusion_matrix, index=classes, columns=classes).astype(int)
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d", linewidths=.5)
    plt.show()

    plt.figure(figsize=(10, 15))
    cm = np.array([np.diag(confusion_matrix), (np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix))]).T
    cm = pd.DataFrame(cm, index=classes, columns=['TP', 'FP']).astype(int)
    norm_cm = cm.div(cm.sum(axis=1), axis=0)
    heatmap = sns.heatmap(norm_cm, annot=cm, fmt="d", linewidths=.5)
    print(cm.sum(axis=0, skipna=True))
    plt.show()
    return confusion_matrix

def prepare_data(data, labels, batch_size):
    (X, x_val, Y, y_val) = train_test_split(data, labels,
                                            test_size=0.2,
                                            stratify=labels)

    (x_train, x_test, y_train, y_test) = train_test_split(X, Y,
                                                          test_size=0.25)

    train_data = ImageDataset(x_train, y_train, train_transform)
    val_data = ImageDataset(x_val, y_val, val_transform)
    test_data = ImageDataset(x_test, y_test, val_transform)

    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return trainloader, testloader, valloader


def fit(model, dataloader, criterion):
    print('Training')
    model.train()
    running_loss = 0.0
    running_correct = 0
    for i, data in tqdm(enumerate(dataloader), total=int(len(dataloader.dataset) / dataloader.batch_size)):
        data, target = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, torch.max(target, 1)[1])
        running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        running_correct += (preds == torch.max(target, 1)[1]).sum().item()
        loss.backward()
        optimizer.step()

    loss = running_loss / len(dataloader.dataset)
    accuracy = 100. * running_correct / len(dataloader.dataset)

    print(f"Train Loss: {loss:.4f}, Train Acc: {accuracy:.2f}")

    return loss, accuracy


def validate(model, dataloader, criterion):
    print('Validating')
    model.eval()
    running_loss = 0.0
    running_correct = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(dataloader.dataset) / dataloader.batch_size)):
            data, target = data[0].to(device), data[1].to(device)
            outputs = model(data)
            loss = criterion(outputs, torch.max(target, 1)[1])

            running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            running_correct += (preds == torch.max(target, 1)[1]).sum().item()

        loss = running_loss / len(dataloader.dataset)
        accuracy = 100. * running_correct / len(dataloader.dataset)
        print(f'Val Loss: {loss:.4f}, Val Acc: {accuracy:.2f}')

        return loss, accuracy


def train(model, trainloader, valloader, epochs, criterion):
    train_loss, train_accuracy = [], []
    val_loss, val_accuracy = [], []
    print(f"Training on {len(trainloader.dataset)} examples, validating on {len(valloader.dataset)} examples...")
    start = time.time()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        train_epoch_loss, train_epoch_accuracy = fit(model, trainloader, criterion)
        val_epoch_loss, val_epoch_accuracy = validate(model, valloader, criterion)
        train_loss.append(train_epoch_loss)
        train_accuracy.append(train_epoch_accuracy)
        val_loss.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)

    end = time.time()
    print((end - start) / 60, 'minutes')
    #torch.save(model.state_dict(), f"./tsnet_epochs{epochs}.pth")
    torch.save(model, f"./tsnet_epochs{epochs}.pt")
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_accuracy, color='green', label='train accuracy')
    plt.plot(val_accuracy, color='blue', label='validataion accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('./accuracy.png')
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(val_loss, color='red', label='validataion loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./loss.png')


def test(model, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(dataloader.dataset) / dataloader.batch_size)):
            data, target = data[0].to(device), data[1].to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)

            target_test = (target == 1).nonzero(as_tuple=True)[1].to('cpu').numpy()
            predicted_test = predicted.to('cpu').numpy()

            total += len(predicted_test)
            correct += np.count_nonzero(target_test == predicted_test)

            # for i in range(len(predicted_test)):
            #     img = inv_transform(data[i])
            #     img = img.cpu().numpy()
            #     img = np.moveaxis(img, 0, 2)
            #     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            #     plt.show()
            #     print("TARGET: ", classes[torch.argmax(target[i]).cpu()])
            #     print("RESULT: ", classes[predicted_test[i]])
    return correct, total

#prepare data
seed_everything(47)
epochs = 30
batch_size = 16
criterion = nn.CrossEntropyLoss()

data, labels = load("../Dataset/Classification/GTSRB/Final_Training/Images")
labels = one_hot(labels)
trainloader, testloader, valloader = prepare_data(data, labels, batch_size)

#prepare model
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

model = TrafficSignNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
print(summary(model, (3, 112, 112)))

#train model
train(model, trainloader, valloader, epochs, criterion)
correct, total = test(model, testloader)
print(correct, total, epochs)

#test model
predictions, targets = [], []
for images, labels in testloader:
    logps = model(images.to(device))
    output = torch.exp(logps)
    pred = torch.argmax(output, 1)

    # convert to numpy arrays
    pred = pred.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    for i in range(len(pred)):
        predictions.append(pred[i])
        targets.append(labels[i])
cm = create_confusion_matrix(targets, predictions, classes)

image = val_transform(testloader.dataset.X[0])
outputs = model(image[None, ...].to(device))
_, predicted = torch.max(outputs.data, 1)
print(classes[predicted])
print(testloader.dataset.Y[0])
cv2.imshow("something", testloader.dataset.X[0])
plt.show()

image = val_transform(testloader.dataset.X[47])
outputs = model(image[None, ...].to(device))
_, predicted = torch.max(outputs.data, 1)
print(classes[predicted])
print(testloader.dataset.Y[47])
cv2.imshow("something", testloader.dataset.X[47])
plt.show()



