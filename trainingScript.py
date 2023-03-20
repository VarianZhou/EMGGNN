import dgl
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from EMG_Model import EMG_Classifier, EarlyStopper
from dgl.dataloading import GraphDataLoader
import matplotlib.pyplot as plt
from dgl import save_graphs
from sklearn.metrics import confusion_matrix
from numpy import save


from sklearn.metrics import accuracy_score, recall_score, precision_score

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:516"

'''Define the Classifier that we shall use'''

BATCH_SIZE = 32
NUM_OF_CLASSES = 65
MAX_NUM_OF_EPOCH = 80
PATIENCE = 5
EARLY_STOPPER_PATIENCH = 10
hidden_dimension = 109

# Only these two variables need to be adjusted when model is changed
plot_saving_address = 'correlation_model_plots'
model_directory = 'correlation_models'
evaluation_directory = 'correlation'

# plot_saving_address = 'combine_model_plots'
# model_directory = 'combind_models'


# We define an early stopper to assist regularizing the model.
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

##############################################################################################
##############################################################################################
##############################################################################################

# We use this function to save the training data, validation data, and test data respectively, so that we can
# repeat the experiment result quickly.
def save_data(training_data, validation_data, test_data, model_name):

    # Assuming your DataLoader object is named "dataloader"
    print('Saving the dataset used')
    torch.save(training_data, f'data/{evaluation_directory}/training/{model_name}.pt',_use_new_zipfile_serialization = False)
    torch.save(validation_data, f'data/{evaluation_directory}/validation/{model_name}.pt',_use_new_zipfile_serialization = False)
    torch.save(test_data, f'data/{evaluation_directory}/test/{model_name}.pt',_use_new_zipfile_serialization = False)
    print('Saved successfully!')


def evaluate(model, dataloader,device):
    num_correct = 0
    num_tests = 0
    for batched_graph, labels in dataloader:
        pred = model(batched_graph.to(device), batched_graph.ndata["signal_window"].to(device).float())
        num_correct += (pred.argmax(1) == labels.to(device)).sum().item()
        num_tests += len(labels)
    return num_correct / num_tests

# This function is used to measure the performance of the model on different classes, it shall also construct a confusion matrix.
def evaluate_the_performance_of_each_task(model,model_name, dataloader, device):
    # Initialize the lists to store the true labels and predicted labels
    true_labels = []
    pred_labels = []

    # Loop through the data in the dataloader
    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)
            labels = labels.to(device)
            # Pass the data through the model
            outputs = model(data, data.ndata["signal_window"].float())

            # Get the predicted labels from the output
            _, pred = torch.max(outputs, dim=1)

            # Convert the labels and predictions to numpy arrays
            labels = labels.detach().cpu().numpy()
            pred = pred.detach().cpu().numpy()

            # Append the true labels and predicted labels to the lists
            true_labels.extend(labels)
            pred_labels.extend(pred)

    # Convert the lists of true labels and predicted labels to numpy arrays
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    # Build the confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    np.save(f'evaluations/{evaluation_directory}/{model_name}confusion_matrix', cm)

def plot_the_accuracy_graph(train,test,name):
    plt.plot(train,label='training accuracy')
    plt.plot(test,label='test accuracy')
    plt.xlabel = 'epoch'
    plt.ylabel = 'accuracy'
    plt.legend()
    # Saving the figure.
    plt.savefig(f"plots/{plot_saving_address}/accuracy_trend_diagram_{name}.jpg")
    plt.show()
    plt.cla()
    plt.clf()

def plot_the_accuracy_chart_for_each_class(correct_list,total_list):
    accuracy = [0]*NUM_OF_CLASSES
    for i in range(NUM_OF_CLASSES):
        accuracy[i] = correct_list[i]/total_list[i]
        print(f'The accuracy of the model on classifying class{i} is {accuracy[i]}')


'''We use this function to train a model on a specific dataset, the name of the model should be provided'''
# Provide the name of the model
def trainOnTheModel(dataset,model_name):
    print('We are now training the data')
    # We monitor and record the training accuracy and validate accuracy, these will be used to plot the diagrams.
    training_accuracy=[]
    test_accuracy=[]

    # Randomly split the dataset into training,validate, and testing datasets
    train_dataset,valid_dataset,test_dataset = dgl.data.split_dataset(dataset,[0.6,0.2,0.2],True)
    # Load these datasets
    train_dataloader = GraphDataLoader(
        train_dataset, batch_size=BATCH_SIZE, drop_last=True
    )
    validate_dataloader = GraphDataLoader(
        valid_dataset, batch_size=BATCH_SIZE, drop_last=True
    )
    test_dataloader = GraphDataLoader(
        test_dataset, batch_size=BATCH_SIZE, drop_last=True
    )
    save_data(train_dataloader,validate_dataloader,test_dataloader,model_name)
##############################################################################################
##############################################################################################
##############################################################################################


    '''
    Define the training model and optimizers to use
    '''
    # define the device to use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # initialize model and scheduler
    model = EMG_Classifier(65, hidden_dimension, 66)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # We use the ReduceLROnPlateau method
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True,patience=PATIENCE)
    # Add an earlyStopping regularization to preclude overfitting and reduce training time
    early_stopper = EarlyStopper(patience=EARLY_STOPPER_PATIENCH)

    ##############################################################################################
    ##############################################################################################
    ##############################################################################################

    '''
    Train the model
    '''
    for epoch in range(MAX_NUM_OF_EPOCH):
        totalLoss = 0
        train_num_correct = 0
        for batched_graph, labels in train_dataloader:
            model.train()
            optimizer.zero_grad()
            pred = model(batched_graph.to(device), batched_graph.ndata["signal_window"].to(device).float())
            train_num_correct += (pred.argmax(1) == labels.to(device)).sum().item()
            loss = F.cross_entropy(pred, labels.to(device))
            loss.backward()
            optimizer.step()
            totalLoss += loss.item()
        model.eval()
        validLoss = 0
        valid_num_correct = 0

        for valid_graph, valid_labels in validate_dataloader:
            pred = model(valid_graph.to(device), valid_graph.ndata['signal_window'].to(device).float())
            valid_num_correct += (pred.argmax(1) == valid_labels.to(device)).sum().item()
            loss = F.cross_entropy(pred, valid_labels.to(device))
            validLoss += loss.item()
        scheduler.step(validLoss)

        validate_loss = validLoss / len(validate_dataloader)
        train_loss = totalLoss / len(train_dataloader)
        train_accuracy = train_num_correct / (len(train_dataloader)*BATCH_SIZE)
        valid_accuracy = valid_num_correct / (len(validate_dataloader)*BATCH_SIZE)
        test_accuracy.append(valid_accuracy)
        training_accuracy.append(train_accuracy)

        print(f'Epoch {epoch + 1} of {model_name} \t\t Training Loss: {train_loss} \t\t Validation Loss: {validate_loss} \t\t Training Accuracy: {train_accuracy} \t\t Validation Accuracy: {valid_accuracy}')

        if early_stopper.early_stop(validLoss):
            break

    ##############################################################################################
    ##############################################################################################
    ##############################################################################################
    '''
    In the End, we evaluate the model
    '''

    model.eval()
    print(f"Training accuracy for {model_name}:{evaluate(model, train_dataloader, device)}")
    print(f"Test accuracy for {model_name}: {evaluate(model, test_dataloader, device)}")
    filepath = f'models/{model_directory}/{model_name}_model'
    torch.save(model.state_dict(), filepath)
    plot_the_accuracy_graph(training_accuracy,test_accuracy,model_name)
    evaluate_the_performance_of_each_task(model,model_name,test_dataloader,device)
