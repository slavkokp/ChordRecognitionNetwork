import json
import torch
from torch import nn
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from model import Network
from custom_image_dataset import CustomImageDataset
from datetime import datetime
from torchsampler import ImbalancedDatasetSampler
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from statistics import mean 

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(device)


# to get average class report for kfold
def average_report(report_dicts):
    reports_count = len(report_dicts)
    stats = ["precision", "recall", "f1-score", "support"]
    classes_count = 19
    average_dict = dict((str(i), dict((stat, 0) for stat in stats)) for i in range(classes_count))
    average_dict["accuracy"] = 0
    average_dict["macro avg"] = dict((stat, 0) for stat in stats)
    average_dict["weighted avg"] = dict((stat, 0) for stat in stats)
    for i in range(classes_count):
        for stat in stats:
            not_zero_reports_count = reports_count
            for j in range(reports_count):
                if report_dicts[j].get(str(i), {"support": 0})["support"] == 0:
                    not_zero_reports_count -= 1
                average_dict[str(i)][stat] += report_dicts[j].get(str(i), {stat: 0})[stat]
            average_dict[str(i)][stat] /= not_zero_reports_count if not_zero_reports_count != 0 and stat != "support" else reports_count
    for j in range(reports_count):
        average_dict["accuracy"] += report_dicts[j]["accuracy"]
    average_dict["accuracy"] /= reports_count
    for measure in ["macro avg", "weighted avg"]:
        for stat in stats:
            for j in range(reports_count):
                average_dict[measure][stat] += report_dicts[j][measure][stat]
            average_dict[measure][stat] /= reports_count
    return average_dict

# avg class report dict to string
def report_to_string(avg_dict):
    headers = ["precision", "recall", "f1-score", "support"]
    width = len("weighted avg")
    digits = 2
    head_fmt = "{:>{width}s} " + " {:>9}" * len(headers)
    result = head_fmt.format("", *headers, width=width)
    result += "\n\n"
    row_fmt = "{:>{width}s} " + " {:>9.{digits}f}" * 3 + " {:>9}\n"
    for i in avg_dict.keys():
        if len(i) < 3:
            result += row_fmt.format(i, avg_dict[i]["precision"], avg_dict[i]["recall"], avg_dict[i]["f1-score"], int(avg_dict[i]["support"]), width=width, digits=digits)
    result += "\n"
    row_fmt_accuracy = (
        "{:>{width}s} "
        + " {:>9.{digits}}" * 2
        + " {:>9.{digits}f}"
        + " {:>9}\n"
    )
    result += row_fmt_accuracy.format("accuracy", "", "", avg_dict["accuracy"], int(avg_dict["macro avg"]["support"]), width=width, digits=digits)
    headings = ["macro avg", "weighted avg"]
    for heading in headings:
        result += row_fmt.format(heading, avg_dict[heading]["precision"], avg_dict[heading]["recall"], avg_dict[heading]["f1-score"], int(avg_dict[heading]["support"]), width=width, digits=digits)
    return result

# create dict with class reports and accuracies for each string seperately 
def save_metrics(metrics_dict, class_report, test_loss, correct, type_str):
    loss_str = ""
    class_report_str = ""
    for i in range(6):
        loss_str = loss_str + f"String {i}: {type_str} Error: \n Accuracy: {(100*correct[i]):>0.1f}%, Avg loss: {test_loss[i]:>8f} \n"
        class_report_str = class_report_str + f"\nString {i}\n" + report_to_string(class_report[i])
    epoch = {
        "report" : class_report_str,
        "loss" : loss_str
        }
    metrics_dict["epochs"].append(epoch)

def get_class_report_string(class_reports):
    class_report_str = ""
    for i in range(6):
        class_report_str = class_report_str + f"\nString {i}\n" + report_to_string(class_reports[i])
    return class_report_str

def train_loop(dataloader, model, loss_fn, optimizer, train_epoch_stats, metrics_dict):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    print(f"Number of batches for train: {num_batches}")

    model.train()
    train_loss, correct = [0 for _ in range(6)], [0 for _ in range(6)]
    global_pred = [[], [], [], [], [], []]
    global_exp = [[], [], [], [], [], []]
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)

        # calcuate losses
        losses = [None for _ in range(6)]
        for i in range(6):
            losses[i] = loss_fn(pred[i], y[:, i])
            train_loss[i] += losses[i].item()
            prd = pred[i].argmax(1)
            exp = y[:, i].argmax(1)
            correct[i] += (prd == exp).type(torch.float).sum().item()
            global_pred[i] = global_pred[i] + prd.tolist()
            global_exp[i] = global_exp[i] + exp.tolist()

        total_loss = sum(losses)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    class_report = [None] * 6
    for i in range(6):
        class_report[i] = classification_report(global_exp[i], global_pred[i], zero_division=0, output_dict=True)
        train_loss[i] /= num_batches
        correct[i] /= size
    save_metrics(metrics_dict, class_report, train_loss, correct, "Train")
    train_epoch_stats[0] = train_loss
    train_epoch_stats[1] = class_report



def test_loop(dataloader, model, loss_fn, test_epoch_stats, metrics_dict):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = [0 for _ in range(6)], [0 for _ in range(6)]
    print(f"Number of batches for test: {num_batches}")
    global_pred = [[], [], [], [], [], []]
    global_exp = [[], [], [], [], [], []]
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            for i in range(6):
                test_loss[i] += loss_fn(pred[i], y[:, i]).item()
                prd = pred[i].argmax(1)
                exp = y[:, i].argmax(1)
                correct[i] += (prd == exp).type(torch.float).sum().item()
                global_pred[i] = global_pred[i] + prd.tolist()
                global_exp[i] = global_exp[i] + exp.tolist()
    
    class_report = [None] * 6
    for i in range(6):
        class_report[i] = classification_report(global_exp[i], global_pred[i], zero_division=0, output_dict=True)
        test_loss[i] /= num_batches
        correct[i] /= size
    save_metrics(metrics_dict, class_report, test_loss, correct, "Test")
    test_epoch_stats[0] = test_loss
    test_epoch_stats[1] = class_report


# main with kfold without saving nn, only for performance validation
if __name__ == "__main__":
    print("Executing validation main")
    print("Loading data")
    # path to dataset file created with PrepareDataset.py
    data = CustomImageDataset("global_res_hope.json", labels=False, drop_half_empty=False)
    print("Data loaded")
    model_name = "resnet_3layers19classes_cross"
    loss_fn = nn.CrossEntropyLoss()

    seed = 42
    torch.manual_seed(seed)
    learning_rate = 1e-3  # 0.1 * batch_size / 256 for sgd
    folds = 3
    epochs = 7
    kfold = KFold(n_splits=folds, shuffle=True)
    global_train_stats = [None] * folds
    global_test_stats = [None] * folds
    for fold, (train_ids, test_ids) in enumerate(kfold.split(data)):
        train_dataloader = DataLoader(data, batch_size=512, sampler=SubsetRandomSampler(train_ids))
        test_dataloader = DataLoader(data, batch_size=512, sampler=SubsetRandomSampler(test_ids))
        net = Network().to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        train_epoch_stats = [None] * 2
        test_epoch_stats = [None] * 2
        test_metrics_dict = {"epochs": []}
        train_metrics_dict = {"epochs": []}
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            print(f"Started training at: {datetime.now()}\n")
            train_loop(train_dataloader, net, loss_fn, optimizer, train_epoch_stats, train_metrics_dict)
            print(f"Started testing at: {datetime.now()}\n")
            test_loop(test_dataloader, net, loss_fn, test_epoch_stats, test_metrics_dict)
        print("Done!")
        global_train_stats[fold] = train_epoch_stats
        global_test_stats[fold] = test_epoch_stats
    
    avg_stats = {"test_loss": None, "train_loss": None, "report_test": None, "report_train": None}
    avg_stats["report_train"] = [average_report([global_train_stats[m][1][i] for m in range(folds)]) for i in range(6)]
    avg_stats["report_test"] = [average_report([global_test_stats[m][1][i] for m in range(folds)]) for i in range(6)]
    avg_stats["train_loss"] = [mean([global_train_stats[m][0][i] for m in range(folds)]) for i in range(6)]
    avg_stats["test_loss"] = [mean([global_test_stats[m][0][i] for m in range(folds)]) for i in range(6)]

    class_rep_string_train = get_class_report_string(avg_stats["report_train"])
    class_rep_string_test = get_class_report_string(avg_stats["report_test"])
    
    with open(f"../my_data/metricsTrain{model_name}.txt", "w") as outfile:
        outfile.write(f"Epoch {epochs}\n")
        outfile.write(class_rep_string_train)
    with open(f"../my_data/metricsTest{model_name}.txt", "w") as outfile:
        outfile.write(f"Epoch {epochs}\n")
        outfile.write(class_rep_string_test)

    print(f"Train losses")
    for i in range(6):
        print(f"String {i}: ", avg_stats["train_loss"][i])
    print(f"Test losses")
    for i in range(6):
        print(f"String {i}: ", avg_stats["test_loss"][i])
    
    print("train average")
    print(mean(avg_stats["train_loss"]))
    print("test average")
    print(mean(avg_stats["test_loss"]))
    
# main to crate and save single network instance with its metrics
# if __name__ == "__main__":
#     print("Executing production main")
#     print("Loading data")
#     data = CustomImageDataset("global_res_hope.json", labels=False, drop_half_empty=False)
#     print("Data loaded")

#     #generator1 = torch.Generator().manual_seed(42)
#     seed = 42
#     torch.manual_seed(seed)

#     split_data = random_split(data, [2/3, 1/3]) # , generator=generator1
#     # construction of dataset by subset to use custom sampler, custom_image_dataset should have labels true

#     # new_dataset = CustomImageDataset()
#     # new_dataset.data = {"sections" : []}
#     # for index in split_data[0].indices:
#     #     new_dataset.data["sections"].append(split_data[0].dataset.data["sections"][index])
#     # new_dataset.labels = split_data[0].dataset.get_specific_labels(split_data[0].indices)
#     # train_dataloader = DataLoader(new_dataset, batch_size=512, sampler=ImbalancedDatasetSampler(new_dataset))
#         # sampler=ImbalancedDatasetSampler(split_data[0].dataset, indices=split_data[0].indices, labels=split_data[0].dataset.get_specific_labels(split_data[0].indices)))
#     train_dataloader = DataLoader(split_data[0], shuffle=True, batch_size=512)
#     test_dataloader = DataLoader(split_data[1], shuffle=True, batch_size=512)

#     learning_rate = 1e-3 # 0.1 * batch_size / 256 for sgd
#     loss_fn = nn.CrossEntropyLoss()
#     net = Network().to(device)
#     optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    
#     epochs = 7
#     epoch_stats = []
#     train_epoch_stats = []
#     test_metrics_dict = {"epochs": []}
#     train_metrics_dict = {"epochs": []}
#     for t in range(epochs):
#         print(f"Epoch {t+1}\n-------------------------------")
#         print(f"Started training at: {datetime.now()}\n")
#         train_loop(train_dataloader, net, loss_fn, optimizer, train_epoch_stats, train_metrics_dict)
#         print(f"Started testing at: {datetime.now()}\n")
#         test_loop(test_dataloader, net, loss_fn, epoch_stats, test_metrics_dict)
#     print("Done!")
#     print("Saving model")
#     model_name = "resnet_3layers19classes"
#     torch.save(net.state_dict(), f"../my_data/PauseChampDict{model_name}.json")
#     torch.save(net, f"../my_data/PauseChampFull{model_name}.json")
#     print("Model saved")
#     print("Saving metrics")
#     with open(f"../my_data/metricsTrain{model_name}.txt", "w") as outfile:
#         for i in range(epochs):
#             outfile.write(f"Epoch {i + 1}")
#             outfile.write(train_metrics_dict["epochs"][i]["report"])
#             outfile.write(train_metrics_dict["epochs"][i]["loss"])
#     with open(f"../my_data/metricsTest{model_name}.txt", "w") as outfile:
#         for i in range(epochs):
#             outfile.write(f"Epoch {i + 1}")
#             outfile.write(test_metrics_dict["epochs"][i]["report"])
#             outfile.write(test_metrics_dict["epochs"][i]["loss"])
#     print("Metrics saved")


#     epoch_stats = torch.Tensor(epoch_stats)
#     labels = ["e", "A", "D", "G", "B", "E"]
#     colors = ["k", "m", "c", "r", "g", "b"]
    
#     test_loss = torch.zeros((epochs,))
#     train_loss = torch.zeros((epochs,))
#     epoch_index = [i + 1 for i in range(epochs)]

#     for i in range(6):
#         test_loss += epoch_stats[:, 0, i]
#         plt.plot(epoch_index, epoch_stats[:, 0, i], label=labels[i], color=colors[i])

#     train_epoch_stats = torch.Tensor(train_epoch_stats)
#     for i in range(6):
#         train_loss += train_epoch_stats[:, 0, i]
#         plt.plot(epoch_index, train_epoch_stats[:, 0, i], '--', label=labels[i], color=colors[i])

#     test_loss /= 6
#     train_loss /= 6
#     print(train_loss)
#     print(test_loss)
#     plt.xlabel('epoch')
#     plt.ylabel('loss')
#     plt.title('Seperate losses graph')
#     plt.legend(loc="lower left")
#     plt.xticks(epoch_index)
#     plt.savefig(f"../my_data/Plot{model_name}.png")
#     plt.show()


#     plt.plot(epoch_index, train_loss, '--', label='train', color="r")
#     plt.plot(epoch_index, test_loss, label='test', color="r")
#     plt.xlabel('epoch')
#     plt.ylabel('loss')
#     plt.title('Average losses graph')
#     plt.legend(loc="lower left")
#     plt.xticks(epoch_index)
#     plt.savefig(f"../my_data/PlotAvg{model_name}.png")
#     plt.show()