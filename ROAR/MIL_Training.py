import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')
import sys
from model import *         # actual MIL model
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from mil_dataloader import MilDataLoader
import torch.optim as optim
import pickle
from random import randint
from time import sleep

PERCENT = float(sys.argv[2])
WHAT = sys.argv[1]


CLASS_NUM = #number of classes


train_loader = torch.utils.data.DataLoader(MilDataLoader(WHAT,PERCENT,train=True), num_workers=1)
test_loader = torch.utils.data.DataLoader(MilDataLoader(WHAT,PERCENT,train=False), num_workers=1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ngpu = torch.cuda.device_count()
print("Found device: ", ngpu, "x ", device)

model = SCEMILA(class_count=CLASS_NUM, multicolumn=False, device=device)

if(ngpu > 1):
    model = torch.nn.DataParallel(model)
model = model.to(device)
print("Setup complete.")
print("")

##### set up optimizer and scheduler
optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9, nesterov=True)
scheduler = None
epochs = 120

s_test = 0
#Training
loss_func = nn.CrossEntropyLoss()

for ep in range(epochs):
    model.train()

# initialize data structures to store results
    corrects = 0
    train_loss = 0.
    time_pre_epoch = time.time()


    for bag, label in train_loader:
        optimizer.zero_grad()

        # send to gpu
        label = label.to(device)
        bag = bag.to(device).squeeze()


        prediction, att_raw, att_softmax = model(bag)

        loss_out = loss_func(prediction, label)
        train_loss += loss_out.data

      
        loss_out.backward()

        optimizer.step()
        optimizer.zero_grad()
               
            
            # transforms prediction tensor into index of position with highest value
        label_prediction = torch.argmax(prediction, dim=1).item()
        label_groundtruth = label.item()
            
        if(label_prediction == label_groundtruth):
            corrects += 1
           

    samples = len(train_loader)
    train_loss /= samples

    accuracy = corrects/samples

    print('- ep: {}/{}, loss: {:.3f}, acc: {:.3f}, {}s'.format(
        ep+1, epochs, train_loss.cpu().numpy(), 
        accuracy, int(time.time()-time_pre_epoch)),end=' ')
    

    
    #Testing
    model.eval()


    # initialize data structures to store results
    corrects = 0
    train_loss = 0.
    time_pre_epoch = time.time()
    confusion_matrix = np.zeros((5, 5), np.int16)
        #data_obj = DataMatrix()

    optimizer.zero_grad()
    backprop_counter = 0
    loss_func = nn.CrossEntropyLoss()

    for bag, label in test_loader:

                # send to gpu
        label = label.to(device)
        bag = bag.to(device)

        bag = bag.squeeze()

    # forward pass
        prediction, att_raw, att_softmax = model(bag)



        loss_out = loss_func(prediction, label)
        train_loss += loss_out.data

                
                
                # transforms prediction tensor into index of position with highest value
        label_prediction = torch.argmax(prediction, dim=1).item()
        label_groundtruth = label.item()
                
        if(label_prediction == label_groundtruth):
            corrects += 1
            confusion_matrix[label_groundtruth, label_prediction] += int(1)

                # print('----- loss: {:.3f}, gt: {} , pred: {}, prob: {}'.format(loss_out, label_groundtruth, label_prediction, prediction.detach().cpu().numpy()))

    samples = len(test_loader)
    train_loss /= samples

    accuracy = corrects/samples

    if accuracy > s_test:
        s_test = accuracy
        torch.save(model, os.path.join("models","mil_" + WHAT + "_" + str(PERCENT) + ".pt"))


    print('test_loss: {:.3f}, test_acc: {:.3f}, {}s'.format(
        train_loss.cpu().numpy(), 
        accuracy, int(time.time()-time_pre_epoch)))



sleep(randint(1,10))
# torch.save(model, os.path.join("models","model.pt"))
if os.path.exists("results/" + WHAT + ".dat"):
    with open("results/" + WHAT + ".dat", "rb") as f:
        rundata = pickle.load(f)
else:
    rundata = []

rundata.append([WHAT,PERCENT,s_test,accuracy])
with open("results/" + WHAT + ".dat", "wb") as f:
    data = pickle.dump(rundata,f)














