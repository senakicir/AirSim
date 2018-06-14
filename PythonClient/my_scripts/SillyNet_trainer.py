from SillyNet import *
import helpers as my_helpers
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from torch.nn import MSELoss
from NonAirSimClient import prepareDataForResponse, DummyPhotoResponse
from State import DRONE_POS_IND, DRONE_ORIENTATION_IND
from determine_positions import determine_2d_positions

def load_data(batch_size=32, all_data = True):
    for key, test_set_name in my_helpers.TEST_SETS.items():
        filename_bones = 'test_sets/'+test_set_name+'/groundtruth.txt'
        groundtruth_matrix = pd.read_csv(filename_bones, sep='\t', header=None).ix[:,1:].as_matrix().astype('float')                
        DRONE_INITIAL_POS = groundtruth_matrix[0,0:3]
        groundtruth = groundtruth_matrix[1:,:-1]

        dataloader = []
        dataset_size = groundtruth.shape[0]

        shuffled_indices = np.random.permutation(dataset_size)
        for i in range(0, dataset_size%batch_size): #how many batches will come out of this
            bone_2d_arr = torch.zeros(batch_size, 2*21).float().cuda()
            bone_3d_arr = torch.zeros(batch_size, 3, 21).float().cuda()
            m = 0
            for lines in shuffled_indices[i*batch_size:batch_size*(i+1)]: #shuffle within each batch
                response = prepareDataForResponse(groundtruth, DRONE_INITIAL_POS, lines)
                bone_pos_3d_GT = response.bone_pos
                bone_2d, _ = determine_2d_positions(3, response.unreal_positions, bone_pos_3d_GT)
                bone_2d = (bone_2d - torch.mean(bone_2d, dim=1).unsqueeze(1))/torch.std(bone_2d, dim=1).unsqueeze(1)
                hip_2d = bone_2d[:, 0].unsqueeze(1)
                bone_2d = bone_2d - hip_2d

                #normalize pose
                mean_3d = np.mean(bone_pos_3d_GT,axis=1)
                std_3d = np.std(bone_pos_3d_GT,axis=1)
                bone_pos_3d_GT = (bone_pos_3d_GT - mean_3d[:, np.newaxis])/std_3d[:, np.newaxis]
                hip = bone_pos_3d_GT[:, 0]
                bone_3d_temp = bone_pos_3d_GT - hip[:, np.newaxis]
                
                bone_2d_arr[m, :] = bone_2d.contiguous().view(-1, 2*21)
                bone_3d_arr[m, :, :] = torch.from_numpy(bone_3d_temp)
                m += 1

            data = {"inputs": bone_2d_arr, "labels": bone_3d_arr}
            dataloader.append(data)

    return dataloader

def initNet():
    net = SillyNet().cuda()
    return net

def trainer():
    net = initNet()

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=1, momentum=0.8)
    batch_size = 32
    num_epochs = 500
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        dataloader = load_data(batch_size)

        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            # get the inputs
            inputs, labels = data["inputs"], data["labels"]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            outputs_norm = outputs.view(batch_size, 3,21)
            outputs_norm = (outputs_norm - torch.mean(outputs_norm, dim=2).unsqueeze(2))/torch.std(outputs_norm, dim=2).unsqueeze(2)
            hip_new = outputs_norm[:, :, 0].unsqueeze(2)
            outputs_norm = outputs_norm - hip_new

            loss = criterion(outputs_norm, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        if epoch % 10 == 9:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

    print('Finished Training')
    torch.save(net.state_dict(), 'SillyNetWeights.pth')

def tester():
    net = SillyNet().cuda()
    net.load_state_dict(torch.load('SillyNetWeights.pth'))
    net.train(False)
    
    dataloader = load_data(1)

    for i, data in enumerate(dataloader, 0):
        # get the inputs
        inputs, labels = data["inputs"], data["labels"]
        # forward + backward + optimize
        outputs = net(inputs)
        
        bone_2d = inputs[0,:,:].data.cpu().numpy()
        bone_3d = outputs.data.cpu().numpy()
        bone_3d_gt = labels.data.cpu().numpy()

        
        fig = plt.figure()
        for i, bone in enumerate(my_helpers.bones_h36m):
            plt.plot(bone_2d[0,bone], bone_2d[1,bone], c='b', marker='^')               
        plt.savefig('/cvlabdata2/home/kicirogl/PythonClient/my_scripts/temp_main/test2d_' + str(i) + '.png', bbox_inches='tight', pad_inches=0)
        plt.close()

        fig = plt.figure()
        gs1 = gridspec.GridSpec(1, 1)
        ax = fig.add_subplot(gs1[0], projection='3d')

        # maintain aspect ratio
        X = bone_3d[0,:]
        Y = bone_3d[1,:]
        Z = -bone_3d[2,:]
        max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() * 0.8
        mid_x = (X.max()+X.min()) * 0.5
        mid_y = (Y.max()+Y.min()) * 0.5
        mid_z = (Z.max()+Z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        for i, bone in enumerate(my_helpers.bones_h36m):
            ax.plot(bone_3d[0,bone], bone_3d[1,bone], -bone_3d[2,bone], c='r', marker='^')
        for i, bone in enumerate(my_helpers.bones_h36m):
            ax.plot(bone_3d_gt[0,bone], bone_3d_gt[1,bone], -bone_3d_gt[2,bone], c='b', marker='^')
        plt.savefig('/cvlabdata2/home/kicirogl/PythonClient/my_scripts/temp_main/test3d_' + str(i) + '.png', bbox_inches='tight', pad_inches=0)
        plt.close()

        break

if __name__ == "__main__":
    trainer()