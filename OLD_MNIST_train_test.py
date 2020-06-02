

def train(args, model, device, train_loader, optimizer, epoch):
    '''
    #This is your training function. When you call this function, the model is
    #trained for 1 epoch.
    '''
    model.train()   # Set the model to training mode
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()               # Clear the gradient
        output, hidden_layer = model(data)                # Make predictions
        loss = F.nll_loss(output, target)   # Compute loss
        loss.backward()                     # Gradient computation
        optimizer.step()                    # Perform a single optimization step
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                100. * batch_idx / len(train_loader), loss.item()))

        total_loss = total_loss + loss.item()
    #train loss for each epoch is an average of the loss over all mini-batches
    train_loss = total_loss/batch_idx

    return train_loss



# OLD test MNIST for refernce
def test(model, device, test_loader, evaluate = False):
    model.eval()    # Set the model to inference mode
    test_loss = 0
    correct = 0
    test_num = 0

    images = []
    allimages = []
    master_preds = []
    master_truths = []
    master_hidden_layers = []
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, hidden_layer = model(data)

            #feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            print(len(hidden_layer))
            print(len(hidden_layer[0]))
            #print(hidden_layer[0])


            correct += pred.eq(target.view_as(pred)).sum().item()
            test_num += len(data)


            if evaluate:
                for i in range(len(pred)):
                    master_preds.append(pred[i][0].item())
                    master_truths.append(target[i].item())
                    layer = hidden_layer[i].cpu()
                    master_hidden_layers.append(layer.numpy())
                    image = data[i][0].cpu()
                    allimages.append(image.numpy())
                    if pred[i][0] == target[i]:
                        continue
                    else:
                        #print("not equal")
                        #print("pred is ", pred[i][0].item(), "and target is ", target[i].item())
                        image = data[i][0].cpu()
                        images.append([image.numpy(),pred[i][0].item(),target[i].item()])

        if evaluate:

            #print(len(master_hidden_layers))
            #print(master_hidden_layers[0])

            distances = np.zeros(len(master_hidden_layers))

            #x0 = master_hidden_layers[0]

            for i in range(len(distances)):
                length = 0
                for dim in range(len(master_hidden_layers[0])):
                    length = length + (master_hidden_layers[i][dim] - master_hidden_layers[15][dim])**2
                length = math.sqrt(length)
                distances[i] = length

            sorted_distance_index = np.argsort(distances)

            figa = plt.figure()


            print("test")
            for i in range(9):
                sub = figa.add_subplot(9, 1, i + 1)
                sub.imshow(allimages[sorted_distance_index[i]], interpolation='nearest', cmap='gray')

            X = master_hidden_layers
            y = np.array(master_truths)
            tsne = TSNE(n_components=2, random_state=0)
            X_2d = np.array(tsne.fit_transform(X))

            target_ids = range(10)

            cdict = {0: 'orange', 1: 'red', 2: 'blue', 3: 'green', 4: 'salmon', 5:'c', 6: 'm', 7: 'y', 8: 'k', 9: 'lime'}

            fig, ax = plt.subplots()
            for g in np.unique(y):
                ix = np.where(y == g)
                ax.scatter(X_2d[ix, 0], X_2d[ix, 1], c=cdict[g], label=g, s=5)
            ax.legend()
            plt.show()


            #i = 1
            #plt.figure(figsize=(6, 5))
            #plt.scatter(X_2d[10*i:10*i+10,0],X_2d[:10,1])



            CM = confusion_matrix(master_truths,master_preds)
            CMex = CM
            #for i in range(len(CM)):
            #    for j in range(len(CM)):
            #        if CM[i][j] > 0:
            #            CMex[i][j] = log(CM[i][j])
            #        else:
            #            CMex[i][j] = CM[i][j]

            print(CM)
            print(CMex)

            df_cm = pd.DataFrame(CM, range(10), range(10))
            #plt.figure(figsize=(10,7))
            fig0,ax0 = plt.subplots(1)
            sn.set(font_scale=1)  # for label size
            sn.heatmap(df_cm, annot=True, annot_kws={"size": 11})  # font size
            #ax0.set_ylim(len(CMex) - 0.5, 0.5)
            plt.xlabel("predicted")
            plt.ylabel("ground truth")
            plt.show()




            fig = plt.figure()

            for i in range(9):
                sub = fig.add_subplot(3, 3, i + 1)
                sub.imshow(images[i + 10][0], interpolation='nearest', cmap='gray')

                title = "Predicted: " + str(images[i+ 10][1]) + " True: " + str(images[i+ 10][2])
                sub.set_title(title)

            kernels = model.conv1.weight.cpu().detach().clone()
            kernels = kernels - kernels.min()
            kernels = kernels / kernels.max()

            kernels = kernels.numpy()
            print(np.shape(kernels))

            fig2 = plt.figure()
            for i in range(8):

                sub = fig2.add_subplot(2, 4, i + 1)
                sub.imshow(kernels[i][0], interpolation='nearest', cmap='gray')

                title = "Kernel #" + str(i + 1)
                sub.set_title(title)


        #fig, axs = plt.subplots(3, 3, constrained_layout=True)
        #for i in range(9):
        #    fig[i].imshow(images[i][0], interpolation='nearest', cmap='gray')
        #    axs[i].set_title("all titles")





    test_loss /= test_num

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, test_num,
        100. * correct / test_num))

    return test_loss

