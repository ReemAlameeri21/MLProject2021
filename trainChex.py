import torch
from textChex import TestChex

class TrainChex:
    def trainchxpert_nn(self, model, train_loader, test_loader, criterion, optimizer, epochs, modelname, device):

        best_val_acc = 0
        for epoch in range(epochs):

            print(f'Epoch number{epoch}')
            # set the model to training mode:
            model.train()
            running_loss = 0.0
            running_correct = 0.0
            total = 0

            for batch in train_loader:
                images = batch["img"]
                labels = batch["lab"]
                labels = torch.nan_to_num(labels)
                labels = labels.long()
                _, labels = torch.max(labels, 1)
                images = images.to(device)
                labels = labels.to(device)
                total += labels.size(0)  # get integer value

                optimizer.zero_grad()

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                loss = criterion(outputs, labels)
                loss.backward()

                optimizer.step()

                running_correct += (labels == predicted).sum().item()

            # epoch_loss= running_loss/total
            epoch_acc = (running_correct / total) * 100

            testCheXObj = TestChex
            epoch_val_acc = testCheXObj.testchxpert_model(model, test_loader, criterion, device)

            print(f'Training dataset result: {epoch_acc}% of the images classified correctly.')
            # best_val_acc=0
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                self.save_checkpoint(model, optimizer, epoch, best_val_acc, modelname)

        print('Done training')
        return model

    def save_checkpoint(model, optimizer, epoch, best_val_acc, modelname):

        state = {
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best accuracy': best_val_acc,
            'optimizer': optimizer.state_dict()

        }

        torch.save(state, f'{modelname}_checkpoint.pth.tar')