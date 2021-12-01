import torch

class TestChex:
    def testchxpert_model(model, test_loader, criterion, device):
        model.eval()
        predicted_correctly = 0
        total = 0
        test_running_loss = 0
        t_loss = 0.0

        with torch.no_grad():
            for batch in test_loader:
                images = batch["img"]
                labels = batch["lab"]
                labels = torch.nan_to_num(labels)
                labels = labels.long()
                _, labels = torch.max(labels, 1)

                images = images.to(device)
                labels = labels.to(device)
                total += labels.size(0)  # get integer value

                outputs = model(images)

                # loss = loss_func_test(outputs, labels)
                _, predicted = torch.max(outputs, 1)

                predicted_correctly += (predicted == labels).sum().item()
                # test_running_loss+= loss.item() * labels.size(0)

        test_acc = (predicted_correctly / total) * 100
        # t_loss = test_running_loss/total

        print(f'Test dataset result: {predicted_correctly} images out of {total} are classified correctly {test_acc}')

        return test_acc