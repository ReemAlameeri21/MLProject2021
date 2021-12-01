import torchxrayvision as xrv
import torchvision
from torch.utils.data import DataLoader

class LoadChex:

    @staticmethod
    def loadingChex():
        batch_size = 32
        transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                                    xrv.datasets.XRayResizer(224)])
        d_chex_train = xrv.datasets.CheX_Dataset(
            imgpath="/home/shahad.hardan/Documents/covid19-radiography-database/CheXpert-v1.0-small",
            csvpath="/home/shahad.hardan/Documents/covid19-radiography-database/CheXpert-v1.0-small/train.csv",
            transform=transform)

        d_chex_valid = xrv.datasets.CheX_Dataset(
            imgpath="/home/shahad.hardan/Documents/covid19-radiography-database/CheXpert-v1.0-small",
            csvpath="/home/shahad.hardan/Documents/covid19-radiography-database/CheXpert-v1.0-small/valid.csv",
            transform=transform)

        train_dataloader = DataLoader(dataset=d_chex_train,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=8)

        valid_dataloader = DataLoader(dataset=d_chex_valid,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=8)