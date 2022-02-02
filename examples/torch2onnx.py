import numpy as np
import onnx
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import timm


input_size = 256

class FaceSynthetics(pl.LightningModule):
    def __init__(self, backbone):
        super().__init__()
        self.save_hyperparameters()
        backbone = timm.create_model(backbone, num_classes=68*2)
        self.backbone = backbone
        self.loss = nn.L1Loss(reduction='mean')
        self.hard_mining = False

    def forward(self, x):
        # use forward for inference/predictions
        y = self.backbone(x)
        return y

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        if self.hard_mining:
            loss = torch.abs(y_hat - y) #(B,K)
            loss = torch.mean(loss, dim=1) #(B,)
            B = len(loss)
            S = int(B*0.5)
            loss, _ = torch.sort(loss, descending=True)
            loss = loss[:S]
            loss = torch.mean(loss) * 5.0
        else:
            loss = self.loss(y_hat, y) * 5.0
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss, on_step=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = self.loss(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        #return torch.optim.Adam(self.parameters(), lr=0.0002)
        opt = torch.optim.SGD(self.parameters(), lr = 0.1, momentum=0.9, weight_decay = 0.0005)
        def lr_step_func(epoch):
            return 0.1 ** len([m for m in [15, 25, 28] if m <= epoch])
        scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=opt, lr_lambda=lr_step_func)
        lr_scheduler = {
                'scheduler': scheduler,
                'name': 'learning_rate',
                'interval':'epoch',
                'frequency': 1}
        return [opt], [lr_scheduler]


def convert_onnx(path_module, output, opset=11, simplify=False):
    
    img = np.random.randint(0, 255, size=(input_size, input_size, 3), dtype=np.int32)
    img = img.astype(np.float)
    img = (img / 255. - 0.5) / 0.5  # torch style norm
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()

    net = FaceSynthetics.load_from_checkpoint(path_module)
    net.eval()
    assert isinstance(net, torch.nn.Module)

    torch.onnx.export(net, img, output, keep_initializers_as_inputs=False, verbose=False, opset_version=opset, input_names = ['input'])
    model = onnx.load(output)

    graph = model.graph
    graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'None'

    # TODO: look into why simplify throws an error
    if simplify:
        from onnxsim import simplify
        model, check = simplify(model)
        assert check, "Simplified ONNX model could not be validated"
    
    onnx.save(model, output)


if __name__ == '__main__':
    import os
    import argparse

    parser = argparse.ArgumentParser(description=' PyTorch to onnx')
    # parser.add_argument('--backbone', default='resnet50d', type=str)
    parser.add_argument('input', type=str, help='input backbone.pth file or path')
    parser.add_argument('--output', type=str, default=None, help='output onnx path')
    parser.add_argument('--simplify', type=bool, default=False, help='onnx simplify')
    args = parser.parse_args()
    input_file = args.input
    if os.path.isdir(input_file):
        input_file = os.path.join(input_file, "backbone.pth")
    assert os.path.exists(input_file)
    model_name = os.path.basename(os.path.dirname(input_file)).lower()

    output_path = args.output
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), 'onnx')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    assert os.path.isdir(output_path)
    
    output_file = os.path.join(output_path, "%s.onnx" % model_name)

    convert_onnx(input_file, output_file, simplify=args.simplify)

    print('model is saved to ' + output_path)