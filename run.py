#!/usr/bin/env python

import numpy
import PIL.Image
import torch
import os

# === User config ===
args_strIn = r'image/01001_Mask_Mouth_Chin.jpg'
args_strOut = './output_unmask.png'
args_strModel = 'bsds500'

# === Auto Resize to 480x320 ===
def load_and_resize_image(path, width=480, height=320):
    image = PIL.Image.open(path).convert('RGB')
    image = image.resize((width, height), PIL.Image.BICUBIC)
    return image

# === HED Network Definition ===
class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.netVggOne = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, 1, 1), torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(64, 64, 3, 1, 1), torch.nn.ReLU(inplace=False)
        )
        self.netVggTwo = torch.nn.Sequential(
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(64, 128, 3, 1, 1), torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(128, 128, 3, 1, 1), torch.nn.ReLU(inplace=False)
        )
        self.netVggThr = torch.nn.Sequential(
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(128, 256, 3, 1, 1), torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(256, 256, 3, 1, 1), torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(256, 256, 3, 1, 1), torch.nn.ReLU(inplace=False)
        )
        self.netVggFou = torch.nn.Sequential(
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(256, 512, 3, 1, 1), torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(512, 512, 3, 1, 1), torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(512, 512, 3, 1, 1), torch.nn.ReLU(inplace=False)
        )
        self.netVggFiv = torch.nn.Sequential(
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(512, 512, 3, 1, 1), torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(512, 512, 3, 1, 1), torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(512, 512, 3, 1, 1), torch.nn.ReLU(inplace=False)
        )

        self.netScoreOne = torch.nn.Conv2d(64, 1, 1)
        self.netScoreTwo = torch.nn.Conv2d(128, 1, 1)
        self.netScoreThr = torch.nn.Conv2d(256, 1, 1)
        self.netScoreFou = torch.nn.Conv2d(512, 1, 1)
        self.netScoreFiv = torch.nn.Conv2d(512, 1, 1)

        self.netCombine = torch.nn.Sequential(
            torch.nn.Conv2d(5, 1, 1),
            torch.nn.Sigmoid()
        )

        # Load pretrained weights from official URL
        state_dict = torch.hub.load_state_dict_from_url(
            url=f'http://content.sniklaus.com/github/pytorch-hed/network-{args_strModel}.pytorch',
            file_name=f'hed-{args_strModel}',
            map_location='cpu'
        )
        self.load_state_dict({k.replace('module', 'net'): v for k, v in state_dict.items()})

    def forward(self, x):
        x = x * 255.0
        x = x - torch.tensor([104.00698793, 116.66876762, 122.67891434], dtype=x.dtype).view(1, 3, 1, 1)

        one = self.netVggOne(x)
        two = self.netVggTwo(one)
        thr = self.netVggThr(two)
        fou = self.netVggFou(thr)
        fiv = self.netVggFiv(fou)

        one = torch.nn.functional.interpolate(self.netScoreOne(one), size=x.shape[2:], mode='bilinear', align_corners=False)
        two = torch.nn.functional.interpolate(self.netScoreTwo(two), size=x.shape[2:], mode='bilinear', align_corners=False)
        thr = torch.nn.functional.interpolate(self.netScoreThr(thr), size=x.shape[2:], mode='bilinear', align_corners=False)
        fou = torch.nn.functional.interpolate(self.netScoreFou(fou), size=x.shape[2:], mode='bilinear', align_corners=False)
        fiv = torch.nn.functional.interpolate(self.netScoreFiv(fiv), size=x.shape[2:], mode='bilinear', align_corners=False)

        return self.netCombine(torch.cat([one, two, thr, fou, fiv], 1))

# === Run Detection ===
def run_edge_detection():
    image = load_and_resize_image(args_strIn, width=480, height=320)
    np_image = numpy.array(image)[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)
    tensor_input = torch.FloatTensor(numpy.ascontiguousarray(np_image)).unsqueeze(0)

    model = Network().cpu().eval()
    with torch.no_grad():
        output = model(tensor_input)[0].cpu()

    edge_np = output.clip(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0] * 255.0
    edge_img = PIL.Image.fromarray(edge_np.astype(numpy.uint8))
    edge_img.save(args_strOut)
    print(f"✅ Edge map saved to: {args_strOut}")

# === Run main ===
if __name__ == '__main__':
    run_edge_detection()
