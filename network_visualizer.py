import argparse

import torch
from torchviz import make_dot

from models import get_network
from utils.data import get_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show netwotk architecture')
    parser.add_argument('--model', type=str, default='ggcnn2', help='ggcnn or ggcnn2')
    parser.add_argument('--dataset_path', type=str, default='/Users/piotrwodecki/Projects/Grasping/cornell_grasp', help='path')
    args = parser.parse_args()
    dataset = 'cornell'
    Dataset = get_dataset(dataset)
    test_dataset = Dataset(args.dataset_path, start=0.9, end=1.0,
                           random_rotate=False, random_zoom=False,
                           include_depth=1, include_rgb=0)
    test_data = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )
    ggcnn = get_network(args.model)
    net = ggcnn()
    batch = next(iter(test_data))
    yhat = net(batch)  # Give dummy batch to forward().
    make_dot(yhat, params=dict(list(net.named_parameters()))).render("ggcnn2_torchviz", format="png")