import torch


def pytest_configure(config):
    torch.set_num_threads(1)
