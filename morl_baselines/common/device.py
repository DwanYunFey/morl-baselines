import torch


def avilable_device()->torch.device:
    name = "cpu"
    if torch.backends.mps.is_available():
        name = "mps"
    elif torch.cuda.is_available():
        name = "cuda"
    return torch.device(name)

if __name__ == "__main__":
    print(avilable_device())