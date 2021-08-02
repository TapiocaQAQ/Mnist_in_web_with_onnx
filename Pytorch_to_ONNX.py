import torch

from my_model import Net

def main():
    pytorch_model = Net()
    pytorch_model.load_state_dict(torch.load('mnist_cnn.model'))
    pytorch_model.eval()
    dummy_input = torch.zeros(1, 1, 28, 28)
    torch.onnx.export(pytorch_model, dummy_input, 'onnx_model.onnx', verbose=True)
    
if __name__ == '__main__':
    main()