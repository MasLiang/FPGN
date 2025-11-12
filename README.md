This is the code for **FPGN: Ultra-Fast Programmable Gate-based Neural Acceleration with Differentiable LUTs**

**Including**

  cifar10: for training a model on CIFAR-10 dataset
  
  Compiler: automatically generate the RTL from a trained model for a given FPGA 

**Environment**

****For Training****
  
  Python                    3.10
  
  Pytorch                   2.2.1

  torchvision               0.17.1           

****For Compiler****

We used a commercial solver. Please install it here: https://www.gurobi.com/.

  cvxpy                     1.5.2

The generated RTL code can be used in any device. You need to crearte your own Vivado Project according to your requirements.

**How to use**

The training process is easy to follow. For the compiler, you need to provide a json file from your trained model. We provide an example file to generate the json file:

compiler/src/json_save_model.py

Then, add yoru constraint in the json file, like the bandwidth, the resource, and so on.

Finally, run the compiler/src/compiler.py and you will obtain the generated RTL.

  

