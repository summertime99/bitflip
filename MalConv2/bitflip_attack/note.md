### malconvGCT_nocat.checkpoint

This file contains the weights for the GCT model from our paper’s results. It has some extra parameters that were never used due to some lines left commented in durning model training. It also has an off-by-one “bug” that says its the 21’st epoch 
instead of the 20’th. 

To load this file, you want to have code that looks like:

加载现有模型

```python
from MalConvGCT_nocat import MalConvGCT

mlgct = MalConvGCT(channels=256, window_size=256, stride=64,)
x = torch.load("malconvGCT_nocat.checkpoint.checkpoint")
mlgct.load_state_dict(x['model_state_dict'], strict=False)
```

### 部署攻击需要修改的四个部分
- model 模型量化 -----> 加载模型  主要替换 linear layer
- dataset 加载数据集 ------> load_data 中的 dataloader 
- metrics 计算 ASR / ACC / LOSS 需要处理模型的输入输出
- quant 量化后的线性层的计算方法可能需要修改

目前看到主要的难点是模型量化和加载数据集。metrics 和 quant 应该不需要修改

模型量化难点在于 MalConv2 的源代码实现的结构可能比较复杂，还需要分析基类的结构，考虑基类需不需要量化。（MalConvML 里面有两个 linear）done for now

加载数据集难点在于源代码只需要记录一个文件路径的文件，也没有保存“真正的”输入数据。可能需要仔细读一读源代码有没有预处理数据，然后将数据给保存下来。不然就只能复用源代码里加载数据的方式，但这样可能需要大幅修改 load_data.py
不用修改loader，可以直接复用malconv的文件

metrics 可能也需要修改。目前不清楚 model 的输出是什么格式
for inputs, labels in tqdm(train_loader):
outputs, penultimate_activ, conv_active = model(inputs)
loss = criterion(outputs, labels)
