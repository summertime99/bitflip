训练正常，模型参数存储路径为 model/best.pt
layer_norm = False
gnn_model = GNNStack(input_dim=(268+224), hidden_dim=128, output_dim=2, conv_func=None, global_pool='mix', layer_norm=layer_norm)

