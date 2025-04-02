vae epoch: 50 100 150 mlp acc:(0.8439 -> 0.855)区间 benign和malware都(sigma - 1.2)
不减，0.840左右
normalize 0.840左右

# 数据：
feature是379
data/benign.npy 29198个apk，[29198 * 379]
data/malware.npy 15911个apk [15911 * 379]
# 训练：
benign 和 malware都用12000个样本来训练，其它用来val
`python torch_main.py` 就可以进行训练，训练的输出保存在文件夹：model内部
`train_vae_mlp.log` 是目前模型的训练效果的输出
# 攻击：
bitflip attack是攻击文件夹，int8量化也在bitflip attack内部，训练好的fp32模型是vae_model_f32.pth/mlp_model_f32.pth
