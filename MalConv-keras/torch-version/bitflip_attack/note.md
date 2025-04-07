# 
utils 文件夹，一些函数文件
trigger_generation 产生trigger的.py 文件
robust_amd_targeted.py target 攻击
robust_amd_untargeted.py untargeted 攻击
# 数据
模型训练
ben mal的前12000个是用来训练的train set，然后各取了3500个作为val set
# 数据特征
前147个是Permission
特征向量构成： permissions_147 | intent_actions_126 | sensitive_apis_106 

## untarget：
ben mal的前aux个作为测试集合，其余作为验证集合
环境：conda activate lkc_msdroid_bnb