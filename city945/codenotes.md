* 知识点

  * 差异说明

    * kitti_dataset.yaml `WORLD_SCALE_RANGE` `used_feature_list` 无意义修改
  * mlcnet 与 pcdet 在数据集预处理文件的生成上， 通过差异代码初步判断，kitti/nus/waymo 完全相同
  * mlcnet 没有预训练，直接源域数据与目标域数据一起训练

* 代码阅读

  * 程序执行图

    * `python train_mean_teacher.py cfgs/kitti_models/pointrcnn_mean_teacher_waymo.yaml`

      * 自定义 BN 层 `custom_batchnorm.py`

        * 根据实际的成员变量值控制 BN 层行为，`BatchNorm1d` `BatchNorm2d` 实际使用的是普通的 nn.BatchNorm 层
        * `BatchNorm1d`

          * ```python
            """
            新增两个成员变量控制自定义 BN 层的行为
            1. momentum_update_for_teacher && update_running: 动量更新策略
                计算均值与方差
                动量更新均值与方差
                利用均值方差归一化输入特征向量 x、
            2. !momentum_update_for_teacher && !update_running: 普通归一化
            3. 其他，nn.BatchNorm 普通归一化

            # BatchNorm 是对 C 通道的切面进行，结果就是 C 个常数统计量，x 一般的形式是 [B, C, H*W]
            mean = torch.mean(x, dim=[0, 2])
            """
            ```
      * `build_dataloader_mt`

        * ```python
          """
          相当于执行两次 build_dataloader
          Returns:
          train_set, train_loader, train_sampler: 源域数据集相关
          train_loader_target, train_sampler_target: 目标数据集相关
          """
          ```
      * `build_network`* 2

        * `PointRCNNMeanTeacherMerge`
        * 配置参数都是 `MODEL`，所以是两个一模一样的模型
        * ema 教师网络的参数分离，不计算梯度
        * 将学生模型的 BN 参数拷贝到教师模型，则教师模型在训练过程中没有要训练的参数，启用 eval 模式；实际的拷贝操作发生在 `update_ema_variables`中
      * `train_model`

        * `train_one_epoch`

          * `KittiDataset.__getitem__` + `DatasetTemplate.collate_batch`
          * `WaymoDatasetMeanTeacher.__getitem__` + `WaymoDatasetMeanTeacher.collate_batch`

            * ```python
              """
              原始数据进行数据增强得到增强后的数据，增强后的数据作为教师模型目标域数据，再增强一次得到学生模型目标域数据
              Returns:
                  output: [data_dict1, data_dict2]
              """
              ```

              * `WaymoDatasetMeanTeacher`

                *
          * `model_fn_decorator_for_mt_merge_source_target`

            * ```python
              """
              三个函数没有本质区别，model_fn_decorator_for_mt 最直观
              model_fn_decorator_for_mt: 
                  batch_dict 包含一个源域数据 s1，两个不同数据增强强度的目标域数据 t1(强增强) t2
                  t2 经过教师模型的结果与 s1 t1 经学生模型计算损失
              model_fn_decorator_for_mt_merge_source_target: batch_dict 包括一个源域数据 s1，两个目标域数据 t1 t2，s1 t1 合并成 batch_merge，再同理计算
              model_fn_decorator_for_mt_merge_both_models: batch_dict 包括两个源域数据 s1 s2，两个目标域数据 t1 t2，分别合并为 batch_student batch_teacher，再同理计算

              Funcs:
                  merge_batch_dicts: 字典拼接，list 都 cat 拼接上
              """
              ```

              ```
              ```
            * `PointRCNNMeanTeacherMerge.forward`

              * `PointNet2MSG`

                * ```python
                  bottom_features: MSG 多尺度特征拼接，这里的最后一层 SA 输出是最大尺度的特征
                  ```
              * `PointHeadBoxMerge`

                * ```python
                  """
                  可以只对源域数据分配标签
                  """
                  ```
              * `PointRCNNHeadMTMerge`
              * `batch_dict`
            * `PointRCNNMeanTeacherMerge.forward`

              * `PointNet2MSG`
              * `PointHeadBoxMerge`

                * 模块进行过修改，可以输入单个或一源域一目标域数据
              * `PointRCNNHeadMTMerge`
              * `batch_merge`
              * `get_training_loss`

                * `PointHeadBoxMerge.get_loss`
                * `RoIHeadTemplate.get_loss`
              * `get_consistency_loss_roi`

                ```python
                """
                Args:

                Cfgs:

                Locals:

                Returns:

                """
                ```
              * `get_consistency_loss_rcnn`
          * `update_ema_variables`

            * ```python
              """
              1. 更新 ema 教师模型的参数，theta' 为教师模型参数，theta 为学生模型参数
              2. 如果要求将学生模型的 BN 参数拷贝到教师模型，则拷贝模型的 model.named_buffers()
                  模型参数分两种，包括可训练参数 parameter 和不可训练的 buffer，都是字典类型，named_buffers 返回 name 和 buffer 值 

              BN_WARM_UP
              BN_EMA_DECAY
              """
              ```

              * $$
                \theta' = m\theta' + (1-m)\theta
                $$