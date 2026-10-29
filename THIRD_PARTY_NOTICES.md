# 复用来源与许可

本项目新增的是 V2 的区域内部覆盖约束、联合截断达标进度、动态停止、异常回退与模型接入。预训练模型架构、视觉编码器、Projector、模板、预处理、分词及自回归生成均复用已有实现。

## AgilePruner / LLaVA

- 仓库：https://github.com/cvsp-lab/AgilePruner
- 固定提交：`6829283ded10bdec58361512a32676c71bbd17d3`
- 许可：Apache-2.0；完整文本在 `third_party/licenses/AgilePruner-Apache-2.0.txt`。相关源码保留上游的 Haotian Liu 版权声明。
- 实际复制：`llava_arch.py`、`llava_llama.py`、`clip_encoder.py`、视觉/Projector builder、`constants.py`、`conversation.py`、`mm_utils.py`，保存在 `third_party/agilepruner/llava/`。其中三份本地源码快照已由前轮保存，其他通过固定提交的 GitHub 文件接口读取；未下载依赖或模型。
- `model/__init__.py` 调整为只导入本次支持的 Llama 实现；包初始化文件为本子集提供。没有改写原模型权重。
- `llava_adapter.py` 的加载、视觉层取特征及对话输入组织改编自上游 `builder.py`、`clip_encoder.py`、`model_vqa_loader.py`。增加离线读取、权重完整性检查及索引映射。为取得未经剪枝的 V0，直接使用上游 CLIP 和 Projector，不调用原 Agile 固定预算筛选。
- 原始源码中的 fixed-K 选择函数保留为来源代码，但本入口不调用；不能把它们误认为 V2 会回填固定数量。
- 生成复用 Transformers `GenerationMixin.generate`，从实际缩短后的 `inputs_embeds` 开始，并沿用原生 KV 与连续位置接口；不使用 Agile 包装层的固定预算生成路径。

## 第三方 jax-coverpruner

- 仓库：https://github.com/teddytennant/jax-coverpruner
- 固定提交：`e5ae93963fb164dd9bcd45de688eaba97be3d5d5`
- 版权：Copyright (c) 2026 Teddy Tennant。
- 许可：MIT；完整文本在 `third_party/licenses/jax-coverpruner-MIT.txt`。
- `coverage.py` 与 `selector.py` 的余弦/行中心化/非负效用、最大代表状态及向量化边际收益由该仓库 `src/coverpruner/coverage.py`、`select.py` 移植为 PyTorch，再加入 V2 的局部约束与停止条件。未引入 JAX 运行依赖。
- 这是第三方算法参考，**不是 CoverPruner 作者官方实现**，也没有把它的合成数组演示当作真实 VLM 结果。

## Transformers 与 RoRA

- Transformers 原生首层接口以 `v4.37.2` 为准，提交 `345b9b1a6a308a1fa6559251eb33ead2211240ac`，许可 Apache-2.0。`probe.py` 直接导入其 `apply_rotary_pos_emb`、`repeat_kv` 等原生功能；生成也复用原生实现，没有复制并维护另一份注意力或 decoder。
- 官方源码：https://github.com/huggingface/transformers/blob/345b9b1a6a308a1fa6559251eb33ead2211240ac/src/transformers/models/llama/modeling_llama.py
- RoRA（https://github.com/LukieLuu/RoRA ，提交 `abf6a7e6bb60a5590402f16cc589f00502e8903f`）提供已有网格/池化处理的参考。V2 使用四邻接峰值、阈值扩展和连通区域；没有复制其固定半径 AAR、角色预算或中层续算算法，也不宣称这些属于 V2 原创。

## Hugging Face Hub 与发布身份

- 直接复用 `huggingface_hub.utils.sha.sha_fileobj` 计算本地文件 SHA256，没有另写分块哈希器或复制一套 Hub 缓存/下载实现。Hub 已是 Transformers 的依赖；参考版本为 `v0.20.3`，本轮未安装或下载依赖。
- 官方 GitHub 固定提交：`2cff405655a58e91e68d2486a3d699e25fadc65b`，源码：https://github.com/huggingface/huggingface_hub/blob/2cff405655a58e91e68d2486a3d699e25fadc65b/src/huggingface_hub/utils/sha.py 。许可 Apache-2.0；本项目只调用已安装依赖的函数，没有分发其源码副本。
- `pairing.py` 的内置身份选择、文件集合检查和失败报告是本项目接入逻辑；模型/CLIP 预期指纹取自官方 HF 固定 revision API 的 LFS SHA256。小 JSON 文件实际字节另核 SHA256 和 Git blob；来源与完整版本见 `configs/pairing_profiles.json`，不是按当前任意本地文件生成可信身份。
- LLaVA 原配置声明 CLIP 仓库名，未指定训练时 CLIP revision。当前只锁定登记的官方发布文件及该仓库关系，不声称证实历史训练版本、模型效果或完成复现。本轮只读取元数据与小 JSON，没有下载权重、词表或数据。

`upstream_sources.json` 记录分发源码及外部函数的来源；未来运行仍需用户已有的相应 Python 依赖与本地权重。本文不包含或授予模型权重许可。
