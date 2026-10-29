# V2 问题引导的区域覆盖与动态视觉 Token 删减

按 2026-10-29 确认的 V2 主方案实现单图 DEMO。模型、图文模板、视觉编码器和 Projector 复用固定版本的 AgilePruner/LLaVA；覆盖计算移植已核对的第三方实现，区域内部约束与动态停止按本项目 V2 补充。

**当前交付状态：代码已编写并进行文本静态审查，尚未执行、编译、导入或运行测试；没有安装依赖、下载模型或数据。不能据此报告精度、速度、显存或复现成功。**

## 支持范围

- 已登记官方固定版本的原始 LLaVA-1.5 7B/13B 与 CLIP-336 权重，单图、单样本、单轮问题。身份登记不表示已验证 13B 在当前硬件上的容量或运行效果。
- Python 3.10；模型接口明确针对 Transformers 4.37.2；模型推理 FP16，Probe 得分/覆盖统计 FP32。
- 复用 Transformers 原生 RoPE、GQA、生成循环及 KV 缓存，不另写语言模型。
- 当前使用 eager 注意力与 CUDA。HF 转换格式、Qwen、多图、多裁剪、S2、量化模型和其他 Transformers 版本会明确拒绝或进入带原因的回退；未声称兼容。
- 所有模型、tokenizer、处理器和图像均从本地读取；入口启用 Hugging Face 离线模式。没有自动安装、自动下载或上传功能。

## 实际流程

1. 上游图像预处理后，调用复用的 CLIP 与 Projector，各计算一次，缓存完整 `V0`。
2. 使用上游对话模板与 tokenizer，定位问题的文本 Token，在完整未删序列上复用首层 LN/Q/K。每个问题 Query 先对全部合法因果 Key 做 softmax，再抽取视觉列，跨问题位置与注意力头平均。
3. 由 `q` 的峰值、阈值扩展与四邻接连通分量登记全部重点区域，区域外位置为 `R0`。
4. 在原投影空间计算 `A=max(cos-row_mean,0)`。全图需求与区域内部需求使用同一个 `A`，分母固定。
5. 按统一截断达标进度 `H` 的真实新增收益逐枚选择；全图和全部局部覆盖达标后停止，`K` 随选集产生。
6. 按原视觉顺序回取原始 `V0`，保留全部文本，调用原生语言模型生成。

不执行固定 K 回填、额外区域可信度过滤、近邻锚点保护或旧数量奖励。`max_tokens` 默认 `null`；显式设置上限且未达标时状态为 `resource_limited_unsatisfied`，不能记为正常成功。

### 位置与边界约定

- 原始视觉索引、二维网格和完整 Probe 序列索引分别保存。
- 最终主干沿用所复用 LLaVA 的原生处理：回取后保持顺序，为缩短后的图文序列使用连续位置输入。**原始顺序不等于把带空洞的旧绝对位置直接传给 4.37.2 的 RoPE。**
- 问题位置采用完整模板上的字符范围，通过前缀 Token 对齐定位；跨越字符边界的 BPE Token 纳入问题范围。真实 tokenizer 边界行为仍待运行核验。
- Probe 不产生用于主干续算的隐藏状态或 KV 缓存；生成从正式主干入口开始。
- 信号退化、无效向量、无效空间映射或覆盖分母异常时保留全部原始视觉 Token，并记录原因。CUDA OOM 为执行失败，不能当作成功回退。
- 问题为空或只含空白时保留原始输入的空问题，跳过 Probe 与剪枝并使用完整视觉 Token；此时模型的回答不保证符合任何具体任务。若视觉编码和投影已成功但无法可靠对应二维网格，也保留完整视觉序列并记录空间映射原因，报告不绘制伪造的网格/区域。若图像本身无法解码、视觉编码未产生 Token 或模型接口失败，完整序列无法取得，仍记录为执行失败。
- 加载时检查缺失/不匹配权重，避免 `from_pretrained` 随机初始化缺失参数后仍产生实验结果。允许的额外键仅对应已声明的延迟视觉塔或完整 CLIP 中非视觉部分，加载信息写入运行记录。

## 文件

| 路径 | 用途 |
|---|---|
| `v2_pruning/llava_adapter.py` | 离线加载、输入与索引映射、原 Token 回取、原生生成接口 |
| `v2_pruning/pairing.py`、`configs/pairing_profiles.json` | 官方发布版本的配套身份、文件指纹与加载前校验 |
| `v2_pruning/probe.py` | 冻结首层 Q/K 问题信号 |
| `v2_pruning/coverage.py` | 问题权重与原投影空间覆盖效用 |
| `v2_pruning/regions.py` | 峰值、平台与四邻接区域 |
| `v2_pruning/selector.py` | 精确联合增益、动态停止、完整回退与轨迹 |
| `v2_pruning/cli.py` | 本地 JSONL 输入、单样本处理、运行记录 |
| `v2_pruning/reporting.py` | 严格 JSON 与自包含 SVG 图示 |
| `v2_pruning/evaluation.py` | DEMO 匹配评分、明确分母的质量与成本汇总 |
| `v2_pruning/timing.py` | 分阶段同步计时 |
| `third_party/agilepruner/` | 保留版权的固定版本上游模型代码子集 |
| `THIRD_PARTY_NOTICES.md`、`upstream_sources.json` | 复用来源、版本、许可和校验信息 |
| `tests/` | 算法、Probe、报告与接入契约测试，尚未运行 |

## 以后执行时的输入

以下仅为使用说明，本次没有执行这些命令。先把已有环境和权重放在实际本地目录。`requirements.reference.txt` 只记录参考依赖，不代表已经安装。

输入 JSONL 兼容 AgilePruner 常见字段：

```json
{"question_id":"case-001","image":"sample.jpg","text":"How many birds are visible?","reference_answer":"3"}
```

也可使用 `question` 代替 `text`，`answer` 或 `answers` 代替 `reference_answer`。参考答案支持字符串、数字或这些类型的列表；不支持的参考结构记录为未评分。图像文件必须位于指定的 `image-folder` 下。例子不包含真实图片或实验结果。

从本目录执行的命令示例（目录需替换成实际已存在路径）：

```bash
python -m v2_pruning \
  --model-path /data/models/llava-v1.5-7b \
  --vision-path /data/models/clip-vit-large-patch14-336 \
  --config configs/v2_example.json \
  --question-file /data/demo/questions.jsonl \
  --image-folder /data/demo/images \
  --output-dir outputs/demo-001
```

输出目录必须尚不存在，防止覆盖既有结果。`configs/v2_example.json` 沿用论文笔记中的教学参数，仅用于接口示例，**不是校准过的正式超参数**；正式测量前需在独立开发数据上确定参数并固定。

入口在创建输出目录前导入必要依赖并检查延迟加载的模型接口；缺依赖时不占用输出目录，修复环境后可重试同一命令。进入模型初始化后发生的失败会保留 `run.json` 和失败原因，后续应使用新的输出目录。问题文件的格式、标识及重复检查仍在批次开始前完成；单张图片缺失、损坏、越出图片目录或被 Pillow 像素保护拒绝时，写入 `errors.jsonl`，继续处理后续样本，最终以非零退出码提示有失败。像素保护保持启用。OOM 记录错误与当前汇总后终止，不能冒充成功回退。

### 配套身份

默认 `--pairing-profile auto` 依据本地模型原始 `config.json` 的字节指纹匹配内置身份；也可明确指定 `llava-v1.5-7b` 或 `llava-v1.5-13b`。两者均使用原配置声明的 `openai/clip-vit-large-patch14-336`，具体文件清单和预期 SHA256 位于 `configs/pairing_profiles.json`。

| 组件 | 固定发布版本 |
|---|---|
| `liuhaotian/llava-v1.5-7b` | `4481d270cc22fd5c4d1bb5df129622006ccd9234` |
| `liuhaotian/llava-v1.5-13b` | `901a44b9113dea67b976e71f58d4e372cf9de81a` |
| `openai/clip-vit-large-patch14-336` | `ce19dc912ca5cd21c8a653c79e251e808ccabcd1` |

在替换内存中的视觉路径和加载权重前，校验模型配置、全部原始 PyTorch 权重分片与索引、tokenizer 文件、CLIP 权重和预处理配置的大小及 SHA256。完整模型分片包含 Projector；官方附带的独立 `mm_projector.bin` 不参与加载。哈希计算直接复用 `huggingface_hub.utils.sha.sha_fileobj`，运行时只读取本地文件。正常 HF 缓存符号链接和本地目录改名不影响身份判断。

相同结构但权重不同、缺文件、未登记的覆盖加载文件、被重写的配置或转换格式均拒绝；没有通过当前文件自报哈希加入信任或跳过核验的开关。`from_pretrained` 固定加载已核的 PyTorch bin 格式，避免额外 safetensors 被优先选用。核验拒绝时记录 `failed_identity` 和可取得的预期/实际信息，停止加载；模型或视觉目录、配置缺失，以及内置清单无法读取/解析或目录无法枚举，也按身份失败记录组件、路径和原因。完整视觉回退不能修复模型配套错误。

成功后 `run.json.pairing_identity` 保存原 `mm_vision_tower`、配套身份、两端来源与 revision、实际文件位置、预期/实际指纹和清单指纹。每次启动需读取全部登记权重计算哈希，属于一次性加载前开销，不计入逐样本推理时延；加载期间文件应保持不变。

这些身份确认的是“与指定官方发布文件一致，且原模型配置声明该 CLIP 仓库”。原配置未声明训练时 CLIP 的精确 revision，不能据此证明历史训练版本或推理效果。后续增加其他模型或转换格式时，应另核来源并审查清单，不能只更改目录名。

## 输出与解释

- `run.json`：配置、模型路径、配套身份及文件指纹、实际软件/GPU 信息、权重加载状态、处理数量与失败/回退统计。
- `results.jsonl`：每个完成样本的答案、原始/保留 Token 数、选择状态、覆盖轨迹、信号与时延。
- `summary.json`：完成/失败/未处理数量，已评分样本的准确率和分母，K 与时延的均值、中位数、P90/P95、极值，按选择状态分组的质量与成本，以及按原 Token 总数加权的压缩率。完整回退按实际 K=N 计入，未运行的阶段不填成 0。
- `answers_agile_format.jsonl`：复用上游答案字段格式，可供以后按相应基准的官方评分脚本处理；本入口不运行官方评分或基线。
- `errors.jsonl`：无法完成的样本，不将其悄悄丢弃为成功。
- `case_000001/result.json`、`visual_tokens.svg`：具体记录和问题热图、区域、实际保留位置。底图是模型实际看到的预处理后画布，不冒充未裁剪原图。

当网格不可用时，`grid_shape` 为 `null`，`spatial_mapping_error` 保存具体原因；报告只展示实际预处理图像和完整保留状态，不展示伪造的 Token 位置。空问题仍需在输入 JSONL 中提供 `question` 或 `text` 字符串，允许空串或空白。

`selection.indices` 为实际交给主干的原始位置；`order/trace` 是贪心尝试过程。完整回退时 `indices` 为全量，而 `order/trace` 不会伪造为已成功覆盖的选择过程。

`input_audit` 保存实际完整模板、问题 Query 位置、视觉序列位置、未删序列 position IDs 和 Token IDs、保留的原视觉/原序列索引，以及最终输入采用的连续位置。索引从 0 开始；视觉槽的 Token ID 是生成记账占位，实际输入仍是原始 V0 向量。

计时从打开图像开始，到生成结束，包含预处理、视觉编码与投影、Probe、区域/覆盖/选择及答案生成；不包含一次性模型加载、评分和报告导出。分项记录预处理、视觉编码器、Projector、文本构造、信号权重、区域生成、覆盖矩阵和贪心选择。`llm_prefill_forward` 是原生 decoder 首次前向，`llm_decode_forward` 是后续 decoder 前向的累计时间；两者不含 LM head、Token 选择及 streamer 开销，不能当作完整解码时间。只生成一枚 Token 时，后续 decoder 前向时间可能为 `null`。TTFT 包含前置选择耗时。

分项和逐 decoder 前向采用同步计时，会扰动时延；大阶段包含子阶段，不能重复相加。当前计时接口设计为无预热的单次观测，尚未执行或取得实际观测，不能作为论文级吞吐、净加速或跨硬件比较结论。

默认 `--quality-protocol exact_match`：只做 Unicode NFKC、大小写和空白归一，再与任一参考答案完整匹配；不自动提取回答中的数字、不删除标点，不采用 VQA 共识评分。没有可用参考的样本保留 `score=null` 和原因，不判成答错；汇总单列已评分分母及评分覆盖率，失败不计入准确率分母。`--quality-protocol none` 可关闭评分。该值明确标为 **DEMO exact_match，不能与 AgilePruner 论文官方准确率直接比较**。正式基准需另按数据集官方协议评分。AgilePruner 论文结果仍是用户指定的文献参照，不是本设备上复现的实测值。

## 后续验证入口

测试已编写，当前未执行。以后允许运行且依赖齐备时，可在本目录执行：

```bash
python -m pytest -q tests
```

统一使用 pytest 收集函数式 Probe 测试和其余 unittest.TestCase 测试，避免漏收集。测试包含 V2 的两组八枚人工向量算例、独立标量目标逐轮核验、跨区域限制、区域外选择、异常回退、原生首层 Probe 对照、GQA/分块一致性、实际缩短输入与连续生成位置、输入路径和报告转义等。另编写缺图或像素保护拒绝后继续处理、缺依赖可重试、输入审计、评分与汇总分母、计时与 hooks 清理测试，以及同尺寸异权重/覆盖文件拒绝、缺身份材料/坏清单/目录不可读时阻止权重加载和 CLI 失败记录测试；均未执行。模拟模型的 CLI 到报告回归与人工测试不能代替真实模型效果和净时延验证。
