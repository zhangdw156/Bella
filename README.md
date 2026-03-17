Bella BFCL MVP
==============

本项目在 `Bella/` 中实现了一个最小可运行的 BFCL 推理 + 评测闭环，完全复用 `gorilla/berkeley-function-call-leaderboard` 仓库中的官方 BFCL 数据与 evaluator，只在 Bella 侧自定义推理循环。

当前 MVP 只跑 BFCL v4 的 `simple_python` 子集的前几条样本，用于验证从：

1. `.env` 读取模型与 BFCL 配置
2. Bella 自己控制推理（调用 OpenAI-compatible endpoint）
3. 使用 BFCL 官方 handler 写结果文件
4. 使用 BFCL 官方 evaluator 打分

的完整闭环。

依赖与安装
----------

1. 安装 gorilla / BFCL（在 gorilla 仓库下）：

```bash
cd gorilla/berkeley-function-call-leaderboard
pip install -e .
```

2. 安装 Bella（在 Bella 仓库下）：

```bash
cd Bella
pip install -e .
```

3. 配置 `.env`

可以参考仓库中提供的 `.env.example`，至少需要配置：

```env
OPENAI_API_KEY=你的key
OPENAI_MODEL=gpt-4o-mini          # 或任意 OpenAI-compatible 模型名
OPENAI_BASE_URL=                  # 可选，留空则使用 SDK 默认

# 可选：自定义在 BFCL 中的实验名（影响 result/score 下的目录名）
BFCL_REGISTRY_NAME=bella-mvp

# 推荐：显式指定 BFCL 项目根目录，用于存放 result/ 与 score/
# 建议直接落在 Bella 项目下，便于查看与清理：
BFCL_PROJECT_ROOT=/home/zhang/work/Bella/outputs/bfcl
```

注意：

- `OPENAI_BASE_URL` 为空或未配置时，Bella 不会写入 `OPENAI_BASE_URL` 环境变量，OpenAI SDK 将走默认行为。
- 如需兼容 OpenAI-compatible endpoint，只需在 `.env` 中按上述方式填写 `OPENAI_BASE_URL` 和 `OPENAI_MODEL`。
- `BFCL_PROJECT_ROOT` 不配置时，BFCL 默认会把 `result/` 与 `score/` 目录写在其安装目录（例如 `.venv/site-packages`）下；
  推荐像上面示例一样显式指向 `Bella/outputs/bfcl`，方便本地调试与备份。

BFCL 模型别名兼容层
--------------------

BFCL 官方 evaluator 要求传入的 `model` 名必须存在于 `MODEL_CONFIG_MAPPING` 中，而我们在 Bella 中希望：

- 用 `.env` 中的 `BFCL_REGISTRY_NAME`（默认 `bella-mvp`）作为实验名与目录名；
- 又不去直接修改 `gorilla / BFCL` 源码。

为此，Bella 增加了一个很薄的兼容辅助函数（见 `bella/utils/bfcl_compat.py`）：

- **函数**：`ensure_bfcl_model_alias(registry_name: str, base_model: str = "gpt-4o-2024-11-20-FC")`
- **作用**：
  - 如果 `registry_name` 不在 BFCL 的 `MODEL_CONFIG_MAPPING` 中，
    则在运行期创建一个别名条目：
    - key 为 `registry_name`（如 `bella-mvp`）
    - value 直接复用 `base_model`（默认为 `gpt-4o-2024-11-20-FC`）对应的配置
  - 这样 evaluator 就能识别自定义模型名，同时：
    - **不改变** BFCL 的 handler / checker / scoring 逻辑；
    - 只影响 result/score 目录中使用的模型子目录名（与 `BFCL_REGISTRY_NAME` 保持一致）。

换句话说，这个兼容层只是一个“配置别名”，确保：

- 生成阶段：使用 `BFCL_REGISTRY_NAME` 作为 handler 的 `registry_name`，结果文件写到
  `result/<BFCL_REGISTRY_NAME>/...`；
- 评测阶段：通过 alias 让 BFCL evaluator 知道“`BFCL_REGISTRY_NAME` 也是一个合法模型名”，
  从而顺利读取刚才生成的结果文件进行打分。

最小验证脚本
------------

1. 仅运行推理（Bella 自己的 infer）：

```bash
cd Bella
python scripts/run_bfcl_infer.py --category simple_python --limit 3
```

参数说明：

- `--category`：BFCL v4 的具体子集名称，默认 `simple_python`。
- `--limit`：推理条数上限，默认 `3`。MVP 只跑前几条数据做快速验证。

脚本会：

- 从 `.env` 读取 OpenAI 配置与可选的 `BFCL_PROJECT_ROOT`、`BFCL_REGISTRY_NAME`。
- 使用 BFCL 官方的 `OpenAICompletionsHandler` 与 `BaseHandler.write()`，对指定 category 的前 N 条样本进行推理。
- 在 BFCL 的 `RESULT_PATH`（即 `BFCL_PROJECT_ROOT/result`，或 BFCL 默认 PROJECT_ROOT 下的 `result/`）中生成官方兼容的
  `BFCL_v4_<category>_result.json`（JSON Lines 文件），目录结构完全遵循 BFCL 官方实现。

2. 仅运行评测（BFCL 官方 evaluator，对 Bella 生成的结果文件打分）：

```bash
cd Bella
python scripts/run_bfcl_eval.py --category simple_python
```

参数说明：

- `--category`：BFCL v4 的具体子集名称，默认 `simple_python`。
- `--no-partial-eval`：可选，显式关闭 `partial_eval`。**MVP 不建议关闭**，原因见下文。

脚本会：

- 再次加载 `.env` 并确保 `BFCL_PROJECT_ROOT` 等环境变量就位。
- 调用 BFCL 官方的 `eval_runner.main(...)`，只使用 evaluator 逻辑，对 Bella 之前生成的 `_result.json` 文件进行评分。
- 在 BFCL 的 `SCORE_PATH`（即 `BFCL_PROJECT_ROOT/score`，或 BFCL 默认 PROJECT_ROOT 下的 `score/`）中生成
  `BFCL_v4_<category>_score.json` 以及 CSV 汇总文件。

> 关于 `partial_eval=True` 的说明
>
> 当前 Bella 的 MVP 只对指定 category 的**前 N 条样本**（例如 `limit=3`）进行推理，而不是覆盖该 category 的全部测试用例。
> 因此在调用 BFCL 官方 evaluator 时，我们默认启用 `partial_eval=True`：
>
> - 评测只会基于当前结果文件中实际存在的条目计算准确率；
> - 对于未生成的条目，evaluator 会跳过，不会视作 0 分；
> - **因此该分数是 “partial eval” 结果，不能与官方排行榜上的完整评测分数直接比较**。
>
> 如需跑完整评测，需要在后续版本中去掉 `limit` 限制，并关闭 `partial_eval`，让 BFCL 对整个 category 的所有样本都进行推理与打分。

3. 一键完整闭环 demo（推理 + 评测）：

```bash
cd Bella
python scripts/run_demo.py --category simple_python --limit 3
```

脚本会顺序执行：

1. 加载 `.env`（含 OpenAI 配置与 BFCL_PROJECT_ROOT 等）。
2. 调用 Bella 的 BFCL 推理入口，使用 BFCL 官方 handler 生成 `_result.json`。
3. 调用 BFCL 官方 evaluator，对生成的结果文件进行评测（默认 `partial_eval=True`）。
4. 在控制台打印最终的 result / score 文件路径，便于你进一步查看结果内容。

当前 MVP 的限制
---------------

- 只支持 BFCL v4 的单一 category（默认 `simple_python`），未覆盖其它 non-live / live / multi-turn / agentic 任务。
- 推理只运行前 N 条样本（默认 `limit=3`），评测启用 `partial_eval=True`，所得分数仅用于验证流程是否通畅，不代表完整榜单成绩。
- 暂未实现：
  - 复杂 agent memory 逻辑
  - 工具执行缓存 / reasoning compression
  - 多模型对比与大规模 batch evaluation

后续扩展方向
-------------

- 通过命令行参数或配置扩展到 BFCL v4 的多种 category（包括 live / multi_turn / memory / web_search）。
- 在 Bella 内部增加更灵活的模型与 pipeline 抽象，同时继续完全复用 BFCL 官方的结果格式与 evaluator。
