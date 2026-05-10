# Bella Benchmark 评测报告：多模型对比（提示词重构后）

## 实验概述

- **日期**: 2026-05-10
- **Case 数量**: 125
- **User 模型**: gpt-5.2（所有实验共用）
- **并发度**: 80（GPT 5.4 使用 Azure pool，并发度 12）
- **提示词变更**: 本次实验使用重构后的系统提示词，ReactAgent 按 interaction_mode 分为 fixed/dynamic 两版，UserAgent 去客服化并改进 [DONE] 判定逻辑

### 模型配置

| 模型 | 协议 | Endpoint | 重复次数 (n) |
|------|------|----------|:---:|
| GPT 5.2 | OpenAI Chat Completions | Copilot Pool (localhost:5152) | 8 |
| GPT 5.4 | Azure OpenAI Chat Completions | CloudGPT (azure) | 4 |
| Claude Opus 4.5 | Anthropic 原生 | Copilot Pool (localhost:5152) | 8 |
| Claude Opus 4.6 | Anthropic 原生 | Copilot Pool (localhost:5152) | 8 |
| Claude Opus 4.7 | Anthropic 原生 | Copilot Pool (localhost:5152) | 8 |

### 提示词重构要点

1. **ReactAgent fixed 模式**：强自主性，禁止提问，自主错误恢复，主动推进所有子步骤
2. **ReactAgent dynamic 模式**：可交互，信息不足时简短提问，遵循确认流程，完成子任务后报告结果
3. **共用 common block**：tool-use policy、grounding policy、output policy
4. **Domain policy 过渡句**：明确 domain policy 优先级高于 base rules
5. **UserAgent**：去客服绑定（"AI assistant" 替代 "customer service assistant"），精确 [DONE] 判定，增加纠错和确认行为

---

## 一、总体结果

| 指标 | GPT 5.2 (n=8) | GPT 5.4 (n=4) | Opus 4.5 (n=8) | Opus 4.6 (n=8) | Opus 4.7 (n=8) |
|------|:---:|:---:|:---:|:---:|:---:|
| **pass@1** | 0.747 | 0.790 | 0.850 | **0.877** | 0.836 |
| **pass@n** | 0.928 | 0.928 | 0.944 | 0.944 | **0.968** |
| **pass^n** | 0.528 | 0.616 | 0.648 | **0.736** | 0.632 |

> 注：GPT 5.4 的 n=4，其余为 n=8，pass@n 和 pass^n 的 n 值不同，跨模型对比时应以 pass@1 为主。

### 排名（按 pass@1）

1. **Opus 4.6** — 0.877
2. Opus 4.5 — 0.850
3. Opus 4.7 — 0.836
4. GPT 5.4 — 0.790
5. GPT 5.2 — 0.747

---

## 二、按 Category 分析

### pass@1

| Category | Cases | GPT 5.2 | GPT 5.4 | Opus 4.5 | Opus 4.6 | Opus 4.7 |
|----------|:-----:|:-------:|:-------:|:--------:|:--------:|:--------:|
| bfclv4_filesystem | 8 | 0.781 | 0.875 | 0.875 | 0.875 | 0.859 |
| bfclv4_trading | 8 | 0.891 | **1.000** | **1.000** | **1.000** | **1.000** |
| bfclv4_travel | 5 | 0.975 | 0.850 | **1.000** | **1.000** | **1.000** |
| bfclv4_vehicle | 15 | 0.492 | 0.750 | **0.992** | 0.967 | 0.900 |
| mcpmark_postgres | 14 | 0.857 | **0.893** | 0.902 | 0.804 | 0.875 |
| tau3_airline | 20 | 0.669 | 0.613 | 0.694 | **0.812** | 0.681 |
| tau3_retail | 55 | 0.770 | 0.791 | 0.816 | **0.866** | 0.823 |

### pass@n

| Category | GPT 5.2 | GPT 5.4 | Opus 4.5 | Opus 4.6 | Opus 4.7 |
|----------|:-------:|:-------:|:--------:|:--------:|:--------:|
| bfclv4_filesystem | **1.000** | 0.875 | 0.875 | 0.875 | **1.000** |
| bfclv4_trading | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| bfclv4_travel | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| bfclv4_vehicle | 0.733 | 0.867 | **1.000** | **1.000** | **1.000** |
| mcpmark_postgres | **1.000** | **1.000** | **1.000** | 0.929 | 0.929 |
| tau3_airline | **1.000** | 0.800 | 0.900 | 0.900 | 0.900 |
| tau3_retail | 0.909 | 0.964 | 0.927 | 0.945 | **0.982** |

### pass^n

| Category | GPT 5.2 | GPT 5.4 | Opus 4.5 | Opus 4.6 | Opus 4.7 |
|----------|:-------:|:-------:|:--------:|:--------:|:--------:|
| bfclv4_filesystem | 0.500 | **0.875** | **0.875** | **0.875** | 0.750 |
| bfclv4_trading | 0.750 | **1.000** | **1.000** | **1.000** | **1.000** |
| bfclv4_travel | 0.800 | 0.600 | **1.000** | **1.000** | **1.000** |
| bfclv4_vehicle | 0.333 | 0.600 | **0.933** | **0.933** | 0.800 |
| mcpmark_postgres | 0.714 | **0.786** | 0.714 | 0.571 | **0.786** |
| tau3_airline | 0.350 | 0.350 | 0.400 | **0.600** | 0.400 |
| tau3_retail | 0.545 | 0.582 | 0.527 | **0.691** | 0.527 |

---

## 三、关键发现

### 1. Opus 4.6 综合最强

Opus 4.6 在 pass@1（0.877）和 pass^n（0.736）上均为最高。优势主要来自 tau3 场景（复杂 domain policy 遵循），tau3_airline 的 pass@1 达到 0.812，显著领先其他模型（第二名 Opus 4.5 仅 0.694）。

### 2. bfclv4_vehicle 是 GPT 系列的短板

GPT 5.2 在 vehicle 场景仅 0.492，GPT 5.4 提升到 0.750，但 Opus 系列全线 0.9+（Opus 4.5 达到 0.992）。Vehicle 场景需要模型自主发现并执行前置条件（如启动引擎前锁门、踩刹车），这是 fixed 模式下的自主错误恢复能力测试。GPT 系列在这方面明显弱于 Opus。

### 3. mcpmark_postgres 是 Opus 4.6 的相对弱项

Opus 4.6 在 postgres 场景的 pass@1 仅 0.804，低于其他四个模型。pass^n 也是最低（0.571）。这可能与 SQL 生成的随机性有关，或该模型在长指令单轮任务上的表现不如多轮交互场景。

### 4. GPT 5.4 vs 5.2 基本持平

GPT 5.4 总体 pass@1（0.790）略高于 GPT 5.2（0.747），但两者在不同 category 上互有胜负。5.4 在 vehicle（+0.258）和 trading（+0.109）上提升明显，但在 travel（-0.125）和 airline（-0.056）上下降。

### 5. Opus 4.7 不如 4.6

Opus 4.7 的 pass@1（0.836）低于 4.6（0.877），主要差距在 tau3_airline（0.681 vs 0.812）和 tau3_retail（0.823 vs 0.866）。但 Opus 4.7 的 pass@n 最高（0.968），说明它的上限更高但一致性不如 4.6。

---

## 四、与前次实验的对比

前次实验（eval_report_20260510.md）使用旧版单一系统提示词，case 数量为 148（含饱和 case），仅测试了 GPT 5.2 和 Opus 4.5。

| 指标 | GPT 5.2 (旧) | GPT 5.2 (新) | Opus 4.5 (旧) | Opus 4.5 (新) |
|------|:---:|:---:|:---:|:---:|
| **pass@1** | 0.613 | 0.747 (+13.4pp) | 0.720 | 0.850 (+13.0pp) |

> 注意：case 集不同（旧 148 含饱和 case，新 125 已移除），所以不能直接归因于提示词改进。但两个模型都有约 13pp 的提升，部分来自移除饱和 case 提高了 base rate，部分可能来自提示词改进。

---

## 五、实验配置备注

- GPT 5.4 通过 Azure OpenAI (CloudGPT) 调用，模型 ID 为 `gpt-5.4-20260305`，使用 Azure AD 认证的 credential pool
- GPT 5.5 未测试：Azure 上无此模型，Copilot Pool 不支持 Chat Completions（返回 400）
- 所有 Opus 模型通过 Copilot Pool 使用 Anthropic 原生协议调用
- 结果目录：`results/eval_all_*` 和 `results/eval_claude_opus_4_*`
