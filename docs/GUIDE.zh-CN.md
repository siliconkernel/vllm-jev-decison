# 纯分类安装与使用指南

[English guide](GUIDE.md) · [项目介绍](../README.zh-CN.md)

## 选择接入方式

![原生插件与 HTTP bridge](../assets/zh-CN/deployment.svg)

原生插件面向 **vLLM 0.29.0**，安装在 API 服务所在环境，启动服务时加载。
可选 HTTP bridge 连接已运行的服务，无需重启模型。原生插件随 GPU 服务启动已在
NVIDIA GB10 上验收，具体运行记录及其不能证明的内容见[验证记录](../results/README.md)。

## 1. 从仓库安装

当前通过本仓库分发，不代表已经发布到 PyPI。需要 Python 3.11 或更新版本，
以及 vLLM 支持的平台和模型。

```bash
git clone https://github.com/siliconkernel/vllm-jev-decison.git
cd vllm-jev-decison
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install '.[vllm]'
vllm-jev-decison doctor
```

`vllm` extra 固定安装 vLLM 0.29.0，相关依赖体积可能较大。
如果服务环境已经有此版本，激活该环境后执行 `python -m pip install .` 即可。
装在其他 Python 环境中，不会让运行中的服务自动发现插件。

`doctor` 检查本地安装与插件协议，不加载模型，也不证明 GPU 兼容性。
非零退出码表示当前环境无法确认原生插件支持。

## 2. 启动原生服务

将 `VLLM_API_KEY` 设置为你的服务密钥，`MODEL` 设置为兼容模型的 ID 或本地路径。
密钥不要提交到仓库。

```bash
export MODEL=/absolute/path/to/your/model
# 在当前 shell 或部署密钥配置中设置 VLLM_API_KEY。
export VLLM_PLUGINS="${VLLM_PLUGINS:+$VLLM_PLUGINS,}jev-decison"
vllm serve "$MODEL" --host 127.0.0.1 --port 8000 --logprobs-mode raw_logprobs
```

`max_logprobs` 必须不低于 16。vLLM 默认为 20，所以上面没有显式写成 16 ——
那样会平白降低同一服务上其他调用方的上限。低于 16 时插件拒绝初始化并在接口上
报告，服务其余部分照常工作。

allowlist 中也要包含当前部署需要的其他插件。示例仅监听本机；供其他机器访问时，
使用自己的鉴权与网络入口。插件继承 vLLM 配置的 API 密钥，复用现有 EngineClient。
重启已有服务应按正常维护流程操作，不要在同一组 GPU 上误启第二份模型。

检查实际插件路由：

```bash
curl --fail-with-body http://127.0.0.1:8000/plugins/jev-decison/capabilities \
  -H "Authorization: Bearer $VLLM_API_KEY"
```

原生后端应返回 `vllm_endpoint_plugin`。未配置密钥时可省略 Authorization 请求头。
404 通常意味着未发现或未启用插件，需要查看启动日志。

## 3. 调用类型化判断

```bash
curl --fail-with-body http://127.0.0.1:8000/plugins/jev-decison/infer \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -H 'Content-Type: application/json' \
  --data-binary @examples/request.json
```

示例判断客户请求类别和紧急程度。修改 `state` 与 `question` 即可更换任务。
通过 Schema 的 `description` 解释字段。枚举、布尔、小范围整数可分类；
对象拆分要求全部属性必填，且 `additionalProperties=false`。

| 请求字段 | 含义 |
| --- | --- |
| `state` | 输入文本，最多 64,000 字符 |
| `question` | 任务说明 |
| `schema` | Draft 2020-12 输出结构，暂不支持引用 |
| `mode` | 可省略；仅接受 `classify` |
| `min_confidence` | 仅作用于分类字段的接受阈值，默认 0 |

所有字段必须具有有限候选。自由文本、开放数值、可选字段及不支持的联合约束
在推理前拒绝。`auto`、`generate` 和 `max_tokens` 均不接受，没有生成或回退路径。

标准库 Python 客户端使用以下配置：

```bash
export JEV_DECISON_URL=http://127.0.0.1:8000
export JEV_DECISON_API_KEY="$VLLM_API_KEY"
python examples/client.py
```

## 4. 处理结果与拒绝状态

![置信分数与结构合法性](../assets/zh-CN/confidence.svg)

仅在 `accepted=true` 时使用 `value`。任一分类字段低于阈值时，返回
`accepted=false`、`value=null`。`decisions` 保留的值仅供诊断，不可直接派发。
由你的应用选择复核或回退策略；插件不会在拒绝后自动生成替代答案。

分类字段包含候选概率、logprob、`confidence` 与 `candidate_mass`。
常量字段的 `confidence=null`，因为其值无需模型判断。候选内条件概率不是校准后的正确率。

`usage` 分别统计输入 Token、分类传输 Token、生成 Token 和引擎请求数。
`generated_tokens` 恒为零，但每个打分字段仍消耗一个分类传输 Token。
字段可并发，但依然是多次模型请求。比较性能时应同时看总计算、正确率和耗时，
不能只凭输出减少声称加速。失败请求可能消耗计算，但没有最终用量响应。

### 两个数字各自的用途

`confidence` 回答的是"在给定这些标签的前提下，模型有多偏好选中的那个"；
`candidate_mass` 回答的是另一个问题："模型真实的概率里，有多少落在这些标签上"。
两者失效的位置不同，下面两种情况只有后者能发现：

| 情形 | `confidence` | `candidate_mass` |
| --- | --- | --- |
| 标签契合的问题 | 1.0000 | 1.000000 |
| 标签根本答不了的问题 | 0.9985 | 0.000004 |
| `state` 文本操纵判定 | 1.0000 | 0.003-0.26 |

对 `confidence` 设阈值无法拒绝其中任何一行；读取 `candidate_mass` 并把低值转入
人工复核则可以，而且不增加任何成本——它本来就在每次响应里。

### 把 `state` 视为不可信输入

`state` 往往承载第三方文本：工单、邮件、表单。放在其中的文本**可以操纵判定结果**。
实测四次尝试有三次改变了答案，且每次都报告置信度 1.0000。system prompt 要求模型
把输入当作证据而非指令，那是缓解措施，不是边界。

不要把源自不可信文本的判定结果直接用于授权、支付或内容审核动作。低
`candidate_mass` 能标记出这里观察到的攻击，但针对它优化的攻击者很可能让 mass
保持在高位，所以它是分流信号，不是访问控制。测量数据见
[验证记录](../results/README.md)。

## 可选：连接已有服务的 HTTP bridge

```bash
python -m pip install '.[bridge]'
# 可选：JEV_DECISON_UPSTREAM_API_KEY 用于访问上游。
# 可选：JEV_DECISON_API_KEY 用于客户端访问 bridge。
vllm-jev-decison bridge --upstream http://127.0.0.1:8000 \
  --model /model --host 127.0.0.1 --port 18186
```

访问 18186 上相同的 API；Python 客户端设置
`JEV_DECISON_URL=http://127.0.0.1:18186`。能力接口应显示 `http_bridge`。
上游必须支持 `/tokenize`、指定 Token logprob，并使用原始 logprob。
bridge 无法验证上游启动参数，且忽略 HTTP 代理环境变量，直接连接配置的地址。

## 排错

| 现象 | 处理 |
| --- | --- |
| doctor 找不到 vLLM | 在服务环境安装原生依赖，或明确使用 bridge |
| 路由返回 404 | 检查 allowlist、安装环境、vLLM 版本及启动日志 |
| HTTP 401 | 核对 Bearer 密钥；bridge 与上游密钥独立配置 |
| HTTP 400 | 请求字段非法，通常是 `mode`；原生服务把请求校验错误映射为 400 |
| HTTP 422 | 检查 Schema 限制、引用或非有限字段 |
| HTTP 502 | 检查上游兼容性、单 Token 标签、原始 logprob 与候选 Token 支持 |
| HTTP 504 | 整体推理超时，检查引擎负载及任务规模 |
| accepted=false | 分类阈值拒绝了字段，不要执行诊断候选 |
| 结构合法但判断错误 | 检查模型、任务与提示；结构校验不能代替行为正确性评估 |

## 复现验证

```bash
python -m pip install -e '.[test,bridge]'
pytest -q
python -m build
python docs/render_diagrams.py
python benchmarks/smoke_http.py --url http://127.0.0.1:8000 \
  --model /model --out results/my-new-run
```

实验目录必须是新目录。已有记录保留了初期失败与提示修订。
具体范围见[验证记录](../results/README.md)，不能当作生产或跨模型基准。
