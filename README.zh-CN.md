# vllm-jev-decison

**给兼容的 vLLM 语言模型增加“判断 + Schema”推理 API。**

[English](README.md)

[安装与使用指南](docs/GUIDE.zh-CN.md) · [示例](examples) · [验证记录](results/README.md)

输入文本、问题和 JSON Schema。枚举、布尔、小范围整数通过候选分数选择；
自由字段通过 Schema 约束生成；程序组装并校验最终对象。
无需新增分类头、训练权重或修改 vLLM 源码。

这是类似 Jev 的接口形式，不代表复现其内部模型，也不保证相同的准确率与速度。
“兼容模型可接入”不等于“任意模型都能可靠判断”。

![Architecture / 架构](assets/zh-CN/architecture.svg)

## 安装与启动

目标版本为 **vLLM 0.29.0**，在 API 服务所在环境安装：

```bash
git clone https://github.com/siliconkernel/vllm-jev-decison.git
cd vllm-jev-decison
pip install '.[vllm]'
vllm-jev-decison doctor
export VLLM_PLUGINS="${VLLM_PLUGINS:+$VLLM_PLUGINS,}jev-decison"
vllm serve YOUR_MODEL --logprobs-mode raw_logprobs --max-logprobs 16
```

已有 vLLM 0.29.0 时只需 `pip install .`。已有其他插件时，将它们也加入 allowlist。
通过 `VLLM_API_KEY` 或 vLLM 的 `--api-key` 配置鉴权；插件自行校验其路由的密钥。
插件使用正式的 [EndpointPlugin 接口](https://docs.vllm.ai/en/v0.29.0/design/endpoint_plugins/)，
复用已有 EngineClient，不加载第二份模型。

## 调用

```bash
curl http://localhost:8000/plugins/jev-decison/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "state":"客户说：信用卡重复扣款，请紧急退款。",
    "question":"判断请求类别及是否紧急。",
    "schema":{
      "type":"object",
      "properties":{
        "category":{"enum":["账单","技术","其他"]},
        "urgent":{"type":"boolean"}
      },
      "required":["category","urgent"],
      "additionalProperties":false
    },
    "mode":"classify",
    "min_confidence":0.8
  }'
```

启用鉴权时追加 `-H "Authorization: Bearer $VLLM_API_KEY"`。
可复用请求与 Python 客户端见 [examples](examples)。

返回 `accepted`、`value`、逐字段 `decisions` 及 `usage`。
任何分类字段低于阈值时，`accepted=false`、`value=null`；`decisions` 中的候选仅供诊断。
用量分别统计输入 Token、分类传输 Token、生成 Token 与引擎请求数。
截断或不符合 Schema 的生成不会作为完整结果返回。
`GET /plugins/jev-decison/capabilities` 可查看后端和能力范围。

## 支持范围

| 模式 | 有限候选字段 | 非有限字段 |
| --- | --- | --- |
| `classify` | 候选打分 | 拒绝请求 |
| `auto` | 候选打分 | Schema 约束生成 |
| `generate` | Schema 约束生成 | Schema 约束生成 |

有限候选包括最多 16 项的枚举、布尔、null、常量以及跨度不超过 16 的整数区间。
所有字段必填且禁止额外属性的简单对象可递归拆分；可选字段、数组、条件与跨字段约束
在 auto 模式下回到整个子结构的约束生成。最终还会验证完整根 Schema。

最多 32 个字段、每字段 16 个候选、64,000 字符输入、32 KB Schema、20 层嵌套、
每个生成字段最多 4,096 Token。使用 Draft 2020-12；v0.1 不支持 Schema 引用，
`format` 仅作为注解。复杂语法是否可生成还取决于 vLLM 的 grammar backend。

## 推理机制与概率含义

程序将候选映射到经过 tokenizer 检查的单 Token 标签 A–P，读取下一 Token 的
候选 logprob，选择最高分并还原类型值。类别无需依赖生成文本解析。

当前通用后端仍使用 vLLM 普通 sampler 的 `max_tokens=1`；其输出虽不用于解释类别，
仍计为一个分类传输 Token。它不是 ClassWeave 中绕过 sampler、保留 KV 续写的专用补丁。
多个字段对应多个引擎请求，每个 API 进程最多同时执行八个字段请求；可以由 vLLM 批处理，
但不能说“一个 forward 回答所有问题”。目前只有普通前缀缓存，没有会话内 KV 融合。

概率只在给定标签集合内归一化，不是校准后的正确率。`candidate_mass` 单独报告标签占
整个词表的原始概率质量。阈值仅作用于分类字段，生成字段与常量没有置信度估计。
标签、顺序、模型训练、输入分布都会影响判断。Schema 正确不代表语义正确。

![Confidence / 概率与结构边界](assets/zh-CN/confidence.svg)

## 验证状态

面向支持文本生成、单 Token 标签、原始指定 Token logprob 的 vLLM 模型。
优先使用指令模型；没有 chat template 的基础模型使用普通文本提示，不保证质量。
Pooling 模型、多模态、LoRA 路由、投机解码交互及所有 tokenizer 组合尚未验证。

接口已核对 vLLM 0.29.0 源码；本地测试覆盖 Schema 规划、鉴权、拒绝状态、用量、取消，
以及模拟 EngineClient 的适配契约。**模拟测试不等于真实 GPU 插件启动验收。**
真实模型的 HTTP bridge 记录见 [results](results/README.md)，也不能冒充原生插件部署证据。

减少输出可能降低 Decode 工作量，但 prefill、字段拆分、排队和生成开销仍然存在。
目前不宣称普遍加速、生产可用或正确率保证。

![Deployment / 部署方式](assets/zh-CN/deployment.svg)

## 不重启现有引擎的桥接测试

```bash
pip install '.[bridge]'
vllm-jev-decison bridge --upstream http://127.0.0.1:8000 --model /model --port 18186
```

访问 18186 上相同的 API。此模式明确返回 `backend=http_bridge`，不是引擎内插件。
上游必须支持 `/tokenize`、指定 Token logprob、约束输出，并配置 `raw_logprobs`；
bridge 无法验证上游启动参数。密钥分别使用 `JEV_DECISON_UPSTREAM_API_KEY` 与
`JEV_DECISON_API_KEY`。默认监听本机，忽略 HTTP 代理环境变量，直接连接配置的上游。

## 开发

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[test,bridge]'
pytest -q
python -m build
```

实验输出必须使用新目录，保留失败记录。采用 MIT 许可；vLLM 保留自身许可。
