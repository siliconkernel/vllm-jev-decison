"""Regenerate the bilingual, self-contained SVG documentation diagrams."""
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'assets'
COLORS = {'ink': '#172e43', 'muted': '#506478', 'teal': '#007f79', 'orange': '#b36718'}


def render(name, language, title, subtitle, boxes, edges, note):
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="610" viewBox="0 0 1200 610" role="img">',
             f'<title>{escape(title)}</title><desc>{escape(subtitle)}</desc>',
             '<defs><marker id="arrow" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0 0 L10 4 L0 8" fill="#506478"/></marker></defs>',
             '<rect width="1200" height="610" rx="18" fill="#f7fafc"/>',
             '<g font-family="Arial, Noto Sans CJK SC, Microsoft YaHei, sans-serif">']
    def text(x, y, content, size=20, color='ink', weight='normal', anchor='start'):
        parts.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{COLORS[color]}">{escape(content)}</text>')
    text(42, 58, title, 30, weight='bold')
    text(42, 94, subtitle, 18, 'muted')
    for x1, y1, x2, y2 in edges:
        parts.append(f'<path d="M{x1} {y1} L{x2} {y2}" stroke="#506478" stroke-width="2.5" fill="none" marker-end="url(#arrow)"/>')
    for x, y, width, height, title_, lines, color in boxes:
        parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="12" fill="white" stroke="{COLORS[color]}" stroke-width="2"/>')
        text(x + 20, y + 35, title_, 21, color, 'bold')
        for index, line in enumerate(lines):
            text(x + 20, y + 70 + index * 29, line, 18)
    for index, line in enumerate(note):
        text(42, 551 + index * 26, line, 17, 'muted')
    parts.append('</g></svg>')
    (ROOT / language / f'{name}.svg').write_text('\n'.join(parts))


for language in ('en', 'zh-CN'):
    zh = language == 'zh-CN'
    def tr(en, cn): return cn if zh else en
    render('architecture', language, tr('Classification only: schema to typed values', '纯分类：从 Schema 到类型值'),
        tr('No free-form generation. Unsupported schemas fail before model inference.', '不生成自由内容，不支持的 Schema 在模型推理前拒绝。'), [
        (40, 220, 240, 150, tr('Request', '请求'), ['state + question', tr('Finite JSON Schema', '有限候选 JSON Schema')], 'ink'),
        (330, 220, 260, 150, tr('Validate and plan', '验证与规划'), [tr('Check finite domains', '检查有限候选域'), tr('Split required fields', '拆分必填字段')], 'ink'),
        (650, 145, 510, 150, tr('Score → select → construct', '打分 → 选择 → 构造'), [tr('One-step label scores; candidate argmax', '读取单步标签分数，选择候选最高分'), tr('Construct typed values and validate the schema', '程序构造类型值，校验完整 Schema')], 'teal'),
        (650, 350, 510, 150, tr('Unsupported → HTTP 422', '不支持的结构 → HTTP 422'), [tr('Free text, open numbers, unsupported constraints', '自由文本、开放数值、不支持的约束'), tr('No inference and no generative fallback', '不执行推理，也不回退到生成')], 'orange')],
        [(280, 290, 330, 290), (590, 260, 650, 220), (590, 330, 650, 415)],
        [tr('Constants need no model call. Multiple finite fields use separate requests that vLLM may batch.', '常量无需模型调用。多个有限字段对应独立请求，可由 vLLM 批处理。'),
         tr('Ordinary sampler: one classification transport token per scored field; no free-form output decoding.', '普通 sampler 每个打分字段产生一个分类传输 Token，不进行自由内容解码。')])
    render('deployment', language, tr('Native plugin and optional HTTP bridge', '原生插件与可选 HTTP bridge'),
        tr('Identical request schema; different integration and validation boundaries.', '相同的请求 Schema，不同的接入方式与验证边界。'), [
        (40, 150, 260, 145, tr('Client', '客户端'), [tr('HTTP request', '发送 HTTP 请求'), '/plugins/jev-decison/infer'], 'ink'),
        (360, 150, 400, 145, tr('vLLM API process', 'vLLM API 进程'), [tr('EndpointPlugin: jev-decison', 'EndpointPlugin：jev-decison'), tr('Direct EngineClient access', '直接调用 EngineClient')], 'teal'),
        (850, 150, 310, 145, tr('Existing engine', '已有推理引擎'), [tr('One loaded model', '复用同一份模型'), tr('No source patch', '无需源码补丁')], 'teal'),
        (40, 350, 260, 145, tr('Client', '客户端'), [tr('Same HTTP API', '相同 HTTP API'), tr('Separate bridge process', '独立 bridge 进程')], 'ink'),
        (360, 350, 400, 145, tr('HTTP compatibility bridge', 'HTTP 兼容桥接'), ['tokenize + completions', tr('No server restart for testing', '测试无需重启已有服务')], 'orange'),
        (850, 350, 310, 145, tr('Existing vLLM API', '已有 vLLM API'), [tr('Existing model server', '已有模型服务'), tr('Not native plugin loading', '不涉及原生插件加载')], 'orange')],
        [(300, 222, 360, 222), (760, 222, 850, 222), (300, 422, 360, 422), (760, 422, 850, 422)],
        [tr('Native target: vLLM 0.29.0. GPU plugin startup remains unverified.', '原生目标版本：vLLM 0.29.0。GPU 服务中的插件启动尚未验收。'),
         tr('Live DeepSeek smoke records exercise the bridge only; they are not native deployment evidence.', '真实 DeepSeek 冒烟记录仅验证 bridge，不代表原生插件已部署验收。')])
    render('confidence', language, tr('Read confidence and validity separately', '分别理解置信分数与结构合法性'),
        tr('Illustrative numbers below are arithmetic examples, not measured model results.', '以下数字为算术示例，不是模型实测结果。'), [
        (40, 175, 335, 290, tr('Raw label probability', '原始标签概率'), ['P(A) = 0.001', 'P(B) = 0.009', tr('Other vocabulary = 0.990', '其余词表概率 = 0.990'), 'candidate_mass = 0.010'], 'ink'),
        (425, 175, 350, 290, tr('Conditional candidates', '候选内条件概率'), ['P(A | A,B) = 0.10', 'P(B | A,B) = 0.90', tr('0.95 threshold → abstain', '阈值 0.95 → 拒绝'), tr('0.80 threshold → accept B', '阈值 0.80 → 接受 B')], 'teal'),
        (825, 175, 335, 290, tr('Separate checks', '仍需独立验证'), [tr('Schema-valid ≠ task-correct', '结构合法 ≠ 任务正确'), tr('0.90 ≠ 90% correctness', '0.90 ≠ 90% 正确率'), tr('Constants need no model score', '常量无需模型打分'), tr('Threshold gates classification', '阈值仅约束分类字段')], 'orange')],
        [(375, 315, 425, 315), (775, 315, 825, 315)],
        [tr('If any classified field fails the threshold: accepted=false and value=null.', '任一分类字段低于阈值：accepted=false，value=null。'),
         tr('Diagnostic proposals in decisions must not be treated as accepted results.', 'decisions 中的诊断候选不能当作已接受结果使用。')])
print('Rendered 3 diagrams in English and Chinese.')
