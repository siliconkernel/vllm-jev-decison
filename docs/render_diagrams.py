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
    render('architecture', language, tr('Typed decisions, two inference paths', '类型化判断，两条推理路径'),
        tr('Same model weights. Runtime-defined questions and output schemas.', '复用模型权重，在请求时定义问题与输出 Schema。'), [
        (40, 205, 240, 150, tr('Request', '请求'), ['state + question', 'JSON Schema'], 'ink'),
        (330, 145, 370, 150, tr('Finite fields → scoring', '有限字段 → 候选打分'),
         [tr('enum / boolean / bounded integer', '枚举 / 布尔 / 小范围整数'), tr('Read label logprobs; select argmax', '读取标签 logprob，选择最高分')], 'teal'),
        (330, 340, 370, 150, tr('Other fields → generation', '其他字段 → 约束生成'),
         [tr('Free text or joint constraints', '自由文本或联合约束'), tr('Generate a schema-constrained value', '生成符合 Schema 的类型值')], 'orange'),
        (765, 205, 395, 205, tr('Assemble and validate', '组装与校验'),
         [tr('Validate the complete root schema', '校验完整根 Schema'), tr('Return accepted value or abstention', '返回接受的结果或拒绝状态'), tr('Report per-field scores and usage', '报告逐字段分数与用量')], 'ink')],
        [(280, 250, 330, 220), (280, 310, 330, 410), (700, 220, 765, 260), (700, 415, 765, 355)],
        [tr('Constants need no model call. Multiple fields use multiple engine requests, which vLLM may batch.', '常量无需模型调用。多个字段对应多个引擎请求，可由 vLLM 批处理。'),
         tr('Classification still uses the ordinary sampler with max_tokens=1; this is not a sampler-bypass patch.', '分类仍使用普通 sampler 的 max_tokens=1，不是绕过采样器的专用补丁。')])
    render('deployment', language, tr('Native plugin and optional HTTP bridge', '原生插件与可选 HTTP bridge'),
        tr('Identical request schema; different integration and validation boundaries.', '相同的请求 Schema，不同的接入方式与验证边界。'), [
        (40, 150, 260, 145, tr('Client', '客户端'), [tr('HTTP request', '发送 HTTP 请求'), '/plugins/decision/infer'], 'ink'),
        (360, 150, 400, 145, tr('vLLM API process', 'vLLM API 进程'), [tr('EndpointPlugin: decision', 'EndpointPlugin：decision'), tr('Direct EngineClient access', '直接调用 EngineClient')], 'teal'),
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
        (825, 175, 335, 290, tr('Separate checks', '仍需独立验证'), [tr('Schema-valid ≠ task-correct', '结构合法 ≠ 任务正确'), tr('0.90 ≠ 90% correctness', '0.90 ≠ 90% 正确率'), tr('No score for generated fields', '生成字段没有置信分数'), tr('Threshold gates classification', '阈值仅约束分类字段')], 'orange')],
        [(375, 315, 425, 315), (775, 315, 825, 315)],
        [tr('If any classified field fails the threshold: accepted=false and value=null.', '任一分类字段低于阈值：accepted=false，value=null。'),
         tr('Diagnostic proposals in decisions must not be treated as accepted results.', 'decisions 中的诊断候选不能当作已接受结果使用。')])
print('Rendered 3 diagrams in English and Chinese.')
