const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const FIG = path.join(__dirname, "figures");
const OUT = path.join(__dirname, "defense");
fs.mkdirSync(OUT, { recursive: true });

const PPTX = path.join(OUT, "本科毕设答辩_实验完整版_大字号_KAN超图药物协同预测.pptx");
const PDF = path.join(OUT, "本科毕设答辩_实验完整版_大字号_KAN超图药物协同预测.pdf");
const NOTES = path.join(OUT, "本科毕设答辩_实验完整版_大字号_讲稿.md");

const C = {
  bg: "F7FAFC",
  ink: "172A3A",
  muted: "526777",
  line: "D9E3EA",
  white: "FFFFFF",
  navy: "0E2B3F",
  teal: "0B8F8A",
  mint: "DDF4F1",
  amber: "F2A65A",
  coral: "D95763",
  blue: "4F8FCB",
  green: "2E8B57",
  pale: "EEF7F7",
};

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "Codex";
pptx.title = "基于KAN超图的药物协同预测研究";
pptx.subject = "本科毕业论文答辩实验完整版";
pptx.lang = "zh-CN";
pptx.theme = {
  headFontFace: "Microsoft YaHei",
  bodyFontFace: "Microsoft YaHei",
  lang: "zh-CN",
};
pptx.defineLayout({ name: "LAYOUT_WIDE", width: 13.333, height: 7.5 });

function shadow() {
  return { type: "outer", color: "071820", opacity: 0.12, blur: 2, offset: 1, angle: 45 };
}

function tx(slide, text, x, y, w, h, opt = {}) {
  slide.addText(text, {
    x, y, w, h,
    margin: opt.margin ?? 0.02,
    fontFace: "Microsoft YaHei",
    fontSize: opt.fontSize ?? 18,
    bold: opt.bold ?? false,
    color: opt.color ?? C.ink,
    align: opt.align ?? "left",
    valign: opt.valign ?? "top",
    fit: "shrink",
    breakLine: opt.breakLine,
  });
}

function bg(slide, color = C.bg) {
  slide.background = { color };
}

function title(slide, main, sub = "") {
  tx(slide, main, 0.55, 0.32, 10.9, 0.55, { fontSize: 27, bold: true, color: C.ink });
  if (sub) tx(slide, sub, 0.58, 0.92, 10.8, 0.35, { fontSize: 14, color: C.muted });
  slide.addShape(pptx.ShapeType.arc, {
    x: 12.05, y: 0.12, w: 0.8, h: 0.8,
    line: { color: C.mint, transparency: 25, width: 8 },
    adjustPoint: 0.42,
  });
}

function footer(slide, n) {
  tx(slide, String(n).padStart(2, "0"), 12.32, 7.02, 0.42, 0.2, { fontSize: 10, color: C.muted, align: "right" });
}

function panel(slide, x, y, w, h, opt = {}) {
  slide.addShape(pptx.ShapeType.rect, {
    x, y, w, h,
    fill: { color: opt.fill ?? C.white },
    line: { color: opt.line ?? C.line, width: 0.7 },
    shadow: opt.shadow === false ? undefined : shadow(),
  });
  if (opt.accent) {
    slide.addShape(pptx.ShapeType.rect, { x, y, w: 0.09, h, fill: { color: opt.accent }, line: { color: opt.accent } });
  }
}

function bulletList(slide, lines, x, y, w, h, opt = {}) {
  slide.addText(lines.map((line, idx) => ({
    text: line,
    options: { bullet: true, breakLine: idx !== lines.length - 1 },
  })), {
    x, y, w, h,
    margin: 0.02,
    fontFace: "Microsoft YaHei",
    fontSize: opt.fontSize ?? 17,
    color: opt.color ?? C.ink,
    valign: "top",
    fit: "shrink",
    paraSpaceAfterPt: opt.paraSpaceAfterPt ?? 8,
  });
}

function stat(slide, value, label, x, y, w, color, opt = {}) {
  panel(slide, x, y, w, opt.h ?? 1.15, { accent: color, fill: opt.fill ?? C.white, shadow: opt.shadow });
  tx(slide, value, x + 0.14, y + 0.16, w - 0.28, 0.38, { fontSize: opt.valueSize ?? 24, bold: true, color, align: "center" });
  tx(slide, label, x + 0.12, y + 0.68, w - 0.24, 0.26, { fontSize: opt.labelSize ?? 12, color: C.muted, align: "center" });
}

function image(slide, file, x, y, w, h) {
  slide.addImage({ path: file, x, y, w, h, sizing: { type: "contain", w, h } });
}

function table(slide, rows, x, y, w, h, colW, fontSize = 13) {
  const headerFill = C.navy;
  const data = rows.map((row, r) => row.map((v) => ({
    text: String(v),
    options: r === 0
      ? { bold: true, color: C.white, fill: { color: headerFill } }
      : { color: C.ink, fill: { color: C.white } },
  })));
  slide.addTable(data, {
    x, y, w, h, colW,
    fontFace: "Microsoft YaHei",
    fontSize,
    border: { pt: 0.7, color: C.line },
    margin: 0.06,
    valign: "mid",
    align: "center",
    fit: "shrink",
  });
}

function notes(slide, text) {
  slide.addNotes(Array.isArray(text) ? text.join("\n") : text);
}

function bar(slide, label, value, max, x, y, w, color, opt = {}) {
  const h = opt.h ?? 0.34;
  tx(slide, label, x, y - 0.03, 2.0, 0.28, { fontSize: opt.fontSize ?? 13, color: C.ink });
  slide.addShape(pptx.ShapeType.rect, { x: x + 2.0, y, w, h, fill: { color: "EDF2F5" }, line: { color: "EDF2F5" } });
  slide.addShape(pptx.ShapeType.rect, { x: x + 2.0, y, w: w * (value / max), h, fill: { color }, line: { color } });
  tx(slide, opt.format ? opt.format(value) : String(value), x + 2.08 + w, y - 0.01, 0.8, 0.28, { fontSize: opt.fontSize ?? 13, bold: true, color });
}

function metricBars(slide, titleText, metrics, x, y, w, h, higherBetter = true) {
  panel(slide, x, y, w, h, { accent: higherBetter ? C.teal : C.coral });
  tx(slide, titleText, x + 0.25, y + 0.2, w - 0.5, 0.32, { fontSize: 17, bold: true, color: higherBetter ? C.teal : C.coral });
  const max = Math.max(...metrics.map((m) => m.value)) * 1.08;
  metrics.forEach((m, i) => {
    bar(slide, m.label, m.value, max, x + 0.3, y + 0.78 + i * 0.52, w - 3.1, m.color, {
      format: (v) => m.fmt ?? v.toFixed(4),
      fontSize: 13.5,
    });
  });
}

function simpleLine(slide, x1, y1, x2, y2, color = C.teal) {
  slide.addShape(pptx.ShapeType.line, {
    x: x1, y: y1, w: x2 - x1, h: y2 - y1,
    line: { color, width: 2, beginArrowType: "none", endArrowType: "triangle" },
  });
}

// 1 cover
{
  const s = pptx.addSlide();
  bg(s, C.navy);
  s.addShape(pptx.ShapeType.rect, { x: 0, y: 0, w: 4.25, h: 7.5, fill: { color: C.deep ?? C.teal }, line: { color: C.teal } });
  s.addShape(pptx.ShapeType.rect, { x: 0, y: 0, w: 4.25, h: 7.5, fill: { color: "164D59" }, line: { color: "164D59" } });
  s.addShape(pptx.ShapeType.arc, { x: 8.8, y: -0.35, w: 3.2, h: 3.2, line: { color: C.teal, transparency: 5, width: 16 }, adjustPoint: 0.26 });
  tx(s, "实验完整版 · 大字号", 0.72, 0.72, 2.3, 0.35, { fontSize: 15, bold: true, color: C.mint, align: "center" });
  tx(s, "基于KAN超图的\n药物协同预测研究", 0.72, 1.72, 8.4, 1.6, { fontSize: 36, bold: true, color: C.white });
  tx(s, "本科毕业论文答辩", 0.78, 3.6, 3.2, 0.4, { fontSize: 18, bold: true, color: C.amber });
  stat(s, "10页", "含完整实验链条", 8.1, 2.15, 2.0, C.teal, { valueSize: 24 });
  stat(s, "16万+", "DrugComb样本", 10.45, 2.15, 2.0, C.amber, { valueSize: 24 });
  stat(s, "0", "cold-drug泄漏", 8.1, 3.6, 2.0, C.coral, { valueSize: 24 });
  stat(s, "HgKAN-Agg", "主结果模型", 10.45, 3.6, 2.0, C.teal, { valueSize: 20 });
  tx(s, "答辩人：________    指导教师：________    专业：________", 0.78, 6.55, 7.5, 0.32, { fontSize: 14, color: "C7D9E2" });
  notes(s, "各位老师好，本版PPT把实验链条展开，并统一放大字号，便于投屏答辩。");
}

// 2 problem contributions
{
  const s = pptx.addSlide();
  bg(s);
  title(s, "研究问题与贡献", "本文关注KAN在药物协同预测中的有效放置方式，而不是简单比较模型名称");
  panel(s, 0.7, 1.58, 3.75, 4.9, { accent: C.coral });
  tx(s, "任务难点", 1.02, 1.9, 2.2, 0.35, { fontSize: 21, bold: true, color: C.coral });
  bulletList(s, [
    "药物组合空间快速增长",
    "协同正例少，AUPR更关键",
    "random split易高估泛化",
    "需要可解释的结构线索",
  ], 1.05, 2.55, 2.9, 2.5, { fontSize: 18 });
  panel(s, 4.8, 1.58, 3.75, 4.9, { accent: C.teal });
  tx(s, "核心问题", 5.12, 1.9, 2.2, 0.35, { fontSize: 21, bold: true, color: C.teal });
  bulletList(s, [
    "KAN能否直接替换MLP？",
    "DrugKAN是否提升单药图表示？",
    "交互层结构是否更重要？",
    "模型是否依赖药物对信息？",
  ], 5.15, 2.55, 2.9, 2.5, { fontSize: 18 });
  panel(s, 8.9, 1.58, 3.75, 4.9, { fill: C.navy, line: C.navy, shadow: false });
  tx(s, "本文贡献", 9.22, 1.9, 2.2, 0.35, { fontSize: 21, bold: true, color: C.amber });
  bulletList(s, [
    "构建16万级药物协同数据",
    "系统消融KAN放置位置",
    "验证HgKAN-Agg在cold-drug中的趋势",
    "建立扰动 + saliency解释流程",
  ], 9.25, 2.55, 2.85, 2.7, { fontSize: 17, color: C.white });
  footer(s, 2);
  notes(s, "这页用于告诉老师：本文不是只做一个模型，而是在系统比较KAN的放置位置。");
}

// 3 data protocol
{
  const s = pptx.addSlide();
  bg(s);
  title(s, "数据与评价协议", "实验覆盖分类、回归、random插值和cold-drug未见药物泛化");
  stat(s, "160,228", "样本", 0.75, 1.55, 2.1, C.teal);
  stat(s, "1,896", "药物", 3.25, 1.55, 2.1, C.navy);
  stat(s, "156", "细胞系", 5.75, 1.55, 2.1, C.amber);
  stat(s, "14.8%", "正例比例", 8.25, 1.55, 2.1, C.coral);
  stat(s, "0", "药物泄漏", 10.75, 1.55, 1.55, C.green);
  panel(s, 0.8, 3.25, 5.55, 2.6, { accent: C.teal });
  tx(s, "输入与任务", 1.1, 3.55, 2.3, 0.32, { fontSize: 21, bold: true, color: C.teal });
  bulletList(s, [
    "输入：药物A图 + 药物B图 + 细胞表达",
    "分类：AUPR / AUC",
    "回归：RMSE / MAE",
  ], 1.12, 4.18, 4.5, 1.2, { fontSize: 18 });
  panel(s, 6.9, 3.25, 5.55, 2.6, { accent: C.navy });
  tx(s, "划分设计", 7.2, 3.55, 2.3, 0.32, { fontSize: 21, bold: true, color: C.navy });
  bulletList(s, [
    "random：已有药物空间内插值",
    "cold-drug：测试药物未出现在训练集中",
    "主结论更重视cold-drug泛化",
  ], 7.22, 4.18, 4.5, 1.2, { fontSize: 18 });
  footer(s, 3);
  notes(s, "强调评价协议：random只是插值，cold-drug更接近新药筛选。");
}

// 4 method variants
{
  const s = pptx.addSlide();
  bg(s);
  title(s, "模型设计：KAN放在哪里？", "同一任务下比较预测头、药物图编码器、三元超图和药物对聚合图");
  image(s, path.join(FIG, "model_architecture_overview.png"), 0.6, 1.45, 7.35, 4.55);
  panel(s, 8.35, 1.45, 4.15, 4.55, { accent: C.teal });
  tx(s, "对照模型", 8.65, 1.75, 2.2, 0.35, { fontSize: 21, bold: true, color: C.teal });
  bulletList(s, [
    "MLP baseline：拼接后预测",
    "KAN head：直接替换MLP",
    "DrugKAN：药物图编码器使用KAN",
    "HgKAN-HG：三元超图交互",
    "HgKAN-Agg：药物对聚合图交互",
    "DrugKAN + HgKAN：叠加模型",
  ], 8.7, 2.35, 3.2, 2.6, { fontSize: 15.8, paraSpaceAfterPt: 5 });
  tx(s, "比较目的：区分“增加KAN参数”和“把KAN放在合适结构中”。", 1.1, 6.42, 10.8, 0.35, { fontSize: 18, bold: true, color: C.coral, align: "center" });
  footer(s, 4);
  notes(s, "方法页用来说明实验矩阵。接下来每组实验都围绕KAN放置位置展开。");
}

// 5 random result
{
  const s = pptx.addSlide();
  bg(s);
  title(s, "实验一：Random split 插值性能", "结论：MLP baseline仍然很强；直接KAN head没有带来稳定收益");
  table(s, [
    ["模型", "AUPR↑", "AUC↑", "RMSE↓", "MAE↓"],
    ["MLP baseline", "0.5889", "0.8833", "18.9204", "12.0727"],
    ["KAN head", "0.5064", "0.8561", "19.4118", "12.5460"],
    ["DrugKAN", "0.5556", "0.8717", "19.1304", "12.2923"],
    ["HgKAN-HG", "0.5782", "0.8805", "21.8455", "14.2129"],
    ["HgKAN-Agg", "0.5611", "0.8713", "19.2011", "12.2187"],
    ["DrugKAN+HG", "0.5345", "0.8669", "20.5617", "13.1371"],
    ["DrugKAN+Agg", "0.4213", "0.8177", "20.1643", "13.0252"],
  ], 0.7, 1.45, 7.25, 4.85, [2.25, 1.15, 1.15, 1.35, 1.35], 14.5);
  panel(s, 8.35, 1.45, 4.1, 4.85, { accent: C.coral });
  tx(s, "关键观察", 8.65, 1.78, 2.2, 0.35, { fontSize: 22, bold: true, color: C.coral });
  bulletList(s, [
    "random主要反映已见药物空间内插值",
    "KAN head分类AUPR下降：0.5889 → 0.5064",
    "DrugKAN / HgKAN接近但未超过baseline",
    "说明KAN不能无结构地替代最终MLP",
  ], 8.7, 2.45, 3.2, 2.7, { fontSize: 17 });
  footer(s, 5);
  notes(s, "random实验是负结果证据：KAN理论表达能力不能自动转化为任务收益。");
}

// 6 cold drug main
{
  const s = pptx.addSlide();
  bg(s);
  title(s, "实验二：Cold-drug 新药泛化主结果", "结论：HgKAN-Agg在分类与回归主指标上方向一致优于MLP baseline");
  metricBars(s, "分类AUPR（越高越好）", [
    { label: "MLP", value: 0.2758, color: C.navy },
    { label: "KAN head", value: 0.1717, color: C.coral },
    { label: "DrugKAN", value: 0.1967, color: C.amber },
    { label: "HgKAN-Agg", value: 0.2765, color: C.teal },
  ], 0.75, 1.45, 5.65, 3.15, true);
  panel(s, 0.75, 4.92, 2.55, 1.35, { accent: C.teal });
  tx(s, "AUC", 1.03, 5.15, 0.8, 0.26, { fontSize: 18, bold: true, color: C.teal });
  tx(s, "0.7333 → 0.7452", 1.03, 5.58, 1.85, 0.28, { fontSize: 17, bold: true, color: C.ink });
  panel(s, 3.65, 4.92, 2.75, 1.35, { accent: C.coral });
  tx(s, "RMSE", 3.93, 5.15, 1.0, 0.26, { fontSize: 18, bold: true, color: C.coral });
  tx(s, "18.3997 → 17.6763", 3.93, 5.58, 2.1, 0.28, { fontSize: 17, bold: true, color: C.ink });
  table(s, [
    ["模型", "AUPR↑", "AUC↑", "RMSE↓", "MAE↓"],
    ["MLP baseline", "0.2758", "0.7333", "18.3997", "13.6750"],
    ["KAN head", "0.1717", "0.5590", "19.2865", "14.1486"],
    ["DrugKAN", "0.1967", "0.6485", "17.8930", "13.1844"],
    ["HgKAN-HG", "0.1557", "0.5413", "19.2780", "13.8723"],
    ["HgKAN-Agg", "0.2765", "0.7452", "17.6763", "12.8615"],
    ["DrugKAN+Agg", "0.2429", "0.7205", "18.7366", "13.4615"],
  ], 6.85, 1.45, 5.75, 3.8, [1.75, 1.0, 1.0, 1.05, 0.95], 13.5);
  tx(s, "核心结论：不是KAN越多越好，而是药物对聚合图给KAN提供了更合适的结构载体。", 6.95, 5.65, 5.45, 0.68, { fontSize: 18, bold: true, color: C.teal, align: "center" });
  footer(s, 6);
  notes(s, "cold-drug是主结果：HgKAN-Agg在AUPR、AUC、RMSE、MAE四个指标上方向一致优于baseline。");
}

// 7 seed stability
{
  const s = pptx.addSlide();
  bg(s);
  title(s, "实验三：二分类两随机种子稳定性", "结论：cold-drug绝对指标随药物划分波动，但聚合图方向更稳定");
  table(s, [
    ["模型", "场景", "seed42", "seed43", "观察"],
    ["MLP baseline", "random", "0.5889 / 0.8833", "0.5799 / 0.8805", "插值稳定"],
    ["DrugKAN", "random", "0.5556 / 0.8717", "0.5614 / 0.8743", "接近baseline"],
    ["MLP baseline", "cold-drug", "0.2758 / 0.7333", "0.0839 / 0.4646", "seed43更难"],
    ["DrugKAN", "cold-drug", "0.1967 / 0.6485", "0.1095 / 0.5175", "分类不稳"],
    ["DrugKAN+KAN-HG", "cold-drug", "0.1685 / 0.5633", "0.0956 / 0.5046", "三元超图弱"],
    ["DrugKAN+KAN-Agg", "cold-drug", "0.2429 / 0.7205", "0.1663 / 0.7014", "优于KAN-HG"],
  ], 0.6, 1.45, 12.1, 3.95, [2.0, 1.4, 2.35, 2.35, 3.0], 13.2);
  panel(s, 0.8, 5.85, 11.75, 0.75, { fill: C.pale, accent: C.teal });
  tx(s, "解读：当前实验足以支持“药物对聚合图方向更有潜力”，但不足以声称cold-drug分类已统计显著领先。", 1.05, 6.06, 11.1, 0.28, { fontSize: 18, bold: true, color: C.ink, align: "center" });
  footer(s, 7);
  notes(s, "这一页补上实验完整性：说明seed波动，也说明本文没有过度包装。");
}

// 8 KAN design + params
{
  const s = pptx.addSlide();
  bg(s);
  title(s, "实验四：KAN基函数与参数量分析", "结论：Fourier更适合作为当前隐藏消息函数；KAN没有带来参数量优势");
  image(s, path.join(FIG, "fourier_bspline_rmse_comparison.png"), 0.65, 1.45, 5.55, 3.25);
  table(s, [
    ["模型", "基函数", "Random RMSE", "Cold RMSE"],
    ["DrugKAN", "Fourier", "19.1304", "17.8930"],
    ["DrugKAN", "B-spline", "18.9037", "19.0523"],
    ["HgKAN-Agg", "Fourier", "19.2011", "17.6763"],
  ], 0.8, 5.05, 5.2, 1.35, [1.55, 1.15, 1.35, 1.15], 13.5);
  table(s, [
    ["模型", "总参数", "KAN参数"],
    ["GCN_mlp", "42.64M", "0"],
    ["KAGCN_mlp", "43.46M", "0.90M"],
    ["GCN_kan_mlp", "43.95M", "1.44M"],
    ["GCN_kan_hypergraph", "45.77M", "3.07M"],
    ["GCN_kan_aggregated", "44.28M", "1.64M"],
    ["KAGCN_kan_aggregated", "45.10M", "2.55M"],
  ], 6.7, 1.45, 5.7, 3.35, [2.45, 1.55, 1.55], 13.5);
  panel(s, 6.85, 5.2, 5.4, 1.15, { accent: C.amber, fill: C.white });
  bulletList(s, [
    "当前KAN输入是隐藏通道，不是真实剂量变量",
    "所有模型总参数量约四千万级，KAN并未显著减参",
  ], 7.12, 5.42, 4.8, 0.55, { fontSize: 15.5, paraSpaceAfterPt: 2 });
  footer(s, 8);
  notes(s, "这页补充基函数与参数量，避免老师觉得只挑了主结果。");
}

// 9 explainability
{
  const s = pptx.addSlide();
  bg(s);
  title(s, "实验五：可解释性与案例分析", "Lenalidomide + mitomycin C @ IGROV1；cold-drug正样本，两药均未出现在训练集中");
  image(s, path.join(FIG, "case_57954_modality_perturbation.png"), 0.65, 1.45, 3.8, 2.2);
  image(s, path.join(FIG, "case_57954_drug_pair_response_curve.png"), 4.75, 1.45, 3.8, 2.2);
  image(s, path.join(FIG, "case_57954_kan_edge_functions.png"), 8.85, 1.45, 3.35, 2.2);
  panel(s, 0.75, 4.0, 3.65, 1.6, { accent: C.teal });
  tx(s, "扰动实验", 1.05, 4.25, 1.8, 0.3, { fontSize: 19, bold: true, color: C.teal });
  tx(s, "原始0.9106；屏蔽两药降至0.3711", 1.05, 4.82, 2.9, 0.35, { fontSize: 16, color: C.ink });
  panel(s, 4.85, 4.0, 3.65, 1.6, { accent: C.amber });
  tx(s, "响应曲线", 5.15, 4.25, 1.8, 0.3, { fontSize: 19, bold: true, color: C.amber });
  tx(s, "药物对表示恢复15%时越过0.5阈值", 5.15, 4.82, 2.9, 0.35, { fontSize: 16, color: C.ink });
  panel(s, 8.95, 4.0, 3.35, 1.6, { accent: C.coral });
  tx(s, "KAN函数", 9.25, 4.25, 1.8, 0.3, { fontSize: 19, bold: true, color: C.coral });
  tx(s, "提供隐藏消息函数级观察窗口", 9.25, 4.82, 2.45, 0.35, { fontSize: 16, color: C.ink });
  tx(s, "解释边界：这些是模型内部依赖性证据，不能直接等同于真实药理机制。", 1.35, 6.45, 10.6, 0.35, { fontSize: 18, bold: true, color: C.coral, align: "center" });
  footer(s, 9);
  notes(s, "可解释性实验包括扰动、药物对表示恢复、KAN函数和原子saliency。");
}

// 10 conclusion
{
  const s = pptx.addSlide();
  bg(s, C.navy);
  tx(s, "结论：结构先行，KAN局部函数化", 0.78, 0.75, 9.8, 0.55, { fontSize: 30, bold: true, color: C.white });
  tx(s, "本文最终结论不是“KAN全面优于MLP”，而是明确KAN在药物协同预测中的适用边界。", 0.82, 1.38, 10.2, 0.35, { fontSize: 16, color: "C9DCE5" });
  panel(s, 0.85, 2.25, 3.55, 2.4, { fill: "173D4C", line: "173D4C", shadow: false });
  tx(s, "主要结论", 1.15, 2.58, 2.0, 0.35, { fontSize: 21, bold: true, color: C.mint });
  bulletList(s, ["KAN head直接替换失败", "DrugKAN收益任务依赖", "HgKAN-Agg是最明确正结果"], 1.18, 3.18, 2.6, 1.0, { fontSize: 16, color: C.white, paraSpaceAfterPt: 5 });
  panel(s, 4.9, 2.25, 3.55, 2.4, { fill: "173D4C", line: "173D4C", shadow: false });
  tx(s, "实验支撑", 5.2, 2.58, 2.0, 0.35, { fontSize: 21, bold: true, color: C.amber });
  bulletList(s, ["random与cold-drug主表", "两seed稳定性分析", "基函数、参数量、解释性实验"], 5.23, 3.18, 2.6, 1.0, { fontSize: 16, color: C.white, paraSpaceAfterPt: 5 });
  panel(s, 8.95, 2.25, 3.55, 2.4, { fill: "173D4C", line: "173D4C", shadow: false });
  tx(s, "后续工作", 9.25, 2.58, 2.0, 0.35, { fontSize: 21, bold: true, color: C.coral });
  bulletList(s, ["更多seed与外部验证", "引入剂量变量", "增强多组学与解释验证"], 9.28, 3.18, 2.6, 1.0, { fontSize: 16, color: C.white, paraSpaceAfterPt: 5 });
  tx(s, "谢谢各位老师，欢迎批评指正", 4.15, 6.35, 5.2, 0.45, { fontSize: 22, bold: true, color: C.white, align: "center" });
  footer(s, 10);
  notes(s, "最后总结：KAN不是万能替代品，关键是放在结构化消息传递中。");
}

const script = `# 实验完整版大字号讲稿

这版答辩PPT共10页，适合8到10分钟讲；如果严格5分钟，可以重点讲1、2、4、6、9、10页，实验补充页在问答时使用。

核心表达：
1. 药物协同预测是drug-drug-cell三元交互任务。
2. 本文系统比较KAN head、DrugKAN、HgKAN-HG、HgKAN-Agg等放置方式。
3. Random split中MLP baseline仍很强，直接KAN head没有稳定收益。
4. Cold-drug中HgKAN-Agg在分类AUPR/AUC和回归RMSE/MAE上均优于MLP baseline。
5. 两seed结果说明cold-drug分类有划分方差，因此本文强调结构趋势而不是夸大SOTA。
6. Fourier/B-spline、参数量和解释性实验补齐证据链。
7. 最终结论：KAN更适合作为结构化消息传递中的可学习非线性函数，而非无结构替代最终MLP。
`;
fs.writeFileSync(NOTES, script, "utf8");
pptx.writeFile({ fileName: PPTX });
console.log(PPTX);
console.log(PDF);
console.log(NOTES);
