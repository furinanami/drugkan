const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "../..");
const FIG = path.join(__dirname, "figures");
const OUT = path.join(__dirname, "defense");
fs.mkdirSync(OUT, { recursive: true });

const PPTX = path.join(OUT, "本科毕设答辩_5分钟_KAN超图药物协同预测_skill版.pptx");
const NOTES = path.join(OUT, "本科毕设答辩_5分钟_讲稿_skill版.md");

const C = {
  night: "102A43",
  deep: "174A5B",
  teal: "0F8B8D",
  mint: "D7F3F0",
  amber: "F2A65A",
  coral: "E85D75",
  ink: "20313F",
  muted: "657786",
  line: "D8E1E8",
  paper: "F8FAFB",
  white: "FFFFFF",
};

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "Codex";
pptx.company = "Generated with pptx skill workflow";
pptx.subject = "本科毕业论文答辩";
pptx.title = "基于KAN超图的药物协同预测研究";
pptx.lang = "zh-CN";
pptx.theme = {
  headFontFace: "Microsoft YaHei",
  bodyFontFace: "Microsoft YaHei",
  lang: "zh-CN",
};
pptx.defineLayout({ name: "LAYOUT_WIDE", width: 13.333, height: 7.5 });

function shadow() {
  return { type: "outer", color: "0B1820", opacity: 0.12, blur: 2, offset: 1, angle: 45 };
}

function bg(slide, color = C.paper) {
  slide.background = { color };
}

function tx(slide, text, x, y, w, h, opt = {}) {
  slide.addText(text, {
    x, y, w, h,
    margin: opt.margin ?? 0.02,
    fontFace: opt.fontFace ?? "Microsoft YaHei",
    fontSize: opt.fontSize ?? 14,
    bold: opt.bold ?? false,
    italic: opt.italic ?? false,
    color: opt.color ?? C.ink,
    align: opt.align ?? "left",
    valign: opt.valign ?? "top",
    breakLine: opt.breakLine,
    fit: opt.fit ?? "shrink",
    paraSpaceAfterPt: opt.paraSpaceAfterPt,
    bullet: opt.bullet,
    rotate: opt.rotate,
  });
}

function title(slide, main, sub) {
  tx(slide, main, 0.55, 0.35, 9.7, 0.55, { fontSize: 23, bold: true, color: C.night });
  if (sub) tx(slide, sub, 0.58, 0.93, 10.2, 0.36, { fontSize: 9.5, color: C.muted });
  slide.addShape(pptx.ShapeType.arc, { x: 11.9, y: 0.18, w: 1.0, h: 1.0, line: { color: C.mint, transparency: 40, width: 8 }, adjustPoint: 0.45 });
}

function footer(slide, n) {
  tx(slide, String(n).padStart(2, "0"), 12.32, 7.02, 0.42, 0.2, { fontSize: 8.5, color: C.muted, align: "right" });
}

function panel(slide, x, y, w, h, opt = {}) {
  slide.addShape(pptx.ShapeType.rect, {
    x, y, w, h,
    fill: { color: opt.fill ?? C.white },
    line: { color: opt.line ?? C.line, width: 0.6 },
    shadow: opt.shadow === false ? undefined : shadow(),
  });
  if (opt.accent) {
    slide.addShape(pptx.ShapeType.rect, { x, y, w: 0.08, h, fill: { color: opt.accent }, line: { color: opt.accent } });
  }
}

function stat(slide, value, label, x, y, w, color) {
  panel(slide, x, y, w, 0.95, { fill: C.white, accent: color });
  tx(slide, value, x + 0.15, y + 0.13, w - 0.3, 0.32, { fontSize: 20, bold: true, color, align: "center" });
  tx(slide, label, x + 0.1, y + 0.55, w - 0.2, 0.25, { fontSize: 8.5, color: C.muted, align: "center" });
}

function bulletList(slide, lines, x, y, w, h, opt = {}) {
  const runs = lines.map((line, idx) => ({
    text: line,
    options: { bullet: true, breakLine: idx !== lines.length - 1 },
  }));
  slide.addText(runs, {
    x, y, w, h, margin: 0.02,
    fontFace: "Microsoft YaHei",
    fontSize: opt.fontSize ?? 12.5,
    color: opt.color ?? C.ink,
    breakLine: true,
    valign: "top",
    fit: "shrink",
    paraSpaceAfterPt: opt.paraSpaceAfterPt ?? 5,
  });
}

function image(slide, file, x, y, w, h) {
  slide.addImage({ path: file, x, y, w, h, sizing: { type: "contain", w, h } });
}

function pill(slide, text, x, y, w, color) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x, y, w, h: 0.34,
    rectRadius: 0.06,
    fill: { color },
    line: { color },
  });
  tx(slide, text, x, y + 0.07, w, 0.18, { fontSize: 8.5, color: C.white, bold: true, align: "center" });
}

function notes(slide, lines) {
  slide.addNotes(lines.join("\n"));
}

// Slide 1
{
  const s = pptx.addSlide();
  bg(s, C.night);
  s.addShape(pptx.ShapeType.rect, { x: 0, y: 0, w: 13.333, h: 7.5, fill: { color: C.night }, line: { color: C.night } });
  s.addShape(pptx.ShapeType.rect, { x: 0, y: 0, w: 4.25, h: 7.5, fill: { color: C.deep }, line: { color: C.deep } });
  s.addShape(pptx.ShapeType.arc, { x: 8.75, y: -0.45, w: 3.6, h: 3.6, line: { color: C.teal, transparency: 10, width: 16 }, adjustPoint: 0.28 });
  s.addShape(pptx.ShapeType.arc, { x: 10.2, y: 4.8, w: 2.1, h: 2.1, line: { color: C.amber, transparency: 15, width: 10 }, adjustPoint: 0.52 });
  pill(s, "5分钟答辩版", 0.72, 0.62, 1.35, C.teal);
  tx(s, "基于KAN超图的\n药物协同预测研究", 0.72, 1.75, 7.9, 1.55, { fontSize: 30, bold: true, color: C.white, fit: "shrink" });
  tx(s, "Kolmogorov-Arnold Network for structured drug-drug-cell interaction", 0.76, 3.55, 7.3, 0.3, { fontSize: 11.5, color: "C8E7E3" });
  stat(s, "16万+", "DrugComb相关样本", 8.2, 2.1, 1.55, C.teal);
  stat(s, "0", "cold-drug药物泄漏", 10.0, 2.1, 1.55, C.amber);
  stat(s, "HgKAN-Agg", "主结果模型", 8.2, 3.4, 3.35, C.coral);
  tx(s, "答辩人：________    指导教师：________    专业：________", 0.78, 6.55, 7.6, 0.3, { fontSize: 10.5, color: "C8D8E0" });
  tx(s, "结构先行，KAN局部函数化", 8.25, 5.55, 3.25, 0.38, { fontSize: 14.5, bold: true, color: C.white, align: "center" });
  notes(s, [
    "各位老师好，我的题目是《基于KAN超图的药物协同预测研究》。",
    "本文关注如何利用药物分子结构、细胞系表达和两药组合关系，预测药物组合协同效应。",
  ]);
}

// Slide 2
{
  const s = pptx.addSlide();
  bg(s);
  title(s, "研究问题：药物协同不是单药性质，而是三元交互", "联合用药模型更适合作为候选排序工具，而不是替代实验判定");
  panel(s, 0.58, 1.65, 3.7, 4.75, { accent: C.coral });
  tx(s, "为什么难？", 0.83, 1.9, 2.6, 0.3, { fontSize: 15, bold: true, color: C.coral });
  bulletList(s, [
    "组合空间随药物数量快速增长",
    "协同正例少：约14.8%",
    "random划分容易高估泛化",
    "药物发现需要可解释线索",
  ], 0.9, 2.48, 2.95, 2.55, { fontSize: 11.5 });
  panel(s, 4.85, 1.65, 3.7, 4.75, { accent: C.teal });
  tx(s, "本文问什么？", 5.1, 1.9, 2.8, 0.3, { fontSize: 15, bold: true, color: C.teal });
  bulletList(s, [
    "KAN能否提升协同预测？",
    "应放在预测头、药物图编码器，还是交互层？",
    "模型是否真正依赖药物对结构？",
  ], 5.17, 2.48, 2.95, 2.65, { fontSize: 11.5 });
  panel(s, 9.12, 1.65, 3.4, 4.75, { fill: C.night, line: C.night, shadow: false });
  tx(s, "本文答案", 9.45, 2.05, 2.5, 0.3, { fontSize: 15, bold: true, color: C.white, align: "center" });
  tx(s, "KAN不是\nMLP的通用替代品", 9.45, 2.72, 2.75, 0.9, { fontSize: 18, bold: true, color: C.amber, align: "center" });
  tx(s, "更适合作为结构化消息传递中的\n可学习非线性函数", 9.45, 4.12, 2.75, 0.8, { fontSize: 14.5, bold: true, color: C.mint, align: "center" });
  footer(s, 2);
  notes(s, [
    "联合用药任务的关键是药物A、药物B和细胞背景的三元交互。",
    "因此本文不只比较模型名字，而是系统比较KAN放置位置，并把cold-drug泛化作为重点。",
  ]);
}

// Slide 3
{
  const s = pptx.addSlide();
  bg(s);
  title(s, "数据与评价：同时覆盖分类、回归和新药泛化", "严格区分random插值性能与cold-drug未见药物泛化性能");
  stat(s, "160,228", "有效样本", 0.65, 1.65, 2.1, C.teal);
  stat(s, "1,896", "唯一药物", 3.05, 1.65, 2.1, C.deep);
  stat(s, "156", "细胞系", 5.45, 1.65, 2.1, C.amber);
  stat(s, "14.8%", "正例比例", 7.85, 1.65, 2.1, C.coral);
  stat(s, "0", "train-test药物重叠", 10.25, 1.65, 2.1, C.night);
  panel(s, 0.75, 3.15, 5.65, 2.65, { accent: C.teal });
  tx(s, "输入与任务", 1.0, 3.42, 2.2, 0.28, { fontSize: 14.5, bold: true, color: C.teal });
  bulletList(s, [
    "输入：药物A分子图 + 药物B分子图 + 细胞系表达",
    "分类：协同/非协同，报告AUPR与AUC",
    "回归：连续Loewe score，报告RMSE与MAE",
  ], 1.05, 3.92, 4.78, 1.25, { fontSize: 11 });
  panel(s, 6.95, 3.15, 5.65, 2.65, { accent: C.deep });
  tx(s, "评价协议", 7.2, 3.42, 2.2, 0.28, { fontSize: 14.5, bold: true, color: C.deep });
  bulletList(s, [
    "random split：检验已有药物空间中的插值拟合",
    "cold-drug split：测试药物未出现在训练集中",
    "主结论优先依据更接近新药筛选的cold-drug场景",
  ], 7.25, 3.92, 4.78, 1.25, { fontSize: 11 });
  footer(s, 3);
  notes(s, [
    "本文构建约16万条样本，包含药物图、细胞表达、二分类标签和Loewe分数。",
    "分类看AUPR和AUC，回归看RMSE和MAE。cold-drug划分中训练和测试药物集合重叠为0。",
  ]);
}

// Slide 4
{
  const s = pptx.addSlide();
  bg(s);
  title(s, "方法框架：比较KAN的不同放置位置", "同一输入和任务下，辨析“直接替换”与“结构化消息传递”的差异");
  image(s, path.join(FIG, "model_architecture_overview.png"), 0.55, 1.55, 7.25, 4.0);
  panel(s, 8.15, 1.55, 4.55, 4.0, { accent: C.teal });
  tx(s, "四类模型变体", 8.45, 1.85, 2.4, 0.3, { fontSize: 14.5, bold: true, color: C.teal });
  bulletList(s, [
    "MLP baseline：拼接后预测",
    "KAN head：直接替换最终MLP",
    "DrugKAN：药物分子图消息函数",
    "HgKAN-HG：三元超图交互",
    "HgKAN-Agg：drug-pair节点 + cell交互",
  ], 8.48, 2.35, 3.65, 2.25, { fontSize: 10.8, paraSpaceAfterPt: 3 });
  s.addShape(pptx.ShapeType.chevron, { x: 4.0, y: 5.92, w: 1.0, h: 0.42, fill: { color: C.teal }, line: { color: C.teal } });
  tx(s, "药物图编码", 1.2, 5.95, 2.2, 0.28, { fontSize: 10.5, bold: true, color: C.deep, align: "center" });
  tx(s, "drug-drug-cell交互", 5.1, 5.95, 2.5, 0.28, { fontSize: 10.5, bold: true, color: C.deep, align: "center" });
  tx(s, "最终主结果：HgKAN-Agg", 8.8, 5.95, 3.2, 0.36, { fontSize: 15.5, bold: true, color: C.coral, align: "center" });
  footer(s, 4);
  notes(s, [
    "整体框架包括药物图编码器、细胞表达编码器和交互模块。",
    "关键比较是KAN放置位置：直接预测头、单药图编码、三元超图、药物对聚合图。",
    "最终最有效的是HgKAN-Agg，先形成drug-pair节点，再与cell节点交互。",
  ]);
}

// Slide 5
{
  const s = pptx.addSlide();
  bg(s);
  title(s, "实验结果：HgKAN-Agg在cold-drug上方向最一致", "提升幅度不夸大，重点是KAN适用边界与结构趋势");
  const labels = ["MLP", "KAN head", "DrugKAN", "HgKAN-Agg"];
  s.addChart(pptx.ChartType.bar, [
    { name: "AUPR", labels, values: [0.2758, 0.1717, 0.1967, 0.2765] },
    { name: "AUC", labels, values: [0.7333, 0.5590, 0.6485, 0.7452] },
  ], {
    x: 0.55, y: 1.55, w: 5.75, h: 2.15,
    catAxisLabelFontFace: "Microsoft YaHei",
    valAxisLabelFontFace: "Microsoft YaHei",
    catAxisLabelFontSize: 8,
    valAxisLabelFontSize: 8,
    chartColors: [C.teal, "8FB8DE"],
    showLegend: true,
    legendPos: "b",
    showValue: false,
    valGridLine: { color: C.line, size: 0.5 },
    showTitle: true,
    title: "Classification: higher is better",
    titleFontFace: "Microsoft YaHei",
    titleFontSize: 10,
  });
  s.addChart(pptx.ChartType.bar, [
    { name: "RMSE", labels, values: [18.3997, 19.2865, 17.8930, 17.6763] },
    { name: "MAE", labels, values: [13.6750, 14.1486, 13.1844, 12.8615] },
  ], {
    x: 0.55, y: 4.05, w: 5.75, h: 2.15,
    catAxisLabelFontFace: "Microsoft YaHei",
    valAxisLabelFontFace: "Microsoft YaHei",
    catAxisLabelFontSize: 8,
    valAxisLabelFontSize: 8,
    chartColors: [C.coral, "8FB8DE"],
    showLegend: true,
    legendPos: "b",
    showValue: false,
    valGridLine: { color: C.line, size: 0.5 },
    showTitle: true,
    title: "Regression: lower is better",
    titleFontFace: "Microsoft YaHei",
    titleFontSize: 10,
  });
  panel(s, 7.0, 1.55, 5.75, 4.65, { accent: C.amber });
  tx(s, "Cold-drug主结果", 7.3, 1.85, 2.6, 0.3, { fontSize: 14.5, bold: true, color: C.amber });
  const table = [
    [
      { text: "模型", options: { bold: true, color: C.white, fill: { color: C.deep } } },
      { text: "AUPR↑", options: { bold: true, color: C.white, fill: { color: C.deep } } },
      { text: "AUC↑", options: { bold: true, color: C.white, fill: { color: C.deep } } },
      { text: "RMSE↓", options: { bold: true, color: C.white, fill: { color: C.deep } } },
      { text: "MAE↓", options: { bold: true, color: C.white, fill: { color: C.deep } } },
    ],
    ["MLP", "0.2758", "0.7333", "18.3997", "13.6750"],
    ["KAN head", "0.1717", "0.5590", "19.2865", "14.1486"],
    ["DrugKAN", "0.1967", "0.6485", "17.8930", "13.1844"],
    ["HgKAN-Agg", "0.2765", "0.7452", "17.6763", "12.8615"],
  ];
  s.addTable(table, {
    x: 7.32, y: 2.35, w: 5.1, h: 1.65,
    colW: [1.25, 0.88, 0.82, 1.1, 1.05],
    fontFace: "Microsoft YaHei",
    fontSize: 8.3,
    border: { pt: 0.5, color: C.line },
    valign: "mid",
    align: "center",
    margin: 0.04,
  });
  bulletList(s, [
    "直接KAN head退化：高维拼接表示不适合无结构KAN",
    "DrugKAN回归有收益，但分类不稳定",
    "HgKAN-Agg四个主指标均优于MLP baseline",
  ], 7.38, 4.35, 4.75, 1.15, { fontSize: 10.6, paraSpaceAfterPt: 3 });
  footer(s, 5);
  notes(s, [
    "random split下MLP baseline仍然很强，因此KAN不是通用替代品。",
    "关键在cold-drug结果：KAN head显著退化，DrugKAN分类不稳定，而HgKAN-Agg在分类和回归主要指标上均优于MLP baseline。",
    "这说明药物对聚合图给KAN提供了更合适的结构载体。",
  ]);
}

// Slide 6
{
  const s = pptx.addSlide();
  bg(s);
  title(s, "案例解释：模型输出会随药物对信息改变", "Lenalidomide + mitomycin C @ IGROV1；cold-drug正样本，两药均未出现在训练集中");
  image(s, path.join(FIG, "case_57954_visual_explanation_overview.png"), 0.55, 1.47, 7.0, 4.98);
  panel(s, 8.0, 1.55, 4.72, 4.9, { accent: C.teal });
  tx(s, "解释链条", 8.3, 1.86, 2.0, 0.3, { fontSize: 14.5, bold: true, color: C.teal });
  stat(s, "0.9106", "原始协同概率", 8.32, 2.38, 1.35, C.teal);
  stat(s, "0.3711", "屏蔽两药后", 9.9, 2.38, 1.35, C.coral);
  stat(s, "0.6506", "恢复15%药物对表示", 11.48, 2.38, 1.05, C.amber);
  bulletList(s, [
    "药物扰动：屏蔽两药使预测从协同区间降至非协同区间",
    "响应曲线：药物对表示恢复时，输出呈非线性上升",
    "原子saliency：高贡献区域与已知相关子结构重合",
  ], 8.33, 3.72, 3.85, 1.35, { fontSize: 10.6, paraSpaceAfterPt: 4 });
  tx(s, "边界：这是模型内部依赖性证据，不能直接等同于真实药理机制验证。", 8.35, 5.62, 3.7, 0.36, { fontSize: 9.8, bold: true, color: C.coral, align: "center" });
  footer(s, 6);
  notes(s, [
    "案例为Lenalidomide与mitomycin C在IGROV1细胞系上的cold-drug正样本。",
    "模型原始协同概率为0.9106，屏蔽两种药物后降到0.3711；药物对表示恢复到15%时已经越过0.5阈值。",
    "原子saliency还与Lenalidomide的glutarimide环、mitomycin C的quinone/mitosene骨架存在合理重合。",
  ]);
}

// Slide 7
{
  const s = pptx.addSlide();
  bg(s, C.night);
  s.addShape(pptx.ShapeType.rect, { x: 0, y: 0, w: 13.333, h: 7.5, fill: { color: C.night }, line: { color: C.night } });
  tx(s, "结论：KAN的价值取决于结构化放置位置", 0.72, 0.72, 9.7, 0.55, { fontSize: 24, bold: true, color: C.white });
  tx(s, "本文不是证明KAN全面优于MLP，而是明确它在药物协同预测中的适用边界。", 0.75, 1.33, 8.7, 0.3, { fontSize: 11, color: "C8D8E0" });
  panel(s, 0.75, 2.12, 3.65, 2.8, { fill: "173D4C", line: "173D4C", shadow: false });
  tx(s, "主要结论", 1.02, 2.42, 1.8, 0.28, { fontSize: 14.5, bold: true, color: C.mint });
  bulletList(s, [
    "无结构替换预测头效果较差",
    "单药DrugKAN收益有限且任务依赖",
    "HgKAN-Agg是当前最明确正结果",
  ], 1.05, 2.95, 2.75, 1.25, { fontSize: 10.8, color: C.white, paraSpaceAfterPt: 3 });
  panel(s, 4.85, 2.12, 3.65, 2.8, { fill: "173D4C", line: "173D4C", shadow: false });
  tx(s, "本文贡献", 5.12, 2.42, 1.8, 0.28, { fontSize: 14.5, bold: true, color: C.amber });
  bulletList(s, [
    "严格cold-drug评估与泄漏检查",
    "系统比较KAN多种放置方式",
    "构建扰动与原子saliency解释流程",
  ], 5.15, 2.95, 2.75, 1.25, { fontSize: 10.8, color: C.white, paraSpaceAfterPt: 3 });
  panel(s, 8.95, 2.12, 3.65, 2.8, { fill: "173D4C", line: "173D4C", shadow: false });
  tx(s, "后续工作", 9.22, 2.42, 1.8, 0.28, { fontSize: 14.5, bold: true, color: C.coral });
  bulletList(s, [
    "更多seed与外部数据集验证",
    "加入剂量变量和多组学特征",
    "更系统的解释性统计验证",
  ], 9.25, 2.95, 2.75, 1.25, { fontSize: 10.8, color: C.white, paraSpaceAfterPt: 3 });
  tx(s, "结构先行，KAN作为结构化消息传递中的可学习函数模块更有价值。", 1.6, 5.65, 10.1, 0.42, { fontSize: 18, bold: true, color: C.mint, align: "center" });
  tx(s, "谢谢各位老师，欢迎批评指正", 4.0, 6.55, 5.4, 0.35, { fontSize: 16, bold: true, color: C.white, align: "center" });
  footer(s, 7);
  notes(s, [
    "本文最终结论不是KAN全面优于传统深度模型，而是KAN的效果高度依赖放置位置。",
    "在药物协同预测中，它更适合作为结构化消息传递中的可学习非线性函数，而不是无结构地替代最终MLP。",
    "谢谢各位老师。",
  ]);
}

const script = `# 5分钟答辩讲稿（skill版）

## 1. 封面（约20秒）
各位老师好，我的题目是“基于KAN超图的药物协同预测研究”。本文关注如何利用药物分子结构、细胞系表达和两药组合关系，预测药物组合协同效应。

## 2. 研究问题（约40秒）
联合用药任务的关键不是单个药物是否有效，而是药物A、药物B和细胞背景共同决定的三元交互。本文重点不是问KAN这个名字是否更强，而是系统比较它应该放在最终预测头、药物图编码器，还是drug-drug-cell交互层。

## 3. 数据与评价（约40秒）
本文构建约16万条DrugComb相关样本，包含药物分子图、细胞系表达、协同二分类标签和连续Loewe分数。分类报告AUPR和AUC，回归报告RMSE和MAE。cold-drug划分中训练集和测试集药物重叠为0，更接近未见药物筛选场景。

## 4. 方法框架（约60秒）
模型由药物图编码器、细胞表达编码器和交互模块组成。比较的变体包括MLP baseline、直接KAN head、DrugKAN、HgKAN-HG和HgKAN-Agg。最终主结果是HgKAN-Agg，它先把两种药物聚合成drug-pair节点，再与cell节点通过KAN消息函数交互。

## 5. 实验结果（约70秒）
random split下MLP baseline仍然很强，说明KAN不是MLP的通用替代品。关键在cold-drug结果：直接KAN head明显退化，AUC只有0.5590；DrugKAN回归有收益但分类不稳定；HgKAN-Agg在分类AUPR/AUC和回归RMSE/MAE上均优于MLP baseline，尤其RMSE从18.3997降到17.6763。

## 6. 案例解释（约50秒）
案例是Lenalidomide与mitomycin C在IGROV1细胞系上的cold-drug正样本，两种药物都未出现在训练集中。模型原始协同概率为0.9106，屏蔽两药后降至0.3711；药物对表示恢复到15%时，概率升至0.6506并越过阈值。原子saliency也与已知相关子结构有合理重合。

## 7. 结论（约40秒）
本文结论是：KAN的价值取决于结构化放置位置。无结构替换最终预测头效果较差；把KAN作为图或超图消息传递中的可学习非线性函数更合理。后续需要更多随机种子、外部数据集、剂量变量和更系统的解释验证。
`;

fs.writeFileSync(NOTES, script, "utf8");
pptx.writeFile({ fileName: PPTX });
console.log(PPTX);
console.log(NOTES);
