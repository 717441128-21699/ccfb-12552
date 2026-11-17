# 排牙动画工作室 · Aligner Studio

口腔诊所隐形正畸自用排牙动画工具。导入口扫 STL（带预分割面标签）即可做牙齿分割、
移动规划、自动拆步、邻面去釉（IPR）计算、牙龈跟随变形，并通过 WebGPU 实时播放分步动画。

## 功能与需求对应

| 需求 | 实现 | 关键文件 |
| --- | --- | --- |
| 口扫模型按预标顶点/面标签拆成单颗牙和牙龈 | STL 解析 + 顶点焊接 + 按面标签分割（0=牙龈，1..32=FDI 牙位） | `src/core/stl.ts`, `src/core/segmentation.ts` |
| 每颗牙定牙长轴，平移/扭转/倾斜分开设置 | PCA 主方向定牙长轴 + 牙弓切线定近远中/颊舌局部坐标系；6 自由度目标量 | `src/core/tooth-frame.ts`, `src/core/planner.ts` |
| 每步移动量不超上限，超了自动拆更多步 | 按平移/旋转上限取 ceil 统一拆步，逐步插值，测试保证每步不超限 | `src/core/planner.ts` |
| 去釉按邻牙牙冠宽度重叠自动分配，单面 ≤0.5mm | 沿接触方向投影包围盒求重叠，平均分配两邻面，0.5mm 封顶并报告缺口 | `src/core/ipr.ts` |
| 牙龈跟随牙根变形、不撕裂 | 测地距离权重混合 + 应变限制（strain limiting），边应变压到 0.05mm 内 | `src/core/gum-deform.ts` |
| 牙釉质半透明、次表面散射近似 | WGSL：wrap lighting + 背向透射 + 边缘 rim 的 SSS 近似 | `src/core/renderer.ts` |
| 分步动画可拖动、每步显示哪几颗在动 | 可拖动时间轴（支持步间小数插值）+ 播放，面板列出本步活动牙 | `src/app.ts`, `src/main.ts` |
| 测试牙真值校验，去釉误差 ≤0.05mm | 双测试牙盒体模型，解析几何真值，19 项测试全过 | `src/tests/ipr.test.ts`, `tools/gen-test-model.ts` |

## 快速开始

```bash
npm install
npm run dev        # 浏览器打开终端提示的本地地址（Chrome/Edge 113+）
npm test           # 运行全部几何/规划/去釉/牙龈测试（Node 无头）
npm run build      # 类型检查 + 生产构建
npm run gen:testmodel   # 导出两颗已知间距测试牙 STL + 标签到 out/
```

打开后默认载入内置演示牙列（14 颗上颌牙 + 规则栅格牙龈），可直接体验规划与动画；
点「测试牙」载入双测试牙模型用于去釉真值核对。

## 输入文件约定

STL 本身是三角面汤、不携带标签，因此采用 **STL + 旁车标签文件**：

- `*.stl`：标准二进制或 ASCII STL。
- `*.lab` / `*.labels`：纯文本，每行一个整数，按三角形序号给出面标签；`#` 开头为注释。
- 或 `*.json`：`{"faceLabels": [0,11,11,...]}`，也可直接给一个数组。

标签语义：`0` = 牙龈；`11..48` = FDI 两位牙位代号。无标签文件时所有面归为牙龈并提示。

## 邻面去釉（IPR）方法

对每对沿牙弓相邻的牙 A、B：

1. 以两牙质心连线为接触轴 `axis`；
2. 把 A、B 当前位姿下的全部顶点投影到该轴，取 A 的最大投影 `aMax`、B 的最小投影 `bMin`；
3. 几何重叠 `overlap = aMax - bMin`（≤0 表示有间隙，不去釉）；
4. 所需去釉总量 = 正重叠量，平均分配到两个邻面，单面超过 `maxPerSurface`（默认 0.5mm）时截断，
   未满足部分计入 `shortfall` 缺口并在 UI 标警告。

测试牙为轴对齐盒体，包围盒精确，故投影重叠即解析真值；`ipr.test.ts` 在多组重叠量下
断言计算值与真值偏差 ≤ 0.05mm。

## 牙龈变形为何不撕裂

- 牙颈与牙龈的共享边顶点作为 Dirichlet 约束，刚性跟随对应牙；
- 其余牙龈顶点按「到各牙颈的网格测地距离」取 smoothstep 核权重做局部混合，影响半径外严格静止；
- 最后做应变限制迭代：把每条边的长度变化投影回 `[原长-0.04, 原长+0.04]mm` 容差带，
  牙颈点与最外圈固定黏膜锁定，内部自由点 Gauss-Seidel 松弛至收敛。
- `gum.test.ts` 断言变形全过程最大边长变化 ≤ 0.05mm，即数值上保证不撕裂。

## 目录结构

```
src/core/   纯 TS 几何/规划管线（无 DOM、无 WebGPU 依赖，可 Node 无头测试）
  math.ts         vec3/quat/mat4
  mesh.ts stl.ts  网格类型、STL 解析、焊接、法线、分割
  tooth-frame.ts  PCA 牙长轴与局部坐标系
  planner.ts      目标位姿、自动拆步、连续采样
  ipr.ts          邻面去釉分配
  gum-deform.ts   测地权重 + 应变限制牙龈变形
  synthetic.ts    演示牙列与双测试牙真值模型
  renderer.ts     WebGPU 渲染与釉质 SSS 近似
src/tests/  Node test 验收测试
tools/      测试模型导出
test/       Node ESM 解析 .ts 导入的 loader 钩子
```
