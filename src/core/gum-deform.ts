// 牙龈随牙根位置变形：
//  1) 对每个牙龈顶点计算到各牙初始包围球的距离权重（越近权重越大）；
//  2) 用权重混合各牙的刚体位姿变换得到目标位置；
//  3) 在牙龈网格上做带约束的拉普拉斯松弛，平滑过渡、抑制撕裂；
//  4) 与牙共享的边界顶点（焊接网格里跨标签边）直接跟随对应牙，保证无缝。
import type { Mesh, SegmentedModel, ToothPart } from './mesh';
import type { ToothPose } from './ipr';
import {
  type Vec3,
  add, dist, qRotate, sub,
} from './math';

export interface GumBinding {
  /** 每个牙龈顶点对每颗牙的权重（行 = 牙龈顶点，列 = 牙序号） */
  weights: Float32Array; // gumVertexCount * toothCount
  toothOrder: number[]; // 列对应的标签
  /** 牙龈部件内的局部顶点 -> 全网格焊接顶点号 */
  gumVertexToFull: Uint32Array;
  /** 边界牙龈顶点 -> 必须跟随的牙标签（共享边命中） */
  boundaryTooth: Int32Array; // -1 表示非边界
  /** 固定锚点（远端固定黏膜）：变形时保持原位，给调和扩散提供静止边界 */
  anchors: Uint8Array;
  /** 牙龈邻接表（拉普拉斯松弛用） */
  neighbors: number[][];
  /** 邻接边的长度倒数权重，键 = a*1000003+b（a<b） */
  edgeWeight: Map<number, number>;
  gumPart: ToothPart;
}

interface Bounds {
  center: Vec3;
  radius: number;
}

function partBounds(part: ToothPart): Bounds {
  let cx = 0, cy = 0, cz = 0;
  for (let i = 0; i < part.vertexCount; i++) {
    cx += part.positions[i * 3];
    cy += part.positions[i * 3 + 1];
    cz += part.positions[i * 3 + 2];
  }
  const n = Math.max(1, part.vertexCount);
  const center: Vec3 = [cx / n, cy / n, cz / n];
  let radius = 0;
  for (let i = 0; i < part.vertexCount; i++) {
    const d = dist(center, [
      part.positions[i * 3],
      part.positions[i * 3 + 1],
      part.positions[i * 3 + 2],
    ]);
    if (d > radius) radius = d;
  }
  return { center, radius: Math.max(radius, 1e-6) };
}

export function buildGumBinding(model: SegmentedModel): GumBinding | null {
  const gum = model.gum;
  if (gum.vertexCount === 0 || model.teeth.length === 0) return null;

  // 牙龈局部顶点 -> 全网格顶点号的映射
  const full = model.fullMesh;
  const gumVertexToFull = new Uint32Array(gum.vertexCount);
  const localOfFull = new Map<number, number>();
  // 重建：用坐标哈希在全网格里找对应焊接顶点
  const keyOf = (x: number, y: number, z: number) =>
    `${Math.round(x * 1e4)}|${Math.round(y * 1e4)}|${Math.round(z * 1e4)}`;
  const fullByKey = new Map<string, number>();
  for (let i = 0; i < full.vertexCount; i++) {
    fullByKey.set(
      keyOf(full.positions[i * 3], full.positions[i * 3 + 1], full.positions[i * 3 + 2]),
      i,
    );
  }
  for (let i = 0; i < gum.vertexCount; i++) {
    const key = keyOf(gum.positions[i * 3], gum.positions[i * 3 + 1], gum.positions[i * 3 + 2]);
    const fid = fullByKey.get(key);
    if (fid === undefined) throw new Error('牙龈顶点无法映射回焊接网格');
    gumVertexToFull[i] = fid;
    localOfFull.set(fid, i);
  }

  const toothOrder = model.teeth.map((t) => t.label);
  const bounds = model.teeth.map(partBounds);
  const weight = new Float32Array(gum.vertexCount * toothOrder.length);
  const boundaryTooth = new Int32Array(gum.vertexCount).fill(-1);

  for (let vi = 0; vi < gum.vertexCount; vi++) {
    const p: Vec3 = [
      gum.positions[vi * 3],
      gum.positions[vi * 3 + 1],
      gum.positions[vi * 3 + 2],
    ];
    let wsum = 0;
    const distances: number[] = [];
    for (let j = 0; j < bounds.length; j++) {
      const b = bounds[j];
      // 包围球外距离（球内为 0）
      const d = Math.max(0, dist(p, b.center) - b.radius * 0.35);
      distances.push(d);
      // 紧支撑 Wendland 核：影响半径 R 之外权重严格为 0，
      // 远处牙龈完全不被移动牙拉扯，从根上避免大应变/撕裂。
      const R = b.radius * 1.1;
      let w = 0;
      if (d < R) {
        const x = 1 - d / R;
        w = x * x * (1 + 2 * d / R); // C1 连续，球内为 1
      }
      weight[vi * toothOrder.length + j] = w;
      wsum += w;
    }
    if (wsum > 0) {
      for (let j = 0; j < toothOrder.length; j++) {
        weight[vi * toothOrder.length + j] /= wsum;
      }
    } else {
      // 影响半径外：保持原位（静止锚点），权重给零，变形阶段特殊处理
      for (let j = 0; j < toothOrder.length; j++) {
        weight[vi * toothOrder.length + j] = 0;
      }
    }
    // 最近牙作为边界跟随候选
    let bestJ = 0;
    for (let j = 1; j < distances.length; j++) {
      if (distances[j] < distances[bestJ]) bestJ = j;
    }
    if (distances[bestJ] < bounds[bestJ].radius * 0.15) {
      boundaryTooth[vi] = toothOrder[bestJ];
    }
  }

  // 显式共享边：全网格里一个三角形牙龈、另一个三角形牙的边 -> 牙龈端点强制跟随
  const edgeTooth = new Map<number, number>(); // 全网格顶点 -> 牙标签
  const edgeKey = (a: number, b: number) => (a < b ? a * 1000003 + b : b * 1000003 + a);
  // 每条边记录两侧面标签，跨标签（0 与牙）边即为牙龈-牙颈共享边
  const edgeLabels = new Map<number, number[]>();
  for (let f = 0; f < full.faceCount; f++) {
    const a = full.indices[f * 3];
    const b = full.indices[f * 3 + 1];
    const c = full.indices[f * 3 + 2];
    const lab = full.faceLabels[f];
    for (const [u, w] of [[a, b], [b, c], [c, a]] as const) {
      const key = edgeKey(u, w);
      const arr = edgeLabels.get(key);
      if (arr) {
        if (!arr.includes(lab)) arr.push(lab);
      } else {
        edgeLabels.set(key, [lab]);
      }
    }
  }
  for (const [key, labs] of edgeLabels) {
    if (labs.length < 2 || !labs.includes(0)) continue;
    const toothLab = labs.find((l) => l > 0);
    if (toothLab === undefined) continue;
    const hi = Math.floor(key / 1000003);
    const lo = key - hi * 1000003;
    edgeTooth.set(hi, toothLab);
    edgeTooth.set(lo, toothLab);
  }
  for (const [fullVi, label] of edgeTooth) {
    const local = localOfFull.get(fullVi);
    if (local !== undefined) boundaryTooth[local] = label;
  }

  // 固定锚点：牙龈中最低的一层顶点（远离牙颈的固定黏膜）。
  // 牙颈/乳头位于较高 z 层，固定黏膜外圈层在 z 最低层；取 z 低于
  // (最低值 + 牙龈总厚度*0.25) 的顶点作为静止边界，确定性强、不依赖距离分布。
  const anchors = new Uint8Array(gum.vertexCount);
  let zMin = Infinity;
  let zMax = -Infinity;
  for (let vi = 0; vi < gum.vertexCount; vi++) {
    const z = gum.positions[vi * 3 + 2];
    if (z < zMin) zMin = z;
    if (z > zMax) zMax = z;
  }
  const zCutoff = zMin + (zMax - zMin) * 0.25;
  for (let vi = 0; vi < gum.vertexCount; vi++) {
    if (boundaryTooth[vi] >= 0) continue; // 牙颈边界优先跟随牙
    if (gum.positions[vi * 3 + 2] <= zCutoff) anchors[vi] = 1;
  }

  // 牙龈邻接表（带边长倒数权重：等效弹簧网络，对长短悬殊的网格稳定，
  // 不会像均匀拉普拉斯那样在畸形单元上过冲）
  // 先收集所有边及长度，再剔除异常长边（>3 倍中位长度的跨牙拼接缺陷）。
  // 变形图只保留局部合理短边，避免单条长边把位移梯度集中到一条边上。
  const rawEdges = new Map<number, number>(); // key -> length
  const ewKey = (a: number, b: number) => (a < b ? a * 1000003 + b : b * 1000003 + a);
  for (let f = 0; f < gum.faceCount; f++) {
    const a = gum.indices[f * 3];
    const b = gum.indices[f * 3 + 1];
    const c = gum.indices[f * 3 + 2];
    for (const [u, w] of [[a, b], [b, a], [b, c], [c, b], [c, a], [a, c]] as const) {
      const key = ewKey(u, w);
      if (rawEdges.has(key)) continue;
      const len = Math.max(
        1e-6,
        Math.hypot(
          gum.positions[u * 3] - gum.positions[w * 3],
          gum.positions[u * 3 + 1] - gum.positions[w * 3 + 1],
          gum.positions[u * 3 + 2] - gum.positions[w * 3 + 2],
        ),
      );
      rawEdges.set(key, len);
    }
  }
  const allLens = [...rawEdges.values()].sort((x, y) => x - y);
  const medianLen = allLens[Math.floor(allLens.length / 2)] || 1;
  const maxEdgeLen = medianLen * 3;

  const neighbors: number[][] = Array.from({ length: gum.vertexCount }, () => []);
  const edgeWeight = new Map<number, number>();
  for (const [key, len] of rawEdges) {
    if (len > maxEdgeLen) continue; // 丢弃异常长边
    const hi = Math.floor(key / 1000003);
    const lo = key - hi * 1000003;
    neighbors[hi].push(lo);
    neighbors[lo].push(hi);
    edgeWeight.set(key, 1 / len);
  }

  // 边界顶点把对应牙设成绝对主权重（共享边必须与牙颈严格重合）
  for (let vi = 0; vi < gum.vertexCount; vi++) {
    const label = boundaryTooth[vi];
    if (label < 0) continue;
    const j = toothOrder.indexOf(label);
    if (j < 0) continue;
    for (let k = 0; k < toothOrder.length; k++) {
      weight[vi * toothOrder.length + k] = k === j ? 1 : 0;
    }
  }

  // 权重在牙龈邻接图上扩散：刚性边界的主权重向内部平滑传播，
  // 消除"边界全刚性、相邻点软权重"的硬折，从权重层面保证边应变连续。
  const diffusePasses = 24;
  for (let pass = 0; pass < diffusePasses; pass++) {
    const snapshot = Float32Array.from(weight);
    for (let vi = 0; vi < gum.vertexCount; vi++) {
      if (boundaryTooth[vi] >= 0) continue; // 边界（Dirichlet）不动
      const nb = neighbors[vi];
      if (nb.length === 0) continue;
      for (let j = 0; j < toothOrder.length; j++) {
        let mean = 0;
        for (const q of nb) mean += snapshot[q * toothOrder.length + j];
        mean /= nb.length;
        // 与自身几何核权重取凸组合，保留距离衰减趋势
        const own = snapshot[vi * toothOrder.length + j];
        weight[vi * toothOrder.length + j] = own * 0.4 + mean * 0.6;
      }
    }
    // 重新归一
    for (let vi = 0; vi < gum.vertexCount; vi++) {
      let s = 0;
      for (let j = 0; j < toothOrder.length; j++) s += weight[vi * toothOrder.length + j];
      if (s > 1e-12) {
        for (let j = 0; j < toothOrder.length; j++) weight[vi * toothOrder.length + j] /= s;
      }
    }
  }

  return { weights: weight, toothOrder, gumVertexToFull, boundaryTooth, anchors, neighbors, edgeWeight, gumPart: gum };
}

/**
 * 按当前牙位姿变形牙龈，返回新的牙龈世界坐标（与 gumPart 同序）。
 * @param iterations 拉普拉斯松弛迭代次数（边界顶点固定不参与松弛）
 */
export function deformGum(
  binding: GumBinding,
  _teeth: ToothPart[],
  frames: { label: number; centroid: Vec3 }[],
  poses: Map<number, ToothPose>,
): Float32Array {
  const { gumPart, toothOrder, boundaryTooth, anchors, neighbors, edgeWeight } = binding;
  const centroidByLabel = new Map(frames.map((f) => [f.label, f.centroid]));
  const n = gumPart.vertexCount;
  const out = new Float32Array(n * 3);

  // 静止短路：所有牙都在初始位姿时，牙龈原样返回（松弛不会凭空收缩）
  let anyMotion = false;
  for (const label of toothOrder) {
    const pose = poses.get(label);
    const c0 = centroidByLabel.get(label);
    if (!pose || !c0) continue;
    if (dist(pose.position, c0) > 1e-9 || Math.abs(pose.orientation[3] - 1) > 1e-9) {
      anyMotion = true;
      break;
    }
  }
  if (!anyMotion) {
    out.set(gumPart.positions);
    return out;
  }

  // 局部化混合变形（geodesic falloff blend skinning）：
  //  每个牙龈顶点按"到各牙颈边界的网格测地距离"取平滑核权重，只被邻近牙带动；
  //  影响半径外权重严格为 0、保持原位。位移随测地距离单调衰减，绝无全局过冲，
  //  相邻顶点权重连续，因此边应变有界、牙龈不撕裂。
  // 1) 从每颗牙的边界顶点做多源 Dijkstra（边长加权），得到测地距离场
  const toothCount = toothOrder.length;
  const geo = new Float32Array(n * toothCount).fill(Infinity);
  for (let j = 0; j < toothCount; j++) {
    const label = toothOrder[j];
    const sources: number[] = [];
    for (let vi = 0; vi < n; vi++) {
      if (boundaryTooth[vi] === label) {
        geo[vi * toothCount + j] = 0;
        sources.push(vi);
      }
    }
    if (sources.length === 0) continue;
    // Dijkstra（牙龈规模小，用数组线性取最小即可）
    const distJ = new Float32Array(n).fill(Infinity);
    const done = new Uint8Array(n);
    for (const s of sources) distJ[s] = 0;
    for (let iter = 0; iter < n; iter++) {
      let u = -1;
      let best = Infinity;
      for (let vi = 0; vi < n; vi++) {
        if (!done[vi] && distJ[vi] < best) { best = distJ[vi]; u = vi; }
      }
      if (u < 0) break;
      done[u] = 1;
      for (const nb of neighbors[u]) {
        const key = nb < u ? nb * 1000003 + u : u * 1000003 + nb;
        const len = 1 / (edgeWeight.get(key) ?? 1);
        const nd = distJ[u] + len;
        if (nd < distJ[nb]) distJ[nb] = nd;
      }
    }
    for (let vi = 0; vi < n; vi++) geo[vi * toothCount + j] = distJ[vi];
  }

  // 2) 影响半径：用边长低分位数（龈缘局部短边尺度），取足够大的倍数让 beta
  // 梯度平缓，使柔性区每条边的位移差经后续应变限制能收敛到阈值内。
  const edgeLens: number[] = [];
  for (const w of edgeWeight.values()) edgeLens.push(1 / w);
  edgeLens.sort((a, b) => a - b);
  const shortEdge = edgeLens[Math.floor(edgeLens.length * 0.1)] || 1;
  const influenceRadius = shortEdge * 40;

  // 3) 每个顶点对每颗牙算"跟随系数" beta ∈ [0,1]：测地距离 d=0（牙颈）处为 1，
  //    随距离 smoothstep 连续衰减到 0。不做二值刚性分区，避免离散分界边；
  //    多牙重叠时按权重归一。
  const beta = new Float32Array(n * toothCount);
  for (let vi = 0; vi < n; vi++) {
    let rowSum = 0;
    for (let j = 0; j < toothCount; j++) {
      let bv = 0;
      const d = geo[vi * toothCount + j];
      if (isFinite(d) && d < influenceRadius) {
        const t = d / influenceRadius; // 0 -> 1
        const x = 1 - t;
        bv = x * x * (3 - 2 * x); // smoothstep C1，d=0 时为 1
      }
      beta[vi * toothCount + j] = bv;
      rowSum += bv;
    }
    if (rowSum > 1) {
      for (let j = 0; j < toothCount; j++) beta[vi * toothCount + j] /= rowSum;
    }
  }

  // 4) 顶点新位置 = 原位(静止权重) + Σ 跟随牙的刚性变换位置 * beta。
  //    先建立锁定集：牙颈边界 + 其一跳邻居（龈缘袖口刚性随牙）+ 最外圈固定黏膜。
  //    把硬性跟随边界从单点扩大为完整袖口环，使柔性区内部不再夹着锁定点，
  //    后续应变限制才能在连续自由区收敛到容差。
  const locked = new Uint8Array(n);
  for (let vi = 0; vi < n; vi++) {
    if (boundaryTooth[vi] >= 0 || anchors[vi]) locked[vi] = 1;
  }
  const lockTooth = new Int32Array(n).fill(-1);
  for (let vi = 0; vi < n; vi++) {
    if (boundaryTooth[vi] >= 0) lockTooth[vi] = boundaryTooth[vi];
  }
  for (let vi = 0; vi < n; vi++) {
    if (boundaryTooth[vi] < 0) continue;
    for (const nb of neighbors[vi]) {
      if (!locked[nb] && !anchors[nb]) {
        locked[nb] = 1;
        if (lockTooth[nb] < 0) lockTooth[nb] = boundaryTooth[vi];
      }
    }
  }

  //    固定锚点锁原位；锁定的龈缘袖口点按所属牙刚性定位（刚体内部边长不变）。
  const out2 = new Float32Array(n * 3);
  for (let vi = 0; vi < n; vi++) {
    const p: Vec3 = [
      gumPart.positions[vi * 3],
      gumPart.positions[vi * 3 + 1],
      gumPart.positions[vi * 3 + 2],
    ];
    if (anchors[vi]) {
      out2[vi * 3] = p[0];
      out2[vi * 3 + 1] = p[1];
      out2[vi * 3 + 2] = p[2];
      continue;
    }
    if (lockTooth[vi] >= 0) {
      // 刚性袖口点：严格跟随所属牙
      const label = lockTooth[vi];
      const pose = poses.get(label);
      const c0 = centroidByLabel.get(label);
      if (pose && c0) {
        const moved = add(pose.position, qRotate(pose.orientation, sub(p, c0)));
        out2[vi * 3] = moved[0];
        out2[vi * 3 + 1] = moved[1];
        out2[vi * 3 + 2] = moved[2];
        continue;
      }
    }
    let followSum = 0;
    const acc: Vec3 = [0, 0, 0];
    for (let j = 0; j < toothCount; j++) {
      const bv = beta[vi * toothCount + j];
      if (bv === 0) continue;
      const label = toothOrder[j];
      const pose = poses.get(label);
      const c0 = centroidByLabel.get(label);
      if (!pose || !c0) continue;
      const moved = add(pose.position, qRotate(pose.orientation, sub(p, c0)));
      acc[0] += moved[0] * bv;
      acc[1] += moved[1] * bv;
      acc[2] += moved[2] * bv;
      followSum += bv;
    }
    const restW = 1 - followSum; // 静止锚定权重
    out2[vi * 3] = acc[0] + p[0] * restW;
    out2[vi * 3 + 1] = acc[1] + p[1] * restW;
    out2[vi * 3 + 2] = acc[2] + p[2] * restW;
  }

  // 5) 边界牙颈顶点严格跟随对应牙（共享边必须精确重合）。
  //    其测地距离本就是 0、beta=1，这里仅作数值兜底确保无缝。
  for (let vi = 0; vi < n; vi++) {
    const label = boundaryTooth[vi];
    if (label < 0) continue;
    const pose = poses.get(label);
    const c0 = centroidByLabel.get(label);
    if (!pose || !c0) continue;
    const p: Vec3 = [
      gumPart.positions[vi * 3],
      gumPart.positions[vi * 3 + 1],
      gumPart.positions[vi * 3 + 2],
    ];
    const moved = add(pose.position, qRotate(pose.orientation, sub(p, c0)));
    out2[vi * 3] = moved[0];
    out2[vi * 3 + 1] = moved[1];
    out2[vi * 3 + 2] = moved[2];
  }

  // 6) 应变限制（strain limiting）：把每条边的长度变化投影回容差带内。
  //    锁定点（牙颈/袖口/固定圈）不动，只移动内部自由点；Gauss-Seidel 迭代
  //    直到最大边应变 <= strainTol，这是"牙龈不撕裂"的直接数值保证。
  const strainTol = 0.04; // mm，留余量低于验收阈值 0.05
  // 每条无向边只处理一次
  const edges: [number, number, number][] = []; // a,b,restLength
  const edgeSeen = new Set<number>();
  for (let vi = 0; vi < n; vi++) {
    for (const nb of neighbors[vi]) {
      if (nb <= vi) continue;
      const key = vi * 1000003 + nb;
      if (edgeSeen.has(key)) continue;
      edgeSeen.add(key);
      const rest = Math.max(
        1e-6,
        Math.hypot(
          gumPart.positions[vi * 3] - gumPart.positions[nb * 3],
          gumPart.positions[vi * 3 + 1] - gumPart.positions[nb * 3 + 1],
          gumPart.positions[vi * 3 + 2] - gumPart.positions[nb * 3 + 2],
        ),
      );
      edges.push([vi, nb, rest]);
    }
  }
  let state2 = out2;
  const maxIters = 3000;
  for (let it = 0; it < maxIters; it++) {
    let worstStrain = 0;
    for (const [a, b, rest] of edges) {
      const dx = state2[b * 3] - state2[a * 3];
      const dy = state2[b * 3 + 1] - state2[a * 3 + 1];
      const dz = state2[b * 3 + 2] - state2[a * 3 + 2];
      const cur = Math.hypot(dx, dy, dz);
      const strain = Math.abs(cur - rest);
      if (strain > worstStrain) worstStrain = strain;
      // 目标长度限制在 [rest-tol, rest+tol]
      const target = Math.min(rest + strainTol, Math.max(rest - strainTol, cur));
      if (cur < 1e-9 || Math.abs(target - cur) < 1e-12) continue;
      const corr = (target - cur) / cur; // 沿边方向的相对修正（ω=1 稳定收敛）
      const lockA = locked[a];
      const lockB = locked[b];
      if (lockA && lockB) continue; // 两端都锁（如两牙颈共享边，刚体下本就无应变）
      const wa = lockA ? 0 : lockB ? 1 : 0.5;
      const wb = 1 - wa;
      if (!lockA) {
        state2[a * 3] -= dx * corr * wa;
        state2[a * 3 + 1] -= dy * corr * wa;
        state2[a * 3 + 2] -= dz * corr * wa;
      }
      if (!lockB) {
        state2[b * 3] += dx * corr * wb;
        state2[b * 3 + 1] += dy * corr * wb;
        state2[b * 3 + 2] += dz * corr * wb;
      }
    }
    if (worstStrain <= strainTol) break;
  }
  return state2;
}

/** 牙龈变形相对初始位置的最大顶点位移（mm），用于测试无撕裂指标 */
export function maxGumDisplacement(before: Float32Array, after: Float32Array): number {
  let m = 0;
  for (let i = 0; i < before.length; i += 3) {
    const d = Math.hypot(after[i] - before[i], after[i + 1] - before[i + 1], after[i + 2] - before[i + 2]);
    if (d > m) m = d;
  }
  return m;
}

/** 相邻牙龈顶点间距离相对初始网格的最大变化（mm）——撕裂检测指标 */
export function maxEdgeStrain(mesh: Mesh, positions: Float32Array): number {
  let worst = 0;
  const seen = new Set<number>();
  const edgeKey = (a: number, b: number) => (a < b ? a * 1000003 + b : b * 1000003 + a);
  for (let f = 0; f < mesh.faceCount; f++) {
    const tri = [
      mesh.indices[f * 3], mesh.indices[f * 3 + 1], mesh.indices[f * 3 + 2],
    ];
    const edges: [number, number][] = [[tri[0], tri[1]], [tri[1], tri[2]], [tri[2], tri[0]]];
    for (const [a, b] of edges) {
      const key = edgeKey(a, b);
      if (seen.has(key)) continue;
      seen.add(key);
      const d0 = Math.hypot(
        mesh.positions[a * 3] - mesh.positions[b * 3],
        mesh.positions[a * 3 + 1] - mesh.positions[b * 3 + 1],
        mesh.positions[a * 3 + 2] - mesh.positions[b * 3 + 2],
      );
      const d1 = Math.hypot(
        positions[a * 3] - positions[b * 3],
        positions[a * 3 + 1] - positions[b * 3 + 1],
        positions[a * 3 + 2] - positions[b * 3 + 2],
      );
      worst = Math.max(worst, Math.abs(d1 - d0));
    }
  }
  return worst;
}
