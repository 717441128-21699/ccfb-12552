// 邻面去釉（IPR, Interproximal Reduction）：
// 对每对相邻牙，沿接触方向投影牙冠顶点求包围盒，重叠量即所需去釉总量，
// 再平均分配到两个邻面，单面超过上限（默认 0.5mm）时截断并报告缺口。
// 全部为纯几何计算，可在 Node 中无头验证真值。
import type { ToothPart } from './mesh';
import type { ToothFrame } from './tooth-frame';
import {
  type Quat, type Vec3,
  add, dot, normalize, qRotate, sub,
} from './math';

export interface ToothPose {
  position: Vec3; // 世界系位置（初始为帧质心）
  orientation: Quat; // 世界系旋转（初始为单位四元数）
}

export interface IPRPair {
  labelA: number;
  labelB: number;
  /** 接触方向单位向量（A -> B） */
  axis: Vec3;
  /** A 沿轴的最大投影、B 沿轴的最小投影（mm） */
  aMax: number;
  bMin: number;
  /** 几何重叠量（mm，<=0 表示存在间隙，无需去釉） */
  overlap: number;
  /** 需要的去釉总量（mm） */
  required: number;
  /** 分配到 A 邻面 / B 邻面的去釉量（mm） */
  stripA: number;
  stripB: number;
  /** 因单面封顶而无法满足的缺口（mm） */
  shortfall: number;
  /** A / B 沿接触方向的牙冠宽度（mm），用于病历展示 */
  crownWidthA: number;
  crownWidthB: number;
}

export interface IPRReport {
  pairs: IPRPair[];
  totalStrip: number;
  maxSingleSurface: number;
  capped: boolean;
}

/**
 * 统一的顶点变换：初始位姿下 position=质心、orientation=单位四元数，
 * 任意时刻世界坐标 = position + orientation * (p - centroid0)。
 */
export function transformVertex(
  p: Vec3,
  centroid0: Vec3,
  pose: ToothPose,
): Vec3 {
  return add(pose.position, qRotate(pose.orientation, sub(p, centroid0)));
}

function extentAlong(
  part: ToothPart,
  frame: ToothFrame,
  pose: ToothPose,
  axis: Vec3,
): { min: number; max: number; width: number } {
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < part.vertexCount; i++) {
    const p: Vec3 = [
      part.positions[i * 3],
      part.positions[i * 3 + 1],
      part.positions[i * 3 + 2],
    ];
    const w = transformVertex(p, frame.centroid, pose);
    const d = dot(w, axis);
    if (d < min) min = d;
    if (d > max) max = d;
  }
  return { min, max, width: max - min };
}

/** 按牙弓极角排序，同颌内相邻才组成接触对 */
function archOrder(frames: ToothFrame[]): ToothFrame[][] {
  const upper: ToothFrame[] = [];
  const lower: ToothFrame[] = [];
  for (const f of frames) {
    const quadrant = Math.floor(f.label / 10);
    if (quadrant === 1 || quadrant === 2) upper.push(f);
    else if (quadrant === 3 || quadrant === 4) lower.push(f);
  }
  // 极角绕 z 轴；FDI 11->17 与 21->27 用角度连续排序后天然首尾相接于中线
  const byAngle = (list: ToothFrame[]) =>
    [...list].sort((a, b) => Math.atan2(a.centroid[1], a.centroid[0]) - Math.atan2(b.centroid[1], b.centroid[0]));
  return [byAngle(upper), byAngle(lower)].filter((l) => l.length > 0);
}

export interface IPROptions {
  /** 单个邻面去釉上限（mm） */
  maxPerSurface: number;
}

/**
 * 计算所有相邻牙对的去釉分配。
 * @param poses 每颗牙当前/目标位姿（标签索引）；不传则用初始位姿
 */
export function computeIPR(
  teeth: ToothPart[],
  frames: ToothFrame[],
  poses?: Map<number, ToothPose>,
  options: IPROptions = { maxPerSurface: 0.5 },
): IPRReport {
  const partByLabel = new Map(teeth.map((t) => [t.label, t]));
  const initialPose = (f: ToothFrame): ToothPose => ({
    position: [...f.centroid] as Vec3,
    orientation: [0, 0, 0, 1],
  });

  const pairs: IPRPair[] = [];
  for (const arch of archOrder(frames)) {
    for (let i = 0; i + 1 < arch.length; i++) {
      const fa = arch[i];
      const fb = arch[i + 1];
      // 中线两侧（11 与 21、31 与 41）也是邻接对，角度排序后它们同样相邻
      const pa = poses?.get(fa.label) ?? initialPose(fa);
      const pb = poses?.get(fb.label) ?? initialPose(fb);
      const axis = normalize(sub(pb.position, pa.position));
      if (axis[0] === 0 && axis[1] === 0 && axis[2] === 0) continue;

      const partA = partByLabel.get(fa.label)!;
      const partB = partByLabel.get(fb.label)!;
      const extA = extentAlong(partA, fa, pa, axis);
      const extB = extentAlong(partB, fb, pb, axis);

      const overlap = extA.max - extB.min;
      const required = Math.max(0, overlap);
      // 平均分配到两个邻面；单面封顶
      const wantEach = required / 2;
      const stripA = Math.min(wantEach, options.maxPerSurface);
      const stripB = Math.min(wantEach, options.maxPerSurface);
      const shortfall = Math.max(0, required - stripA - stripB);

      pairs.push({
        labelA: fa.label,
        labelB: fb.label,
        axis,
        aMax: extA.max,
        bMin: extB.min,
        overlap,
        required,
        stripA,
        stripB,
        shortfall,
        crownWidthA: extA.width,
        crownWidthB: extB.width,
      });
    }
  }

  const totalStrip = pairs.reduce((s, p) => s + p.stripA + p.stripB, 0);
  const maxSingleSurface = pairs.reduce(
    (m, p) => Math.max(m, p.stripA, p.stripB),
    0,
  );
  return {
    pairs,
    totalStrip,
    maxSingleSurface,
    capped: pairs.some((p) => p.shortfall > 1e-9),
  };
}

/** 给定分步进位姿，找出去釉事件首次出现的步骤序号（重叠由负转正的最早一步） */
export function iprEventStep(
  pair: IPRPair,
  teeth: ToothPart[],
  frames: ToothFrame[],
  stepPoses: Map<number, ToothPose>[],
): number {
  const partA = teeth.find((t) => t.label === pair.labelA)!;
  const partB = teeth.find((t) => t.label === pair.labelB)!;
  const fA = frames.find((f) => f.label === pair.labelA)!;
  const fB = frames.find((f) => f.label === pair.labelB)!;
  for (let s = 0; s < stepPoses.length; s++) {
    const pa = stepPoses[s].get(pair.labelA)!;
    const pb = stepPoses[s].get(pair.labelB)!;
    const axis = normalize(sub(pb.position, pa.position));
    const extA = extentAlong(partA, fA, pa, axis);
    const extB = extentAlong(partB, fB, pb, axis);
    if (extA.max - extB.min > 0) return s;
  }
  return stepPoses.length; // 最终位姿也不碰撞则不安排
}

/** 工具：沿指定轴平移一颗牙的位姿 */
export function shiftedPose(base: ToothPose, frame: ToothFrame, localDelta: Vec3): ToothPose {
  const worldDelta: Vec3 = [
    frame.mesioDistal[0] * localDelta[0] + frame.buccoLingual[0] * localDelta[1] + frame.longAxis[0] * localDelta[2],
    frame.mesioDistal[1] * localDelta[0] + frame.buccoLingual[1] * localDelta[1] + frame.longAxis[1] * localDelta[2],
    frame.mesioDistal[2] * localDelta[0] + frame.buccoLingual[2] * localDelta[1] + frame.longAxis[2] * localDelta[2],
  ];
  return { position: add(base.position, worldDelta), orientation: base.orientation };
}

