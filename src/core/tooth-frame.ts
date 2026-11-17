// 牙体局部坐标系：质心 + PCA 主方向定牙长轴，再按牙弓切线定近远中方向。
// 平移、扭转(绕长轴)、倾斜/转矩(绕两个水平轴)都在这个局部系里设置。
import type { ToothPart } from './mesh';
import {
  type Quat, type Vec3,
  add, cross, dot, normalize, qFromBases, qIdentity, scale, sub, v,
} from './math';

export interface ToothFrame {
  label: number;
  centroid: Vec3; // 局部系原点（牙体质心）
  longAxis: Vec3; // 牙长轴单位向量（指向咬合面方向）
  mesioDistal: Vec3; // 近远中方向（沿牙弓切线）
  buccoLingual: Vec3; // 颊舌方向
  orientation: Quat; // 局部基 -> 世界 的旋转
}

/** 3x3 对称矩阵特征分解（Jacobi 迭代），返回特征值升序及对应特征向量（列） */
function eig3Symmetric(a: number[]): { values: [number, number, number]; vectors: number[] } {
  // 行主序 3x3
  const m = [...a];
  const v = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  for (let iter = 0; iter < 64; iter++) {
    const off = Math.hypot(m[1], m[2], m[5]);
    if (off < 1e-14) break;
    // 找最大非对角元
    let p = 0, q = 1;
    if (Math.abs(m[2]) > Math.abs(m[1]) && Math.abs(m[2]) > Math.abs(m[5])) { p = 0; q = 2; }
    else if (Math.abs(m[5]) > Math.abs(m[1])) { p = 1; q = 2; }
    const app = m[p * 3 + p], aqq = m[q * 3 + q], apq = m[p * 3 + q];
    const phi = 0.5 * Math.atan2(2 * apq, aqq - app);
    const c = Math.cos(phi), s = Math.sin(phi);
    for (let k = 0; k < 3; k++) {
      const mpk = m[p * 3 + k], mqk = m[q * 3 + k];
      m[p * 3 + k] = c * mpk - s * mqk;
      m[q * 3 + k] = s * mpk + c * mqk;
    }
    for (let k = 0; k < 3; k++) {
      const mkp = m[k * 3 + p], mkq = m[k * 3 + q];
      m[k * 3 + p] = c * mkp - s * mkq;
      m[k * 3 + q] = s * mkp + c * mkq;
    }
    for (let k = 0; k < 3; k++) {
      const vkp = v[k * 3 + p], vkq = v[k * 3 + q];
      v[k * 3 + p] = c * vkp - s * vkq;
      v[k * 3 + q] = s * vkp + c * vkq;
    }
  }
  const idx = [0, 1, 2].sort((i, j) => m[i * 3 + i] - m[j * 3 + j]);
  const values: [number, number, number] = [
    m[idx[0] * 3 + idx[0]],
    m[idx[1] * 3 + idx[1]],
    m[idx[2] * 3 + idx[2]],
  ];
  const vectors = [
    v[idx[0] * 3], v[idx[1] * 3], v[idx[2] * 3],
    v[idx[0] * 3 + 1], v[idx[1] * 3 + 1], v[idx[2] * 3 + 1],
    v[idx[0] * 3 + 2], v[idx[1] * 3 + 2], v[idx[2] * 3 + 2],
  ];
  return { values, vectors };
}

export function centroidOf(part: ToothPart): Vec3 {
  const c = v();
  const n = part.vertexCount;
  if (n === 0) return c;
  for (let i = 0; i < n; i++) {
    c[0] += part.positions[i * 3];
    c[1] += part.positions[i * 3 + 1];
    c[2] += part.positions[i * 3 + 2];
  }
  return scale(c, 1 / n);
}

/**
 * 为所有牙建立局部坐标系。
 * @param archCenter 牙弓中心（全部牙质心的平均），用于判定咬合方向与颊舌方向
 * @param occlusalHint 咬合方向提示（世界系单位向量），PCA 长轴符号向它对齐
 */
export function buildFrames(
  teeth: ToothPart[],
  occlusalHint: Vec3 = v(0, 0, 1),
): ToothFrame[] {
  const allCentroid = teeth.reduce<Vec3>((acc, t) => add(acc, centroidOf(t)), v());
  const archCenter = scale(allCentroid, 1 / Math.max(1, teeth.length));

  return teeth.map((part) => {
    const c = centroidOf(part);
    // 协方差矩阵（面积不加权，顶点均匀即可）
    const cov = new Array(9).fill(0);
    for (let i = 0; i < part.vertexCount; i++) {
      const d = sub([
        part.positions[i * 3],
        part.positions[i * 3 + 1],
        part.positions[i * 3 + 2],
      ], c);
      for (let r = 0; r < 3; r++) {
        for (let k = 0; k < 3; k++) cov[r * 3 + k] += d[r] * d[k];
      }
    }
    const { vectors } = eig3Symmetric(cov);
    // 最大特征值对应列索引 2，即主伸长方向 = 牙长轴
    let longAxis: Vec3 = [vectors[6], vectors[7], vectors[8]];
    if (dot(longAxis, occlusalHint) < 0) longAxis = scale(longAxis, -1);
    longAxis = normalize(longAxis);

    // 近远中：牙弓切向 ≈ 水平面上垂直于"中心->该牙"径向的方向
    const radial = sub(c, archCenter);
    radial[2] = 0;
    let md: Vec3;
    if (Math.hypot(radial[0], radial[1]) < 1e-6) {
      md = normalize(cross(v(0, 1, 0), longAxis));
    } else {
      const r = normalize(radial);
      // 切向：绕 z 旋转径向 90°
      md = v(-r[1], r[0], 0);
    }
    // 去掉沿长轴的分量，保证正交
    md = normalize(sub(md, scale(longAxis, dot(md, longAxis))));
    const bl = normalize(cross(longAxis, md)); // 右手系：bl = long × md
    const orientation = qFromBases(md, bl, longAxis);
    return {
      label: part.label,
      centroid: c,
      longAxis,
      mesioDistal: md,
      buccoLingual: bl,
      orientation,
    };
  });
}

/** 世界系向量 -> 牙局部系 */
export function worldToLocal(frame: ToothFrame, d: Vec3): Vec3 {
  // orientation 的列是局部基在世界中的表达，故局部坐标 = 基向量点乘
  return [
    dot(frame.mesioDistal, d),
    dot(frame.buccoLingual, d),
    dot(frame.longAxis, d),
  ];
}

export function frameAtIdentity(frame: ToothFrame): { position: Vec3; orientation: Quat } {
  return { position: [...frame.centroid] as Vec3, orientation: qIdentity() };
}
