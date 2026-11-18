/**
 * D3Q19 格子玻尔兹曼模型的共享常量。
 * 坐标约定：x = 流向（风沿 +x 吹），y = 竖直向上，z = 横向。
 * 单元格编号见 WGSL/TS 两侧共用的 EX/EY/EZ 表，顺序不可随意调换。
 */

export const Q = 19;

// 19 个离散速度方向
export const EX = new Int32Array([
  0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1, 0, 0, 0, 0,
]);
export const EY = new Int32Array([
  0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1,
]);
export const EZ = new Int32Array([
  0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1,
]);

// 权重：静止 1/3，轴向 1/18，面对角 1/36
export const WEIGHTS = new Float32Array([
  1 / 3,
  ...Array(6).fill(1 / 18),
  ...Array(12).fill(1 / 36),
]);

/** 反方向编号表 opp[q] */
export const OPP = new Uint8Array(19);
/** 关于 x 法向平面的镜面反射方向表（自由滑移壁面用） */
export const REFL_X = new Uint8Array(19);
export const REFL_Y = new Uint8Array(19);
export const REFL_Z = new Uint8Array(19);

function dirIndex(dx: number, dy: number, dz: number): number {
  for (let q = 0; q < Q; q++) {
    if (EX[q] === dx && EY[q] === dy && EZ[q] === dz) return q;
  }
  throw new Error(`direction not found: ${dx},${dy},${dz}`);
}

for (let q = 0; q < Q; q++) {
  OPP[q] = dirIndex(-EX[q], -EY[q], -EZ[q]);
  REFL_X[q] = dirIndex(-EX[q], EY[q], EZ[q]);
  REFL_Y[q] = dirIndex(EX[q], -EY[q], EZ[q]);
  REFL_Z[q] = dirIndex(EX[q], EY[q], -EZ[q]);
}

export const FLUID = 0;
export const SOLID = 1;

/** 求解器运行参数（与 WGSL Params 结构体布局严格对应，16 字节对齐） */
export interface SimParams {
  nx: number;
  ny: number;
  nz: number;
  u0: number;
  tau: number;
  smagorinsky: number;
  steps: number;
  forceScale: number;
}

export function paramsBufferData(p: SimParams): ArrayBuffer {
  const buf = new ArrayBuffer(48);
  const u32 = new Uint32Array(buf);
  const f32 = new Float32Array(buf);
  u32[0] = p.nx;
  u32[1] = p.ny;
  u32[2] = p.nz;
  u32[3] = 0;
  f32[4] = p.u0;
  f32[5] = p.tau;
  f32[6] = p.smagorinsky;
  u32[7] = p.steps;
  f32[8] = p.forceScale;
  f32[9] = 0;
  f32[10] = 0;
  f32[11] = 0;
  return buf;
}
