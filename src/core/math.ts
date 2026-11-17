// 最小 3D 数学库：vec3 / quat / mat4，纯 TS、无依赖，可在 Node 中无头运行。
// 约定：长度单位毫米(mm)，角度在对外 API 用度、内部计算用弧度。

export type Vec3 = [number, number, number];
export type Quat = [number, number, number, number]; // x,y,z,w
export type Mat4 = Float32Array; // 列主序，直接喂给 WebGPU

export const v = (x = 0, y = 0, z = 0): Vec3 => [x, y, z];
export const vec3 = v;

export function add(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}
export function sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}
export function scale(a: Vec3, s: number): Vec3 {
  return [a[0] * s, a[1] * s, a[2] * s];
}
export function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
export function cross(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}
export function length(a: Vec3): number {
  return Math.hypot(a[0], a[1], a[2]);
}
export function dist(a: Vec3, b: Vec3): number {
  return length(sub(a, b));
}
export function normalize(a: Vec3): Vec3 {
  const l = length(a);
  return l > 1e-12 ? [a[0] / l, a[1] / l, a[2] / l] : [0, 0, 0];
}
export function lerp(a: Vec3, b: Vec3, t: number): Vec3 {
  return [
    a[0] + (b[0] - a[0]) * t,
    a[1] + (b[1] - a[1]) * t,
    a[2] + (b[2] - a[2]) * t,
  ];
}
export function clamp(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x));
}

/** 四元数（标量部分 w 在第 4 位） */
export const qIdentity = (): Quat => [0, 0, 0, 1];

export function qMultiply(a: Quat, b: Quat): Quat {
  const [ax, ay, az, aw] = a;
  const [bx, by, bz, bw] = b;
  return [
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz,
  ];
}

export function qNormalize(q: Quat): Quat {
  const l = Math.hypot(q[0], q[1], q[2], q[3]);
  return l > 1e-12 ? [q[0] / l, q[1] / l, q[2] / l, q[3] / l] : qIdentity();
}

/** 绕单位轴 axis 旋转 angleRad 弧度 */
export function qFromAxisAngle(axis: Vec3, angleRad: number): Quat {
  const n = normalize(axis);
  const h = angleRad * 0.5;
  const s = Math.sin(h);
  return [n[0] * s, n[1] * s, n[2] * s, Math.cos(h)];
}

/** 旋转向量：q * v * q^-1 */
export function qRotate(q: Quat, p: Vec3): Vec3 {
  const [x, y, z, w] = q;
  // u = q.xyz, t = 2 * cross(u, p)
  const u: Vec3 = [x, y, z];
  const t = scale(cross(u, p), 2);
  // p + w*t + cross(u, t)
  return add(p, add(scale(t, w), cross(u, t)));
}

/** 球面线性插值，处理双覆盖问题 */
export function qSlerp(a: Quat, b: Quat, t: number): Quat {
  let [bx, by, bz, bw] = b;
  let cos = a[0] * bx + a[1] * by + a[2] * bz + a[3] * bw;
  if (cos < 0) {
    bx = -bx; by = -by; bz = -bz; bw = -bw;
    cos = -cos;
  }
  if (cos > 0.9995) {
    return qNormalize([
      a[0] + (bx - a[0]) * t,
      a[1] + (by - a[1]) * t,
      a[2] + (bz - a[2]) * t,
      a[3] + (bw - a[3]) * t,
    ]);
  }
  const theta0 = Math.acos(clamp(cos, -1, 1));
  const theta = theta0 * t;
  const s1 = Math.sin(theta0 - theta) / Math.sin(theta0);
  const s2 = Math.sin(theta) / Math.sin(theta0);
  return [
    a[0] * s1 + bx * s2,
    a[1] * s1 + by * s2,
    a[2] * s1 + bz * s2,
    a[3] * s1 + bw * s2,
  ];
}

/** 两个单位四元数之间的有符号夹角（弧度，范围 -π..π） */
export function qAngle(a: Quat, b: Quat): number {
  const d = qNormalize(qMultiply(conjugate(a), b));
  const w = clamp(d[3], -1, 1);
  let ang = 2 * Math.acos(Math.abs(w));
  if (d[3] < 0) ang = -ang;
  return ang;
}

export function conjugate(q: Quat): Quat {
  return [-q[0], -q[1], -q[2], q[3]];
}

/** 局部坐标系（三个单位正交轴 + 原点）到世界的旋转四元数，列 = 基向量 */
export function qFromBases(xAxis: Vec3, yAxis: Vec3, zAxis: Vec3): Quat {
  const m = [
    xAxis[0], yAxis[0], zAxis[0], 0,
    xAxis[1], yAxis[1], zAxis[1], 0,
    xAxis[2], yAxis[2], zAxis[2], 0,
    0, 0, 0, 1,
  ];
  return quatFromMat4(mat4FromValues(m));
}

export function mat4Identity(): Mat4 {
  const m = new Float32Array(16);
  m[0] = m[5] = m[10] = m[15] = 1;
  return m;
}

export function mat4FromValues(m: number[]): Mat4 {
  return new Float32Array(m);
}

/** 平移 × 旋转 的模型矩阵（列主序） */
export function mat4TR(pos: Vec3, q: Quat): Mat4 {
  const [x, y, z, w] = q;
  const x2 = x + x, y2 = y + y, z2 = z + z;
  const xx = x * x2, xy = x * y2, xz = x * z2;
  const yy = y * y2, yz = y * z2, zz = z * z2;
  const wx = w * x2, wy = w * y2, wz = w * z2;
  return new Float32Array([
    1 - (yy + zz), xy + wz, xz - wy, 0,
    xy - wz, 1 - (xx + zz), yz + wx, 0,
    xz + wy, yz - wx, 1 - (xx + yy), 0,
    pos[0], pos[1], pos[2], 1,
  ]);
}

export function mat4Multiply(a: Mat4, b: Mat4): Mat4 {
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
      out[c * 4 + r] = s;
    }
  }
  return out;
}

export function mat4Perspective(fovyRad: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1 / Math.tan(fovyRad / 2);
  const nf = 1 / (near - far);
  const m = new Float32Array(16);
  m[0] = f / aspect;
  m[5] = f;
  m[10] = far * nf;
  m[11] = -1;
  m[14] = near * far * nf;
  return m;
}

export function mat4LookAt(eye: Vec3, target: Vec3, up: Vec3): Mat4 {
  const z = normalize(sub(eye, target)); // 相机看向 -z
  const x = normalize(cross(up, z));
  const y = cross(z, x);
  // 视图矩阵 = 旋转的转置 + 平移
  const m = new Float32Array(16);
  m[0] = x[0]; m[1] = y[0]; m[2] = z[0]; m[3] = 0;
  m[4] = x[1]; m[5] = y[1]; m[6] = z[1]; m[7] = 0;
  m[8] = x[2]; m[9] = y[2]; m[10] = z[2]; m[11] = 0;
  m[12] = -dot(x, eye);
  m[13] = -dot(y, eye);
  m[14] = -dot(z, eye);
  m[15] = 1;
  return m;
}

/** 旋转四元数 -> 列主序矩阵 */
export function quatToMat4(q: Quat): Mat4 {
  return mat4TR([0, 0, 0], q);
}

/** 从旋转矩阵（列主序 4x4）提取四元数 */
export function quatFromMat4(m: Mat4): Quat {
  const trace = m[0] + m[5] + m[10];
  let q: Quat;
  if (trace > 0) {
    const s = Math.sqrt(trace + 1) * 2;
    q = [(m[6] - m[9]) / s, (m[8] - m[2]) / s, (m[1] - m[4]) / s, 0.25 * s];
  } else if (m[0] > m[5] && m[0] > m[10]) {
    const s = Math.sqrt(1 + m[0] - m[5] - m[10]) * 2;
    q = [0.25 * s, (m[4] + m[1]) / s, (m[8] + m[2]) / s, (m[6] - m[9]) / s];
  } else if (m[5] > m[10]) {
    const s = Math.sqrt(1 + m[5] - m[0] - m[10]) * 2;
    q = [(m[4] + m[1]) / s, 0.25 * s, (m[9] + m[6]) / s, (m[8] - m[2]) / s];
  } else {
    const s = Math.sqrt(1 + m[10] - m[0] - m[5]) * 2;
    q = [(m[8] + m[2]) / s, (m[9] + m[6]) / s, 0.25 * s, (m[1] - m[4]) / s];
  }
  return qNormalize(q);
}

export const deg2rad = (d: number): number => (d * Math.PI) / 180;
export const rad2deg = (r: number): number => (r * 180) / Math.PI;

/** 确定性伪随机（演示/测试可复现），mulberry32 */
export function makeRng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
