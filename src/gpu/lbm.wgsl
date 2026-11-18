// D3Q19 LBM 求解器（AA pattern：collide 与 stream 两个 kernel，各槽位只写一次）
// 坐标：x 流向，y 竖直，z 横向。flags: 0=流体 1=固体。

struct Params {
  nx : u32,
  ny : u32,
  nz : u32,
  _pad0 : u32,
  u0 : f32,
  tau : f32,
  smag : f32,
  steps : u32,
  forceScale : f32,
  _pad1 : f32,
  _pad2 : f32,
  _pad3 : f32,
};

const Q : i32 = 19;
const W : array<f32, 19> = array<f32, 19>(
  0.3333333333333333,
  0.0555555555555556, 0.0555555555555556, 0.0555555555555556,
  0.0555555555555556, 0.0555555555555556, 0.0555555555555556,
  0.0277777777777778, 0.0277777777777778, 0.0277777777777778,
  0.0277777777777778, 0.0277777777777778, 0.0277777777777778,
  0.0277777777777778, 0.0277777777777778, 0.0277777777777778,
  0.0277777777777778, 0.0277777777777778, 0.0277777777777778,
);

const EX : array<i32, 19> = array<i32, 19>(
  0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1, 0, 0, 0, 0);
const EY : array<i32, 19> = array<i32, 19>(
  0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1);
const EZ : array<i32, 19> = array<i32, 19>(
  0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1);
const OPP : array<i32, 19> = array<i32, 19>(
  0, 2, 1, 4, 3, 6, 5, 10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15);
const REFL_Y : array<i32, 19> = array<i32, 19>(
  0, 1, 2, 4, 3, 5, 6, 8, 7, 10, 9, 11, 12, 13, 14, 17, 18, 15, 16);
const REFL_Z : array<i32, 19> = array<i32, 19>(
  0, 1, 2, 3, 4, 6, 5, 7, 8, 9, 10, 12, 11, 14, 13, 16, 15, 18, 17);

struct Macro { rho : f32, ux : f32, uy : f32, uz : f32 };

@group(0) @binding(0) var<storage, read_write> fIn  : array<array<f32, 19>>;
@group(0) @binding(1) var<storage, read_write> fOut : array<array<f32, 19>>;
@group(0) @binding(2) var<storage, read> flags : array<u32>;
@group(0) @binding(3) var<storage, read_write> macros : array<Macro>;
@group(0) @binding(4) var<storage, read_write> forceAcc : array<atomic<i32>>;
@group(0) @binding(5) var<uniform> P : Params;

// 定点缩放：兼顾原子累加精度与 i32 溢出余量（2M 格 × 单格贡献仍远小于 2^31）
const FORCE_FP : f32 = 4096.0;

fn cellId(coord : vec3<i32>) -> u32 {
  let nyu = P.ny;
  let nzu = P.nz;
  return u32(coord.x) * (nyu * nzu) + u32(coord.y) * nzu + u32(coord.z);
}

fn feq(w : f32, rho : f32, eu : f32, u2 : f32) -> f32 {
  return w * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u2);
}

// 清零力累加器（6 个 i32：压力 Fxyz + 粘性 Fxyz）
@compute @workgroup_size(64)
fn clearForce(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x >= 6u) { return; }
  atomicStore(&forceAcc[gid.x], 0);
}

// 工作组尺寸用 override 常量，CPU 侧可按显卡调整而无需改 WGSL
override WG_X : u32 = 1u;
override WG_Y : u32 = 2u;
override WG_Z : u32 = 16u;

// 碰撞 + 宏观量 + 动量交换法力统计（含 Smagorinsky 局部松弛）
@compute @workgroup_size(WG_X, WG_Y, WG_Z)
fn collide(@builtin(global_invocation_id) gid : vec3<u32>) {
  let nx = i32(P.nx); let ny = i32(P.ny); let nz = i32(P.nz);
  let i = i32(gid.x); let j = i32(gid.y); let k = i32(gid.z);
  if (i >= nx || j >= ny || k >= nz) { return; }
  let id = cellId(vec3<i32>(i, j, k));
  if (flags[id] == 1u) { return; }

  let f = fIn[id];
  var rho : f32 = 0.0;
  var ux : f32 = 0.0; var uy : f32 = 0.0; var uz : f32 = 0.0;
  for (var q : i32 = 0; q < Q; q++) {
    rho += f[q];
    ux += f[q] * f32(EX[q]);
    uy += f[q] * f32(EY[q]);
    uz += f[q] * f32(EZ[q]);
  }
  let invRho = 1.0 / rho;
  ux *= invRho; uy *= invRho; uz *= invRho;
  macros[id] = Macro(rho, ux, uy, uz);

  // 非平衡应力张量 Π^neq_ij = Σ f^neq e_i e_j
  var pxx : f32 = 0.0; var pyy : f32 = 0.0; var pzz : f32 = 0.0;
  var pxy : f32 = 0.0; var pxz : f32 = 0.0; var pyz : f32 = 0.0;
  let u2 = ux * ux + uy * uy + uz * uz;
  for (var q : i32 = 0; q < Q; q++) {
    let ex = f32(EX[q]); let ey = f32(EY[q]); let ez = f32(EZ[q]);
    let eu = ex * ux + ey * uy + ez * uz;
    let neq = f[q] - feq(W[q], rho, eu, u2);
    pxx += neq * ex * ex; pyy += neq * ey * ey; pzz += neq * ez * ez;
    pxy += neq * ex * ey; pxz += neq * ex * ez; pyz += neq * ey * ez;
  }
  let piMag = sqrt(2.0 * (pxx * pxx + pyy * pyy + pzz * pzz
    + 2.0 * (pxy * pxy + pxz * pxz + pyz * pyz)));

  // Smagorinsky：τ_eff = 0.5 (τ + sqrt(τ² + 6 Cs² |Π| / ρ²))
  var omega : f32 = 1.0 / P.tau;
  if (P.smag > 0.0 && piMag > 1e-12) {
    let tau2 = P.tau * P.tau + 6.0 * P.smag * P.smag * piMag / (rho * rho);
    let tauEff = 0.5 * (P.tau + sqrt(max(tau2, P.tau * P.tau)));
    omega = 1.0 / tauEff;
  }

  var out : array<f32, 19>;
  for (var q : i32 = 0; q < Q; q++) {
    let ex = f32(EX[q]); let ey = f32(EY[q]); let ez = f32(EZ[q]);
    let eu = ex * ux + ey * uy + ez * uz;
    let eq = feq(W[q], rho, eu, u2);
    out[q] = f[q] - omega * (f[q] - eq);
  }
  fOut[id] = out;

  // 动量交换力：对指向固体邻居的链 q，ΔF = (f_q - f_opp) e_q
  // 法向分量归压差、切向分量归摩擦，分别累加
  var fpx : f32 = 0.0; var fpy : f32 = 0.0; var fpz : f32 = 0.0;
  var fvx : f32 = 0.0; var fvy : f32 = 0.0; var fvz : f32 = 0.0;
  let n = vec3<i32>(i, j, k);
  for (var q : i32 = 1; q < Q; q++) {
    let nb = n + vec3<i32>(EX[q], EY[q], EZ[q]);
    if (nb.x < 0 || nb.x >= nx || nb.y < 0 || nb.y >= ny || nb.z < 0 || nb.z >= nz) {
      continue;
    }
    let nid = cellId(nb);
    if (flags[nid] != 1u) { continue; }
    let qo = OPP[q];
    let df = f[q] - f[qo];
    let fx = df * f32(EX[q]);
    let fy = df * f32(EY[q]);
    let fz = df * f32(EZ[q]);
    // 链方向单位向量（D3Q19 链长 1 或 √2）
    let elen = sqrt(f32(EX[q] * EX[q] + EY[q] * EY[q] + EZ[q] * EZ[q]));
    let nx3 = f32(EX[q]) / elen;
    let ny3 = f32(EY[q]) / elen;
    let nz3 = f32(EZ[q]) / elen;
    let normalPart = fx * nx3 + fy * ny3 + fz * nz3;
    fpx += normalPart * nx3; fpy += normalPart * ny3; fpz += normalPart * nz3;
    fvx += fx - normalPart * nx3;
    fvy += fy - normalPart * ny3;
    fvz += fz - normalPart * nz3;
  }
  let s = P.forceScale * FORCE_FP;
  atomicAdd(&forceAcc[0], i32(fpx * s));
  atomicAdd(&forceAcc[1], i32(fpy * s));
  atomicAdd(&forceAcc[2], i32(fpz * s));
  atomicAdd(&forceAcc[3], i32(fvx * s));
  atomicAdd(&forceAcc[4], i32(fvy * s));
  atomicAdd(&forceAcc[5], i32(fvz * s));
}

// 迁移（pull 形式）+ 边界条件：入口 Zou-He 定速，出口零梯度，
// 地面无滑移回弹，顶/侧自由滑移镜面，固体半way回弹
@compute @workgroup_size(WG_X, WG_Y, WG_Z)
fn stream(@builtin(global_invocation_id) gid : vec3<u32>) {
  let nx = i32(P.nx); let ny = i32(P.ny); let nz = i32(P.nz);
  let i = i32(gid.x); let j = i32(gid.y); let k = i32(gid.z);
  if (i >= nx || j >= ny || k >= nz) { return; }
  let id = cellId(vec3<i32>(i, j, k));
  if (flags[id] == 1u) { return; }

  var g : array<f32, 19>;
  let n = vec3<i32>(i, j, k);
  for (var q : i32 = 0; q < Q; q++) {
    let src = n - vec3<i32>(EX[q], EY[q], EZ[q]);
    // 固体邻居：半 way 回弹
    if (src.x >= 0 && src.x < nx && src.y >= 0 && src.y < ny && src.z >= 0 && src.z < nz) {
      let sid = cellId(src);
      if (flags[sid] == 1u) {
        g[q] = fOut[id][OPP[q]];
        continue;
      }
    }
    // 域边界
    if (src.y < 0) {
      // 地面：无滑移
      g[q] = fOut[id][OPP[q]];
      continue;
    }
    if (src.y >= ny) {
      g[q] = fOut[id][REFL_Y[q]];
      continue;
    }
    if (src.z < 0 || src.z >= nz) {
      g[q] = fOut[id][REFL_Z[q]];
      continue;
    }
    if (src.x < 0 || src.x >= nx) {
      // 入口/出口在下面统一处理，先占位
      g[q] = 0.0;
      continue;
    }
    g[q] = fOut[cellId(src)][q];
  }

  // 出口 x=nx-1：ex=-1 的方向源在域外，零梯度复制上游同方向分布
  if (i == nx - 1) {
    let upId = cellId(n + vec3<i32>(-1, 0, 0));
    for (var q : i32 = 0; q < Q; q++) {
      if (EX[q] == -1) { g[q] = fOut[upId][q]; }
    }
  }

  // 入口 x=0：正则化定速边界 (u0,0,0)
  // 已知方向先求密度，再把所有未知方向（ex=+1）替换为该密度/速度下的平衡态，
  // 对 D3Q19 的轴向与面对角未知方向都稳定，不依赖非平衡回弹假设。
  if (i == 0) {
    var sumKnown : f32 = 0.0;
    for (var q : i32 = 0; q < Q; q++) {
      if (EX[q] != 1) { sumKnown += g[q]; }
    }
    let rhoIn = sumKnown / (1.0 - P.u0);
    let u2 = P.u0 * P.u0;
    for (var q : i32 = 0; q < Q; q++) {
      if (EX[q] == 1) {
        let eu = f32(EX[q]) * P.u0;
        g[q] = feq(W[q], rhoIn, eu, u2);
      }
    }
  }

  fIn[id] = g;
}

// 全流场初始化为 u=(u0,0,0) 的平衡态
@compute @workgroup_size(WG_X, WG_Y, WG_Z)
fn resetField(@builtin(global_invocation_id) gid : vec3<u32>) {
  let nx = i32(P.nx); let ny = i32(P.ny); let nz = i32(P.nz);
  let i = i32(gid.x); let j = i32(gid.y); let k = i32(gid.z);
  if (i >= nx || j >= ny || k >= nz) { return; }
  let id = cellId(vec3<i32>(i, j, k));
  var out : array<f32, 19>;
  let u2 = P.u0 * P.u0;
  if (flags[id] == 1u) {
    for (var q : i32 = 0; q < Q; q++) { out[q] = W[q]; }
  } else {
    for (var q : i32 = 0; q < Q; q++) {
      let eu = f32(EX[q]) * P.u0;
      out[q] = feq(W[q], 1.0, eu, u2);
    }
    macros[id] = Macro(1.0, P.u0, 0.0, 0.0);
  }
  fIn[id] = out;
  fOut[id] = out;
}

// binding(6) 复用：reconcileVoxels 时绑旧 flags（u32 解释），
// surfacePressure 时绑压力缓冲（u32 位型存 f32）。两个管线用同一布局。
@group(0) @binding(6) var<storage, read_write> extra : array<u32>;

// 重新体素化后的调和：固体↔流体身份变化的格子重置，其余保留（热启动）
@compute @workgroup_size(WG_X, WG_Y, WG_Z)
fn reconcileVoxels(@builtin(global_invocation_id) gid : vec3<u32>) {
  let nx = i32(P.nx); let ny = i32(P.ny); let nz = i32(P.nz);
  let i = i32(gid.x); let j = i32(gid.y); let k = i32(gid.z);
  if (i >= nx || j >= ny || k >= nz) { return; }
  let id = cellId(vec3<i32>(i, j, k));
  let oldS = extra[id];
  let newS = flags[id];
  if (oldS == newS) { return; }

  var out : array<f32, 19>;
  if (newS == 1u) {
    // 新固体：静止平衡态
    for (var q : i32 = 0; q < Q; q++) { out[q] = W[q]; }
    macros[id] = Macro(1.0, 0.0, 0.0, 0.0);
  } else {
    // 新流体：以来流速度平衡态重新上岗，周围场随后自然松弛
    let u2 = P.u0 * P.u0;
    for (var q : i32 = 0; q < Q; q++) {
      let eu = f32(EX[q]) * P.u0;
      out[q] = feq(W[q], 1.0, eu, u2);
    }
    macros[id] = Macro(1.0, P.u0, 0.0, 0.0);
  }
  fIn[id] = out;
  fOut[id] = out;
}

// 表面相对流速 u/u0：取面邻居流体速度大小均值除以来流速度，写入 extra。
// 钝体绕流的稳健可视化量：驻点/尾流低速→暖色（高压感），加速区→冷色（低压感），
// 不依赖在分离区失效的伯努利假设。内部格（无流体邻居）写 0。
@compute @workgroup_size(WG_X, WG_Y, WG_Z)
fn surfacePressure(@builtin(global_invocation_id) gid : vec3<u32>) {
  let nx = i32(P.nx); let ny = i32(P.ny); let nz = i32(P.nz);
  let i = i32(gid.x); let j = i32(gid.y); let k = i32(gid.z);
  if (i >= nx || j >= ny || k >= nz) { return; }
  let id = cellId(vec3<i32>(i, j, k));
  if (flags[id] != 1u) { extra[id] = bitcast<u32>(0.0); return; }

  var sumSpeed : f32 = 0.0;
  var cnt : f32 = 0.0;
  let n = vec3<i32>(i, j, k);
  const FACE = array<vec3<i32>, 6>(
    vec3<i32>(1, 0, 0), vec3<i32>(-1, 0, 0),
    vec3<i32>(0, 1, 0), vec3<i32>(0, -1, 0),
    vec3<i32>(0, 0, 1), vec3<i32>(0, 0, -1),
  );
  for (var d : i32 = 0; d < 6; d++) {
    let nb = n + FACE[d];
    if (nb.x < 0 || nb.x >= nx || nb.y < 0 || nb.y >= ny || nb.z < 0 || nb.z >= nz) {
      continue;
    }
    let nid = cellId(nb);
    if (flags[nid] == 0u) {
      let m = macros[nid];
      sumSpeed += sqrt(m.ux * m.ux + m.uy * m.uy + m.uz * m.uz);
      cnt += 1.0;
    }
  }
  if (cnt < 0.5) { extra[id] = bitcast<u32>(0.0); return; }
  // 表面相对流速 u/u0：低速（驻点/尾流）→暖色，高速（加速区）→冷色
  let uRatio = (sumSpeed / cnt) / max(P.u0, 1e-6);
  extra[id] = bitcast<u32>(uRatio);
}
