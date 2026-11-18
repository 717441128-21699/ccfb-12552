// 烟线粒子：在 LBM 速度场上做半拉格朗日平流，带历史拖尾环。
// 粒子位置单位为 lattice 坐标（格子 i 占据 [i, i+1]，中心 i+0.5）。

struct ParamsSmoke {
  nx : u32,
  ny : u32,
  nz : u32,
  particleCount : u32,
  u0 : f32,
  speedScale : f32,   // 每渲染帧粒子前进的 lattice 时间步数（可 >1 做亚步）
  life : f32,
  nozzleX : f32,
  nozzleBaseY : f32,
  nozzleBaseZ : f32,
  nozzleStepY : f32,
  nozzleStepZ : f32,
  nozzleCols : u32,    // y 方向喷嘴数
  nozzleRows : u32,    // z 方向喷嘴数
  steps : u32,
};

struct Macro { rho : f32, ux : f32, uy : f32, uz : f32 };
struct Particle {
  pos : vec4<f32>,   // xyz + age
  hist0 : vec4<f32>,
  hist1 : vec4<f32>,
  hist2 : vec4<f32>,
  hist3 : vec4<f32>,
  hist4 : vec4<f32>,
  hist5 : vec4<f32>,
  hist6 : vec4<f32>,
  hist7 : vec4<f32>,
};

const TRAIL : i32 = 8;

@group(0) @binding(0) var<storage, read> macros : array<Macro>;
@group(0) @binding(1) var<storage, read> flags : array<u32>;
@group(0) @binding(2) var<storage, read_write> particles : array<Particle>;
@group(0) @binding(3) var<uniform> P : ParamsSmoke;
// 渲染阶段只读视图（同一粒子缓冲，单独的只读绑定，满足顶点阶段访问限制）
@group(0) @binding(4) var<storage, read> particlesRO : array<Particle>;

fn cid(i : i32, j : i32, k : i32) -> u32 {
  return u32(i) * (P.ny * P.nz) + u32(j) * P.nz + u32(k);
}

fn hash01(x : u32) -> f32 {
  var h = x;
  h = (h ^ 2747636419u) * 2654435769u;
  h = (h ^ (h >> 16)) * 2246822519u;
  h = h ^ (h >> 15);
  return f32(h) / 4294967296.0;
}

// 三线性插值采样速度；固体格与域外按 0 速度处理
fn sampleVel(p : vec3<f32>) -> vec3<f32> {
  let nx = i32(P.nx); let ny = i32(P.ny); let nz = i32(P.nz);
  let base = floor(p) - vec3<f32>(0.5);
  let i0 = clamp(i32(base.x), 0, nx - 2);
  let j0 = clamp(i32(base.y), 0, ny - 2);
  let k0 = clamp(i32(base.z), 0, nz - 2);
  let tx = clamp(p.x - f32(i0) - 0.5, 0.0, 1.0);
  let ty = clamp(p.y - f32(j0) - 0.5, 0.0, 1.0);
  let tz = clamp(p.z - f32(k0) - 0.5, 0.0, 1.0);

  var vel = vec3<f32>(0.0, 0.0, 0.0);
  var wSum : f32 = 0.0;
  for (var dz : i32 = 0; dz < 2; dz++) {
    for (var dy : i32 = 0; dy < 2; dy++) {
      for (var dx : i32 = 0; dx < 2; dx++) {
        let id = cid(i0 + dx, j0 + dy, k0 + dz);
        let wx = select(1.0 - tx, tx, dx == 1);
        let wy = select(1.0 - ty, ty, dy == 1);
        let wz = select(1.0 - tz, tz, dz == 1);
        let w = wx * wy * wz;
        // 只对流体邻居累加并做权重归一化，避免固体 0 速度稀释流场
        if (flags[id] == 0u) {
          let m = macros[id];
          vel += w * vec3<f32>(m.ux, m.uy, m.uz);
          wSum += w;
        }
      }
    }
  }
  if (wSum > 1e-6) { vel = vel / wSum; }
  return vel;
}

fn respawn(id : u32) -> vec4<f32> {
  let col = id % P.nozzleCols;
  let row = id / P.nozzleCols;
  let j1 = hash01(id * 7u + P.steps * 131u) - 0.5;
  let j2 = hash01(id * 13u + P.steps * 977u + 17u) - 0.5;
  let y = P.nozzleBaseY + f32(col) * P.nozzleStepY + j1 * P.nozzleStepY * 0.5;
  let z = P.nozzleBaseZ + f32(row) * P.nozzleStepZ + j2 * P.nozzleStepZ * 0.5;
  // 初始年龄错开，避免所有烟线同时重生
  let stagger = hash01(id * 31u + 3u) * P.life;
  return vec4<f32>(P.nozzleX, y, z, -stagger);
}

@compute @workgroup_size(128)
fn advect(@builtin(global_invocation_id) gid : vec3<u32>) {
  let id = gid.x;
  if (id >= P.particleCount) { return; }
  var pt = particles[id];

  // 拖尾环整体后移一格，hist0 记录上一帧位置
  pt.hist7 = pt.hist6; pt.hist6 = pt.hist5; pt.hist5 = pt.hist4;
  pt.hist4 = pt.hist3; pt.hist3 = pt.hist2; pt.hist2 = pt.hist1;
  pt.hist1 = pt.hist0; pt.hist0 = pt.pos;

  var p = pt.pos.xyz;
  // RK2，两个亚步
  let sub = 2.0;
  let dt = P.speedScale / sub;
  for (var s : i32 = 0; s < 2; s++) {
    let k1 = sampleVel(p);
    let k2 = sampleVel(p + 0.5 * dt * k1);
    p = p + dt * k2;
  }
  pt.pos = vec4<f32>(p, pt.pos.w + P.speedScale);

  // 出界 / 进固体 / 寿命到 → 回喷嘴
  let outOfDomain =
    p.x < 0.5 || p.x > f32(P.nx) - 0.5 ||
    p.y < 0.5 || p.y > f32(P.ny) - 0.5 ||
    p.z < 0.5 || p.z > f32(P.nz) - 0.5;
  var inSolid = false;
  if (!outOfDomain) {
    let ic = clamp(i32(p.x), 0, i32(P.nx) - 1);
    let jc = clamp(i32(p.y), 0, i32(P.ny) - 1);
    let kc = clamp(i32(p.z), 0, i32(P.nz) - 1);
    inSolid = flags[cid(ic, jc, kc)] == 1u;
  }
  let dead = pt.pos.w > P.life;
  if (outOfDomain || inSolid || dead) {
    let fresh = respawn(id);
    pt.pos = fresh;
    pt.hist0 = fresh; pt.hist1 = fresh; pt.hist2 = fresh; pt.hist3 = fresh;
    pt.hist4 = fresh; pt.hist5 = fresh; pt.hist6 = fresh; pt.hist7 = fresh;
  }
  particles[id] = pt;
}

// ---------- 渲染 ----------
struct Camera {
  viewProj : mat4x4<f32>,
  worldFromLattice : mat4x4<f32>,
  u0 : f32,
  trailAlpha : f32,
  // 表面相对流速色带的实测范围（CPU 按分位数每秒更新）
  speedMin : f32,
  speedMax : f32,
  // 世界空间相机右向量（ribbon 展开方向），xyz + 1 填充
  right : vec4<f32>,
};
// 相机在 binding 0：烟线/地面 BGL 与表面 BGL 都包含该绑定，指向同一 uniform 缓冲。
@group(0) @binding(0) var<uniform> cam : Camera;

struct VOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) alpha : f32,
};

// 烟丝带半宽（lattice 单位）。ribbon 沿相机右向量展开，形成有粗度的连续烟丝。
const RIBBON_HALF : f32 = 0.6;

// 每段历史点生成一个面向相机的四边形（2 三角形 = 6 顶点）。
// 顶点顺序：a左, b左, b右, a左, b右, a右
@vertex
fn vsSmoke(@builtin(vertex_index) vid : u32) -> VOut {
  let vertsPerSeg = 6u;
  let segVerts = vertsPerSeg * u32(TRAIL - 1);
  let pidx = vid / segVerts;
  let local = vid % segVerts;
  let seg = local / vertsPerSeg;
  let c = local % vertsPerSeg;

  let pt = particlesRO[pidx];
  let h = array<vec4<f32>, 9>(
    pt.pos, pt.hist0, pt.hist1, pt.hist2, pt.hist3,
    pt.hist4, pt.hist5, pt.hist6, pt.hist7);
  let a = h[seg].xyz;
  let b = h[seg + 1u].xyz;
  let right = normalize(cam.right.xyz) * RIBBON_HALF;
  let aL = a - right; let aR = a + right;
  let bL = b - right; let bR = b + right;

  var lattice = aL;
  if (c == 1u) { lattice = bL; }
  else if (c == 2u) { lattice = bR; }
  else if (c == 3u) { lattice = aL; }
  else if (c == 4u) { lattice = bR; }
  else if (c == 5u) { lattice = aR; }

  var out : VOut;
  out.clip = cam.viewProj * cam.worldFromLattice * vec4<f32>(lattice, 1.0);
  let fade = 1.0 - f32(seg) / f32(TRAIL);
  out.alpha = cam.trailAlpha * (0.35 + 0.65 * fade);
  return out;
}

@fragment
fn fsSmoke(in : VOut) -> @location(0) vec4<f32> {
  // 冷白烟丝，半透明
  return vec4<f32>(0.85, 0.9, 1.0, in.alpha);
}

// ---------- 车身表面实例化立方体 ----------
@group(0) @binding(5) var<storage, read> surfaceCells : array<u32>;
@group(0) @binding(6) var<storage, read> pressureBuf : array<u32>;

struct SOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) color : vec3<f32>,
  @location(1) normalW : vec3<f32>,
};

@vertex
fn vsSurface(
  @location(0) localPos : vec3<f32>,
  @location(1) localNormal : vec3<f32>,
  @builtin(instance_index) iid : u32,
) -> SOut {
  let cell = surfaceCells[iid];
  let nz = P.nz; let ny = P.ny;
  let k = i32(cell % nz);
  let j = i32((cell / nz) % ny);
  let i = i32(cell / (ny * nz));
  let center = vec3<f32>(f32(i) + 0.5, f32(j) + 0.5, f32(k) + 0.5);
  // 略放大 2% 遮盖相邻面缝隙
  let world = cam.worldFromLattice * vec4<f32>(center + localPos * 1.02, 1.0);

  let p = bitcast<f32>(pressureBuf[cell]);
  // extra 存表面相对流速 u/u0，按实测分位范围归一化：
  // 低速（驻点/尾流，高压感）→红，高速（加速区，低压）→蓝。
  let range = max(cam.speedMax - cam.speedMin, 1e-6);
  let t = 1.0 - clamp((p - cam.speedMin) / range, 0.0, 1.0);
  let col = pressureRamp(t);

  var out : SOut;
  out.clip = cam.viewProj * world;
  out.color = col;
  out.normalW = normalize((cam.worldFromLattice * vec4<f32>(localNormal, 0.0)).xyz);
  return out;
}

fn pressureRamp(t : f32) -> vec3<f32> {
  // 深蓝 → 青 → 绿 → 黄 → 红
  let stops = array<vec3<f32>, 5>(
    vec3<f32>(0.10, 0.18, 0.65),
    vec3<f32>(0.10, 0.65, 0.85),
    vec3<f32>(0.25, 0.75, 0.35),
    vec3<f32>(0.95, 0.80, 0.20),
    vec3<f32>(0.85, 0.15, 0.12));
  let x = clamp(t, 0.0, 0.9999) * 4.0;
  let idx = i32(x);
  let f = x - f32(idx);
  return mix(stops[idx], stops[idx + 1], f);
}

@fragment
fn fsSurface(in : SOut) -> @location(0) vec4<f32> {
  let light = normalize(vec3<f32>(0.4, 0.8, 0.5));
  let shade = 0.55 + 0.45 * abs(dot(normalize(in.normalW), light));
  return vec4<f32>(in.color * shade, 1.0);
}

// ---------- 地面网格 ----------
struct GOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) uv : vec2<f32>,
};
@vertex
fn vsGround(@builtin(vertex_index) vid : u32) -> GOut {
  // 两个三角形拼成大平面，y=0
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0));
  let c = corners[vid];
  let lattice = vec3<f32>(
    (c.x * 0.5 + 0.5) * f32(P.nx),
    0.0,
    (c.y * 0.5 + 0.5) * f32(P.nz));
  var out : GOut;
  out.clip = cam.viewProj * cam.worldFromLattice * vec4<f32>(lattice, 1.0);
  out.uv = lattice.xz;
  return out;
}

@fragment
fn fsGround(in : GOut) -> @location(0) vec4<f32> {
  let gx = abs(fract(in.uv.x * 0.25) - 0.5);
  let gz = abs(fract(in.uv.y * 0.25) - 0.5);
  let line = smoothstep(0.46, 0.5, max(gx, gz));
  let base = vec3<f32>(0.08, 0.09, 0.12);
  return vec4<f32>(base + vec3<f32>(0.12) * line, 1.0);
}
