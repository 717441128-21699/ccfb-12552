// D3Q19 格子玻尔兹曼求解器(WebGPU compute)
// - BGK 碰撞 + Smagorinsky 亚格子应力(高雷诺数稳定)
// - push 式碰撞-迁移融合 kernel,固体反弹边界,动量交换法积分气动力
// - 边界:左侧平衡态入口,右侧零梯度出口,其余四壁自由滑移(镜面反射)
// - 障碍物只改 flags 掩码,流场状态保留;新露出的格子单独置平衡态

const DIRS: [number, number, number][] = [
  [0, 0, 0],
  [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
  [1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0],
  [1, 0, 1], [-1, 0, -1], [1, 0, -1], [-1, 0, 1],
  [0, 1, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1],
];
const WEIGHTS = [1 / 3, ...Array(6).fill(1 / 18), ...Array(12).fill(1 / 36)];
const findDir = (x: number, y: number, z: number) =>
  DIRS.findIndex((d) => d[0] === x && d[1] === y && d[2] === z);
const OPP = DIRS.map((d) => findDir(-d[0], -d[1], -d[2]));
const MIRY = DIRS.map((d) => findDir(d[0], -d[1], d[2]));
const MIRZ = DIRS.map((d) => findDir(d[0], d[1], -d[2]));

export const FLAG_FLUID = 0;
export const FLAG_SOLID = 1;
export const FLAG_INLET = 2;
export const FLAG_OUTLET = 3;
export const FLAG_WALL = 4;

export const FORCE_SCALE = 4096; // 定点力累加缩放

function wgslPrelude(nx: number, ny: number, nz: number): string {
  const v = (a: number[]) => a.map((x) => x + 'u').join(',');
  return `
const NX: u32 = ${nx}u;
const NY: u32 = ${ny}u;
const NZ: u32 = ${nz}u;
const NCELLS: u32 = ${nx * ny * nz}u;
const C_DIRS = array<vec3i, 19>(${DIRS.map((d) => `vec3i(${d[0]},${d[1]},${d[2]})`).join(',')});
const W = array<f32, 19>(${WEIGHTS.map((w) => w.toFixed(8)).join(',')});
const OPP = array<u32, 19>(${v(OPP)});
const MIRY = array<u32, 19>(${v(MIRY)});
const MIRZ = array<u32, 19>(${v(MIRZ)});
fn cellId(x: u32, y: u32, z: u32) -> u32 { return x + y * NX + z * NX * NY; }
fn feq(i: u32, rho: f32, u: vec3f) -> f32 {
  let c = vec3f(C_DIRS[i]);
  let cu = dot(c, u);
  return W[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * dot(u, u));
}
`;
}

export class LBM {
  readonly nx: number;
  readonly ny: number;
  readonly nz: number;
  readonly cells: number;
  private device: GPUDevice;

  fA: GPUBuffer; // 分布函数双缓冲
  fB: GPUBuffer;
  flagsBuf: GPUBuffer;
  macroBuf: GPUBuffer; // vec4f: u.xyz + rho
  private forceBuf: GPUBuffer;
  private staging: GPUBuffer[];
  private stagingFree = [true, true];
  private paramsBuf: GPUBuffer;
  private initListBuf: GPUBuffer;
  private initCountBuf: GPUBuffer;
  private readonly INIT_CAP = 262144;

  private stepPipe: GPUComputePipeline;
  private boundaryPipe: GPUComputePipeline;
  private macroPipe: GPUComputePipeline;
  private initPipe: GPUComputePipeline;
  private initCellsPipe: GPUComputePipeline;
  private stepBG: GPUBindGroup[];
  private boundaryBG: GPUBindGroup[];
  private macroBG: GPUBindGroup[];

  private cur = 0; // 当前 f 缓冲:0 = A,1 = B
  private flagsCPU: Uint32Array<ArrayBuffer>;
  private frameParity = 0;

  u0 = 0.08;
  nu = 0.0008; // 格子运动粘度
  private readonly CSMAG = 0.15;

  constructor(device: GPUDevice, nx: number, ny: number, nz: number) {
    this.device = device;
    this.nx = nx;
    this.ny = ny;
    this.nz = nz;
    this.cells = nx * ny * nz;
    const fBytes = this.cells * 19 * 4;

    const mk = (size: number, usage: GPUBufferUsageFlags) =>
      device.createBuffer({ size, usage });
    this.fA = mk(fBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.fB = mk(fBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.flagsBuf = mk(this.cells * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.macroBuf = mk(this.cells * 16, GPUBufferUsage.STORAGE);
    this.forceBuf = mk(16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    this.staging = [0, 1].map(() => mk(16, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST));
    this.paramsBuf = mk(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.initListBuf = mk(this.INIT_CAP * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.initCountBuf = mk(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

    const pre = wgslPrelude(nx, ny, nz);

    // ---- 主 kernel:碰撞 + 迁移(push)+ 动量交换测力
    const stepCode = pre + `
struct SimParams { tau: f32, u0: f32, css: f32, cells: u32 };
@group(0) @binding(0) var<storage, read> fIn: array<f32>;
@group(0) @binding(1) var<storage, read_write> fOut: array<f32>;
@group(0) @binding(2) var<storage, read> flags: array<u32>;
@group(0) @binding(3) var<storage, read_write> force: array<atomic<i32>>;
@group(0) @binding(4) var<uniform> params: SimParams;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let id = gid.x;
  if (id >= NCELLS) { return; }
  if (flags[id] != 0u) { return; } // 只算流体格子
  let base = id * 19u;
  var f: array<f32, 19>;
  var rho = 0.0;
  var mom = vec3f(0.0);
  for (var i = 0u; i < 19u; i++) {
    f[i] = fIn[base + i];
    rho += f[i];
    mom += vec3f(C_DIRS[i]) * f[i];
  }
  let u = mom / rho;
  var fe: array<f32, 19>;
  for (var i = 0u; i < 19u; i++) { fe[i] = feq(i, rho, u); }
  // 非平衡应力二阶矩 → Smagorinsky 有效松弛时间
  var pxx = 0.0; var pyy = 0.0; var pzz = 0.0;
  var pxy = 0.0; var pxz = 0.0; var pyz = 0.0;
  for (var i = 0u; i < 19u; i++) {
    let fn = f[i] - fe[i];
    let c = vec3f(C_DIRS[i]);
    pxx += fn * c.x * c.x; pyy += fn * c.y * c.y; pzz += fn * c.z * c.z;
    pxy += fn * c.x * c.y; pxz += fn * c.x * c.z; pyz += fn * c.y * c.z;
  }
  let piMag = sqrt(2.0 * (pxx * pxx + pyy * pyy + pzz * pzz) + 4.0 * (pxy * pxy + pxz * pxz + pyz * pyz));
  let tau0 = params.tau;
  let tauEff = 0.5 * (tau0 + sqrt(tau0 * tau0 + params.css * piMag / rho));
  let omega = 1.0 / tauEff;

  let x = i32(id % NX);
  let y = i32((id / NX) % NY);
  let z = i32(id / (NX * NY));
  var forceV = vec3f(0.0);
  for (var i = 0u; i < 19u; i++) {
    let fp = f[i] - omega * (f[i] - fe[i]);
    let ci = C_DIRS[i];
    let nx = x + ci.x; let ny = y + ci.y; let nz = z + ci.z;
    if (nx < 0 || nx >= i32(NX) || ny < 0 || ny >= i32(NY) || nz < 0 || nz >= i32(NZ)) { continue; }
    let nid = cellId(u32(nx), u32(ny), u32(nz));
    if (flags[nid] == 1u) {
      // 固体:反弹回自己,动量差即作用力
      fOut[base + OPP[i]] = fp;
      forceV += 2.0 * vec3f(ci) * fp;
    } else {
      fOut[nid * 19u + i] = fp;
    }
  }
  if (forceV.x != 0.0 || forceV.y != 0.0 || forceV.z != 0.0) {
    atomicAdd(&force[0], i32(forceV.x * ${FORCE_SCALE}.0));
    atomicAdd(&force[1], i32(forceV.y * ${FORCE_SCALE}.0));
    atomicAdd(&force[2], i32(forceV.z * ${FORCE_SCALE}.0));
  }
}`;

    // ---- 边界 kernel:入口 / 出口 / 滑移壁面(主 kernel 之后,读写同一 fOut)
    const boundaryCode = pre + `
struct SimParams { tau: f32, u0: f32, css: f32, cells: u32 };
@group(0) @binding(0) var<storage, read> fIn: array<f32>;
@group(0) @binding(1) var<storage, read_write> fOut: array<f32>;
@group(0) @binding(2) var<storage, read> flags: array<u32>;
@group(0) @binding(3) var<uniform> params: SimParams;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let id = gid.x;
  if (id >= NCELLS) { return; }
  let x = id % NX;
  let y = (id / NX) % NY;
  let z = id / (NX * NY);
  let base = id * 19u;
  let uIn = vec3f(params.u0, 0.0, 0.0);
  if (x == 0u) { // 入口:平衡态
    for (var i = 0u; i < 19u; i++) { fOut[base + i] = feq(i, 1.0, uIn); }
    return;
  }
  if (x == NX - 1u) { // 出口:零梯度(拷贝上游邻居)
    let src = base - 19u;
    for (var i = 0u; i < 19u; i++) { fOut[base + i] = fIn[src + i]; }
    return;
  }
  if (flags[id] != 0u) { return; }
  // 入口旁的流体格子:指向下游的分布没人写入,用入口平衡态补
  if (x == 1u) {
    for (var i = 0u; i < 19u; i++) {
      if (C_DIRS[i].x > 0) { fOut[base + i] = feq(i, 1.0, uIn); }
    }
  }
  // 出口旁的流体格子:指向上游的分布用自身宏观量的平衡态补
  if (x == NX - 2u) {
    var rho = 0.0; var mom = vec3f(0.0);
    for (var i = 0u; i < 19u; i++) {
      rho += fIn[base + i];
      mom += vec3f(C_DIRS[i]) * fIn[base + i];
    }
    let u = mom / rho;
    for (var i = 0u; i < 19u; i++) {
      if (C_DIRS[i].x < 0) { fOut[base + i] = feq(i, rho, u); }
    }
  }
  // 滑移壁面:法向分量镜面反射(读主 kernel 写好的同格分布)
  if (y == 1u) {
    for (var i = 0u; i < 19u; i++) {
      if (C_DIRS[i].y > 0) { fOut[base + i] = fOut[base + MIRY[i]]; }
    }
  }
  if (y == NY - 2u) {
    for (var i = 0u; i < 19u; i++) {
      if (C_DIRS[i].y < 0) { fOut[base + i] = fOut[base + MIRY[i]]; }
    }
  }
  if (z == 1u) {
    for (var i = 0u; i < 19u; i++) {
      if (C_DIRS[i].z > 0) { fOut[base + i] = fOut[base + MIRZ[i]]; }
    }
  }
  if (z == NZ - 2u) {
    for (var i = 0u; i < 19u; i++) {
      if (C_DIRS[i].z < 0) { fOut[base + i] = fOut[base + MIRZ[i]]; }
    }
  }
}`;

    // ---- 宏观量 kernel:rho / u → macro 缓冲(供粒子、云图、表面着色)
    const macroCode = pre + `
@group(0) @binding(0) var<storage, read> fIn: array<f32>;
@group(0) @binding(1) var<storage, read> flags: array<u32>;
@group(0) @binding(2) var<storage, read_write> macro: array<vec4f>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let id = gid.x;
  if (id >= NCELLS) { return; }
  let fl = flags[id];
  if (fl == 1u || fl == 4u) {
    macro[id] = vec4f(0.0, 0.0, 0.0, 1.0);
    return;
  }
  let base = id * 19u;
  var rho = 0.0; var mom = vec3f(0.0);
  for (var i = 0u; i < 19u; i++) {
    let fi = fIn[base + i];
    rho += fi;
    mom += vec3f(C_DIRS[i]) * fi;
  }
  macro[id] = vec4f(mom / rho, rho);
}`;

    // ---- 全场初始化:均匀来流平衡态
    const initCode = pre + `
@group(0) @binding(0) var<storage, read_write> fA: array<f32>;
@group(0) @binding(1) var<storage, read_write> fB: array<f32>;
@group(0) @binding(2) var<uniform> u0u: vec4f;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let id = gid.x;
  if (id >= NCELLS) { return; }
  let u = vec3f(u0u.x, 0.0, 0.0);
  for (var i = 0u; i < 19u; i++) {
    let v = feq(i, 1.0, u);
    fA[id * 19u + i] = v;
    fB[id * 19u + i] = v;
  }
}`;

    // ---- 局部初始化:重新体素化后新露出的格子置平衡态(流场其余部分不动)
    const initCellsCode = pre + `
@group(0) @binding(0) var<storage, read_write> fA: array<f32>;
@group(0) @binding(1) var<storage, read_write> fB: array<f32>;
@group(0) @binding(2) var<storage, read> list: array<u32>;
@group(0) @binding(3) var<uniform> u0u: vec4f; // x: u0, y: count

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let k = gid.x;
  if (k >= u32(u0u.y)) { return; }
  let id = list[k];
  let u = vec3f(u0u.x, 0.0, 0.0);
  for (var i = 0u; i < 19u; i++) {
    let v = feq(i, 1.0, u);
    fA[id * 19u + i] = v;
    fB[id * 19u + i] = v;
  }
}`;

    const mkPipe = (code: string) =>
      device.createComputePipeline({
        layout: 'auto',
        compute: { module: device.createShaderModule({ code }), entryPoint: 'main' },
      });
    this.stepPipe = mkPipe(stepCode);
    this.boundaryPipe = mkPipe(boundaryCode);
    this.macroPipe = mkPipe(macroCode);
    this.initPipe = mkPipe(initCode);
    this.initCellsPipe = mkPipe(initCellsCode);

    const fBufs = [this.fA, this.fB];
    this.stepBG = [0, 1].map((c) =>
      device.createBindGroup({
        layout: this.stepPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: fBufs[c] } },
          { binding: 1, resource: { buffer: fBufs[1 - c] } },
          { binding: 2, resource: { buffer: this.flagsBuf } },
          { binding: 3, resource: { buffer: this.forceBuf } },
          { binding: 4, resource: { buffer: this.paramsBuf } },
        ],
      }),
    );
    this.boundaryBG = [0, 1].map((c) =>
      device.createBindGroup({
        layout: this.boundaryPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: fBufs[c] } },
          { binding: 1, resource: { buffer: fBufs[1 - c] } },
          { binding: 2, resource: { buffer: this.flagsBuf } },
          { binding: 3, resource: { buffer: this.paramsBuf } },
        ],
      }),
    );
    this.macroBG = [0, 1].map((c) =>
      device.createBindGroup({
        layout: this.macroPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: fBufs[c] } },
          { binding: 1, resource: { buffer: this.flagsBuf } },
          { binding: 2, resource: { buffer: this.macroBuf } },
        ],
      }),
    );

    // 初始 flags:边界层
    this.flagsCPU = new Uint32Array(this.cells);
    this.rebuildBoundaryFlags();
    device.queue.writeBuffer(this.flagsBuf, 0, this.flagsCPU);
    this.uploadParams();
    this.resetFlow();
  }

  private rebuildBoundaryFlags() {
    const { nx, ny, nz, flagsCPU: fl } = this;
    fl.fill(FLAG_FLUID);
    for (let z = 0; z < nz; z++)
      for (let y = 0; y < ny; y++) {
        fl[y * nx + z * nx * ny] = FLAG_INLET;
        fl[nx - 1 + y * nx + z * nx * ny] = FLAG_OUTLET;
      }
    for (let z = 0; z < nz; z++)
      for (let x = 0; x < nx; x++) {
        fl[x + z * nx * ny] = FLAG_WALL;
        fl[x + (ny - 1) * nx + z * nx * ny] = FLAG_WALL;
      }
    for (let y = 0; y < ny; y++)
      for (let x = 0; x < nx; x++) {
        fl[x + y * nx] = FLAG_WALL;
        fl[x + y * nx + (nz - 1) * nx * ny] = FLAG_WALL;
      }
  }

  private uploadParams() {
    const tau = 0.5 + 3 * this.nu;
    const css = 18 * Math.SQRT2 * this.CSMAG * this.CSMAG;
    this.device.queue.writeBuffer(
      this.paramsBuf,
      0,
      new Float32Array([tau, this.u0, css, this.cells]),
    );
  }

  setFlow(u0: number, nu: number) {
    this.u0 = u0;
    this.nu = nu;
    this.uploadParams();
  }

  // 全流场重置为均匀来流
  resetFlow() {
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.initPipe);
    pass.setBindGroup(
      0,
      this.device.createBindGroup({
        layout: this.initPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.fA } },
          { binding: 1, resource: { buffer: this.fB } },
          { binding: 2, resource: { buffer: this.paramsBuf } },
        ],
      }),
    );
    pass.dispatchWorkgroups(Math.ceil(this.cells / 128));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    this.cur = 0;
  }

  // 更新障碍物掩码:流场状态保留,只有 固体→流体 的格子被置为平衡态
  updateObstacles(mask: Uint8Array<ArrayBuffer>) {
    const { nx, ny, nz } = this;
    const newlyFluid: number[] = [];
    for (let z = 1; z < nz - 1; z++)
      for (let y = 1; y < ny - 1; y++)
        for (let x = 1; x < nx - 1; x++) {
          const id = x + y * nx + z * nx * ny;
          const wasSolid = this.flagsCPU[id] === FLAG_SOLID;
          const isSolid = mask[id] === 1;
          if (wasSolid && !isSolid) newlyFluid.push(id);
          this.flagsCPU[id] = isSolid ? FLAG_SOLID : FLAG_FLUID;
        }
    this.device.queue.writeBuffer(this.flagsBuf, 0, this.flagsCPU);
    if (newlyFluid.length === 0) return;
    if (newlyFluid.length > this.INIT_CAP) {
      this.resetFlow(); // 变化太大,直接重置更划算
      return;
    }
    this.device.queue.writeBuffer(this.initListBuf, 0, new Uint32Array(newlyFluid));
    this.device.queue.writeBuffer(
      this.initCountBuf,
      0,
      new Float32Array([this.u0, newlyFluid.length, 0, 0]),
    );
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.initCellsPipe);
    pass.setBindGroup(
      0,
      this.device.createBindGroup({
        layout: this.initCellsPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.fA } },
          { binding: 1, resource: { buffer: this.fB } },
          { binding: 2, resource: { buffer: this.initListBuf } },
          { binding: 3, resource: { buffer: this.initCountBuf } },
        ],
      }),
    );
    pass.dispatchWorkgroups(Math.ceil(newlyFluid.length / 64));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  // 推进 n 步(同一 encoder 内),并在末尾把力累加器拷到读回缓冲
  encodeSteps(encoder: GPUCommandEncoder, n: number) {
    this.device.queue.writeBuffer(this.forceBuf, 0, new Int32Array(4));
    const wg = Math.ceil(this.cells / 128);
    const pass = encoder.beginComputePass();
    for (let s = 0; s < n; s++) {
      pass.setPipeline(this.stepPipe);
      pass.setBindGroup(0, this.stepBG[this.cur]);
      pass.dispatchWorkgroups(wg);
      pass.setPipeline(this.boundaryPipe);
      pass.setBindGroup(0, this.boundaryBG[this.cur]);
      pass.dispatchWorkgroups(wg);
      this.cur = 1 - this.cur;
    }
    pass.setPipeline(this.macroPipe);
    pass.setBindGroup(0, this.macroBG[this.cur]);
    pass.dispatchWorkgroups(wg);
    pass.end();
    this.frameParity = 1 - this.frameParity;
    encoder.copyBufferToBuffer(this.forceBuf, 0, this.staging[this.frameParity], 0, 16);
    this.lastSteps = n;
  }

  private lastSteps = 0;

  // 读取上一帧的力(定点 → 浮点,除以步数得单步平均)
  async readForce(): Promise<{ fx: number; fy: number; fz: number } | null> {
    const idx = this.frameParity;
    if (!this.stagingFree[idx]) return null;
    this.stagingFree[idx] = false;
    const buf = this.staging[idx];
    await buf.mapAsync(GPUMapMode.READ);
    const v = new Int32Array(buf.getMappedRange().slice(0));
    buf.unmap();
    this.stagingFree[idx] = true;
    const steps = Math.max(1, this.lastSteps);
    return {
      fx: v[0] / FORCE_SCALE / steps,
      fy: v[1] / FORCE_SCALE / steps,
      fz: v[2] / FORCE_SCALE / steps,
    };
  }
}
