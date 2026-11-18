// 烟线:粒子沿流场平流(RK2),历史环形缓冲 → line-list 半透明细丝
// 喷嘴在车头前方排成一排(8×16 网格),粒子走出计算域或超龄后回喷嘴重生
export class SmokeSystem {
  readonly NPART: number;
  readonly HIST: number;
  private device: GPUDevice;
  private partBuf: GPUBuffer;
  private histBuf: GPUBuffer;
  private paramsBuf: GPUBuffer;
  private slotBuf: GPUBuffer;
  private advectPipe: GPUComputePipeline;
  private advectBG: GPUBindGroup;
  private renderPipe: GPURenderPipeline;
  private renderBG: GPUBindGroup;
  private writeSlot = 0;

  constructor(
    device: GPUDevice,
    private nx: number,
    private ny: number,
    private nz: number,
    macroBuf: GPUBuffer,
    flagsBuf: GPUBuffer,
    camBuf: GPUBuffer,
    format: GPUTextureFormat,
    npart = 2048,
    hist = 48,
  ) {
    this.device = device;
    this.NPART = npart;
    this.HIST = hist;

    this.partBuf = device.createBuffer({
      size: npart * 32,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.histBuf = device.createBuffer({ size: npart * hist * 16, usage: GPUBufferUsage.STORAGE });
    this.paramsBuf = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.slotBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    // 初始粒子:沿 x 散布,年龄错开,避免开局整齐划一
    const init = new ArrayBuffer(npart * 32);
    const fv = new Float32Array(init);
    const uv = new Uint32Array(init);
    for (let i = 0; i < npart; i++) {
      const nz2 = i % 128;
      const row = (nz2 / 16) | 0, col = nz2 % 16;
      fv[i * 8 + 0] = 8 + Math.random() * (nx - 30);
      fv[i * 8 + 1] = 4 + row * 7 + Math.random() * 3;
      fv[i * 8 + 2] = 20 + col * 3.6 + Math.random() * 2;
      fv[i * 8 + 3] = Math.random() * 260; // age
      uv[i * 8 + 4] = nz2; // nozzle
      uv[i * 8 + 5] = (i * 2654435761) >>> 0; // seed
    }
    device.queue.writeBuffer(this.partBuf, 0, init);

    const advectCode = `
const NX: u32 = ${nx}u;
const NY: u32 = ${ny}u;
const NZ: u32 = ${nz}u;
const NPART: u32 = ${npart}u;
const HIST: u32 = ${hist}u;
struct Particle { pos: vec3f, age: f32, nozzle: u32, seed: u32, pad: f32 };
struct PParams { dt: f32, writeSlot: u32, maxAge: f32, frame: u32 };
@group(0) @binding(0) var<storage, read> macro: array<vec4f>;
@group(0) @binding(1) var<storage, read> flags: array<u32>;
@group(0) @binding(2) var<storage, read_write> parts: array<Particle>;
@group(0) @binding(3) var<storage, read_write> hist: array<vec4f>;
@group(0) @binding(4) var<uniform> pp: PParams;

fn cid(x: u32, y: u32, z: u32) -> u32 { return x + y * NX + z * NX * NY; }
fn hashU(n0: u32) -> u32 {
  var n = n0;
  n ^= n >> 16u; n *= 2246822519u; n ^= n >> 13u; n *= 3266489917u; n ^= n >> 16u;
  return n;
}
fn rand(s: u32) -> f32 { return f32(hashU(s) & 0xFFFFFFu) / 16777216.0; }

fn sampleU(p: vec3f) -> vec3f {
  let pc = clamp(p, vec3f(1.0), vec3f(f32(NX) - 2.001, f32(NY) - 2.001, f32(NZ) - 2.001));
  let i0 = vec3u(floor(pc));
  let fr = pc - floor(pc);
  var acc = vec3f(0.0);
  for (var dz = 0u; dz < 2u; dz++) {
    for (var dy = 0u; dy < 2u; dy++) {
      for (var dx = 0u; dx < 2u; dx++) {
        let w = select(1.0 - fr.x, fr.x, dx == 1u) * select(1.0 - fr.y, fr.y, dy == 1u) * select(1.0 - fr.z, fr.z, dz == 1u);
        acc += w * macro[cid(i0.x + dx, i0.y + dy, i0.z + dz)].xyz;
      }
    }
  }
  return acc;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= NPART) { return; }
  var p = parts[i];
  var pos = p.pos;
  // 子步进,防止大步长穿过车身
  var dt = min(pp.dt, 24.0);
  let nsub = max(1u, u32(ceil(dt / 4.0)));
  dt /= f32(nsub);
  for (var s = 0u; s < nsub; s++) {
    let u1 = sampleU(pos);
    let u2 = sampleU(pos + 0.5 * dt * u1);
    pos += dt * u2;
  }
  p.age += 1.0;
  var dead = pos.x > f32(NX - 4u) || pos.x < 1.0 || pos.y < 1.0 || pos.y > f32(NY - 2u) ||
             pos.z < 1.0 || pos.z > f32(NZ - 2u) || p.age > pp.maxAge;
  if (!dead) {
    let c = cid(u32(pos.x), u32(pos.y), u32(pos.z));
    if (flags[c] == 1u) { dead = true; }
  }
  if (dead) {
    let row = p.nozzle / 16u;
    let col = p.nozzle % 16u;
    let r1 = rand(p.seed + pp.frame * 747u);
    let r2 = rand(p.seed + pp.frame * 1511u + 7u);
    let r3 = rand(p.seed + pp.frame * 2281u + 13u);
    pos = vec3f(8.0 + r1 * 2.0, 4.0 + f32(row) * 7.0 + r2 * 3.0, 20.0 + f32(col) * 3.6 + r3 * 2.0);
    p.age = r2 * 12.0; // 错开重生
    p.seed = hashU(p.seed + pp.frame);
  }
  p.pos = pos;
  parts[i] = p;
  hist[i * HIST + pp.writeSlot] = vec4f(pos, 1.0);
}`;

    this.advectPipe = device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: advectCode }), entryPoint: 'main' },
    });
    this.advectBG = device.createBindGroup({
      layout: this.advectPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: macroBuf } },
        { binding: 1, resource: { buffer: flagsBuf } },
        { binding: 2, resource: { buffer: this.partBuf } },
        { binding: 3, resource: { buffer: this.histBuf } },
        { binding: 4, resource: { buffer: this.paramsBuf } },
      ],
    });

    // 渲染:每个历史线段 2 个顶点,从环形缓冲直接取数
    const renderCode = `
const NPART: u32 = ${npart}u;
const HIST: u32 = ${hist}u;
struct Cam { vp: mat4x4f, eye: vec3f };
@group(0) @binding(0) var<uniform> cam: Cam;
@group(0) @binding(1) var<storage, read> hist: array<vec4f>;
@group(0) @binding(2) var<uniform> slotU: vec4u;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) alpha: f32,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  let seg = vi / 2u;
  let e = vi % 2u;
  let p = seg / (HIST - 1u);
  let s = seg % (HIST - 1u);
  let slotA = (slotU.x + HIST - s) % HIST;       // 较新
  let slotB = (slotU.x + HIST - s - 1u) % HIST;  // 较旧
  let A = hist[p * HIST + slotA].xyz;
  let B = hist[p * HIST + slotB].xyz;
  var wp = select(A, B, e == 1u);
  var alpha = 0.5 * (1.0 - f32(s) / f32(HIST));
  if (distance(A, B) > 2.5) { alpha = 0.0; wp = vec3f(0.0); } // 重生跳变,隐藏
  var o: VOut;
  o.pos = cam.vp * vec4f(wp, 1.0);
  o.alpha = alpha;
  return o;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4f {
  return vec4f(0.72, 0.82, 0.95, in.alpha);
}`;

    this.renderPipe = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: device.createShaderModule({ code: renderCode }), entryPoint: 'vs' },
      fragment: {
        module: device.createShaderModule({ code: renderCode }),
        entryPoint: 'fs',
        targets: [
          {
            format,
            blend: {
              color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
              alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            },
          },
        ],
      },
      primitive: { topology: 'line-list' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less' },
    });
    this.renderBG = device.createBindGroup({
      layout: this.renderPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: camBuf } },
        { binding: 1, resource: { buffer: this.histBuf } },
        { binding: 2, resource: { buffer: this.slotBuf } },
      ],
    });
  }

  private frame = 0;

  // 每帧:平流 + 推进环形写指针
  encodeAdvect(encoder: GPUCommandEncoder, dt: number) {
    this.writeSlot = (this.writeSlot + 1) % this.HIST;
    this.frame++;
    const buf = new ArrayBuffer(16);
    const f = new Float32Array(buf);
    const u = new Uint32Array(buf);
    f[0] = dt;
    u[1] = this.writeSlot;
    f[2] = 260; // maxAge
    u[3] = this.frame;
    this.device.queue.writeBuffer(this.paramsBuf, 0, buf);
    this.device.queue.writeBuffer(this.slotBuf, 0, new Uint32Array([this.writeSlot, 0, 0, 0]));
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.advectPipe);
    pass.setBindGroup(0, this.advectBG);
    pass.dispatchWorkgroups(Math.ceil(this.NPART / 64));
    pass.end();
  }

  // 启动前把历史缓冲填满,避免开局出现乱线
  prime() {
    for (let k = 0; k < this.HIST; k++) {
      const enc = this.device.createCommandEncoder();
      this.encodeAdvect(enc, 1);
      this.device.queue.submit([enc.finish()]);
    }
  }

  encodeRender(pass: GPURenderPassEncoder) {
    pass.setPipeline(this.renderPipe);
    pass.setBindGroup(0, this.renderBG);
    pass.draw((this.HIST - 1) * 2 * this.NPART);
  }
}
