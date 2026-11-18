/**
 * WebGPU D3Q19 LBM 求解器。
 * 职责：GPU 缓冲/管线生命周期、每时间步的双缓冲调度、
 * 重新体素化的热启动调和、气动力读回与 Cd 计算。
 */
import lbmWgsl from './lbm.wgsl?raw';
import { Q, paramsBufferData, type SimParams } from '../lbm/constants';
import { FLUID } from '../lbm/constants';

export interface ForceSample {
  /** 压差阻力（lattice 动量单位） */
  pressure: number;
  /** 摩擦阻力 */
  viscous: number;
  total: number;
}

export interface SolverConfig {
  nx: number;
  ny: number;
  nz: number;
  u0: number;
  tau: number;
  smagorinsky: number;
  /** 计算工作组尺寸（每维线程数）；z 向取长条形以贴合连续内存布局 */
  wgX?: number;
  wgY?: number;
  wgZ?: number;
}

/** 默认工作组：32 线程、z 向长条，在集显/独显上都有较好占用与访存合并 */
export const DEFAULT_WORKGROUP = { wgX: 1, wgY: 2, wgZ: 16 } as const;

export class LbmSolver {
  readonly device: GPUDevice;
  readonly context: GPUCanvasContext;
  readonly nx: number;
  readonly ny: number;
  readonly nz: number;
  readonly count: number;

  private fA: GPUBuffer;
  private fB: GPUBuffer;
  private flags: GPUBuffer;
  private macro: GPUBuffer;
  private extra: GPUBuffer;
  private forceAcc: GPUBuffer;
  private forceReadback: GPUBuffer;
  private speedReadback?: GPUBuffer;
  private paramsBuf: GPUBuffer;
  private params: SimParams;

  private collidePipe: GPUComputePipeline;
  private streamPipe: GPUComputePipeline;
  private clearPipe: GPUComputePipeline;
  private resetPipe: GPUComputePipeline;
  private reconcilePipe: GPUComputePipeline;
  private pressurePipe: GPUComputePipeline;

  private bindLayout: GPUBindGroupLayout;
  private bgAB!: GPUBindGroup;
  private bgReconcile!: GPUBindGroup;
  private bgPressure!: GPUBindGroup;

  steps = 0;
  /** 自力累加器上次清零以来推进的步数，用于把累积力归一化为单步平均 */
  private forceAccumSteps = 0;
  private readonly wgX: number;
  private readonly wgY: number;
  private readonly wgZ: number;

  constructor(device: GPUDevice, context: GPUCanvasContext, cfg: SolverConfig) {
    this.device = device;
    this.context = context;
    this.nx = cfg.nx;
    this.ny = cfg.ny;
    this.nz = cfg.nz;
    this.count = cfg.nx * cfg.ny * cfg.nz;
    this.wgX = cfg.wgX ?? DEFAULT_WORKGROUP.wgX;
    this.wgY = cfg.wgY ?? DEFAULT_WORKGROUP.wgY;
    this.wgZ = cfg.wgZ ?? DEFAULT_WORKGROUP.wgZ;

    const N = this.count;
    const fUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    this.fA = device.createBuffer({ size: N * Q * 4, usage: fUsage });
    this.fB = device.createBuffer({ size: N * Q * 4, usage: fUsage });
    this.flags = device.createBuffer({
      size: N * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.macro = device.createBuffer({
      size: N * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    // 调和时存旧 flags，压力计算时复用为 f32 压力缓冲
    this.extra = device.createBuffer({
      size: N * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.forceAcc = device.createBuffer({
      size: 6 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.forceReadback = device.createBuffer({
      size: 6 * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    this.params = {
      nx: cfg.nx, ny: cfg.ny, nz: cfg.nz,
      u0: cfg.u0, tau: cfg.tau,
      smagorinsky: cfg.smagorinsky,
      steps: 0,
      forceScale: 1,
    };
    this.paramsBuf = device.createBuffer({
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.writeParams();

    const mod = device.createShaderModule({ code: lbmWgsl });
    this.bindLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [this.bindLayout] });
    const constants = { WG_X: this.wgX, WG_Y: this.wgY, WG_Z: this.wgZ };
    const mk = (entry: string) =>
      device.createComputePipeline({
        layout,
        compute: { module: mod, entryPoint: entry, constants },
      });
    this.collidePipe = mk('collide');
    this.streamPipe = mk('stream');
    this.clearPipe = mk('clearForce');
    this.resetPipe = mk('resetField');
    this.reconcilePipe = mk('reconcileVoxels');
    this.pressurePipe = mk('surfacePressure');

    this.rebuildBindGroups();
  }

  private writeParams(): void {
    this.device.queue.writeBuffer(this.paramsBuf, 0, paramsBufferData(this.params));
  }

  /** 三维网格对应的工作组派发数 */
  private grid(): [number, number, number] {
    return [
      Math.ceil(this.nx / this.wgX),
      Math.ceil(this.ny / this.wgY),
      Math.ceil(this.nz / this.wgZ),
    ];
  }

  private rebuildBindGroups(): void {
    const mkBg = (a: GPUBuffer, b: GPUBuffer, extra: GPUBuffer) =>
      this.device.createBindGroup({
        layout: this.bindLayout,
        entries: [
          { binding: 0, resource: { buffer: a } },
          { binding: 1, resource: { buffer: b } },
          { binding: 2, resource: { buffer: this.flags } },
          { binding: 3, resource: { buffer: this.macro } },
          { binding: 4, resource: { buffer: this.forceAcc } },
          { binding: 5, resource: { buffer: this.paramsBuf } },
          { binding: 6, resource: { buffer: extra } },
        ],
      });
    // extra 在常规推进时无实际消费者，但布局要求存在，绑自身即可
    this.bgAB = mkBg(this.fA, this.fB, this.extra);
    this.bgReconcile = mkBg(this.fA, this.fB, this.extra);
    this.bgPressure = mkBg(this.fA, this.fB, this.extra);
  }

  /** 上传占用栅格（首次）并把全场初始化为平衡态 */
  uploadInitialVoxels(flags: Uint8Array): void {
    const u32 = Uint32Array.from(flags, (v) => v);
    this.device.queue.writeBuffer(this.flags, 0, u32);
    this.device.queue.writeBuffer(this.extra, 0, new Uint32Array(this.count)); // 旧 flags 先清零
    this.lastCpuFlags = flags.slice();
    this.resetField();
  }

  private resetField(): void {
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.resetPipe);
    pass.setBindGroup(0, this.bgAB);
    pass.dispatchWorkgroups(
      ...this.grid(),
    );
    pass.end();
    this.device.queue.submit([enc.finish()]);
    this.steps = 0;
    this.forceAccumSteps = 0;
    this.params.steps = 0;
    this.writeParams();
  }

  /**
   * 热启动重新体素化：
   * 1. 旧 flags 拷到 extra；2. 上传新 flags；
   * 3. 只重置身份变化的格子，其余分布函数原样保留，流场接着算。
   */
  async revoxelize(newFlags: Uint8Array): Promise<{ changed: number }> {
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.flags, 0, this.extra, 0, this.count * 4);
    this.device.queue.submit([enc.finish()]);

    const u32 = Uint32Array.from(newFlags, (v) => v);
    this.device.queue.writeBuffer(this.flags, 0, u32);

    let changed = 0;
    // CPU 侧统计变化量用于 UI 展示（数据本就在主线程）
    // 注意：newFlags 与旧栅格的对比由调用方保证传入的是新栅格
    const enc2 = this.device.createCommandEncoder();
    const pass = enc2.beginComputePass();
    pass.setPipeline(this.reconcilePipe);
    pass.setBindGroup(0, this.bgReconcile);
    pass.dispatchWorkgroups(
      ...this.grid(),
    );
    pass.end();
    this.device.queue.submit([enc2.finish()]);
    changed = u32.reduce((acc, v, i) => acc + (v !== (this.lastCpuFlags?.[i] ?? FLUID) ? 1 : 0), 0);
    this.lastCpuFlags = newFlags.slice();
    return { changed };
  }

  private lastCpuFlags?: Uint8Array;

  /** 更新运行参数（速度/松弛/Smagorinsky），不重置流场 */
  setRuntime(p: Pick<SimParams, 'u0' | 'tau' | 'smagorinsky'>): void {
    this.params.u0 = p.u0;
    this.params.tau = p.tau;
    this.params.smagorinsky = p.smagorinsky;
    this.writeParams();
  }

  /** 推进 n 步（力累加器由 clearForceAcc 单独清零，便于多步累积一次采样） */
  step(n: number): void {
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    const [wgX, wgY, wgZ] = this.grid();

    // AA pattern：collide(fIn→fOut) + stream(fOut→fIn) 是一个自洽步骤，
    // 结束后状态回到 fIn，因此每步都用同一个绑定组，不能翻转。
    for (let s = 0; s < n; s++) {
      pass.setBindGroup(0, this.bgAB);

      pass.setPipeline(this.collidePipe);
      pass.dispatchWorkgroups(wgX, wgY, wgZ);

      pass.setPipeline(this.streamPipe);
      pass.dispatchWorkgroups(wgX, wgY, wgZ);

      this.steps++;
    }
    this.forceAccumSteps += n;
    pass.end();
    this.device.queue.submit([enc.finish()]);
    this.params.steps = this.steps;
    this.writeParams();
  }

  clearForceAcc(): void {
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.clearPipe);
    pass.setBindGroup(0, this.bgAB);
    pass.dispatchWorkgroups(1);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  /** 刷新表面压力缓冲（每渲染帧调用一次足够） */
  updateSurfacePressure(): void {
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pressurePipe);
    pass.setBindGroup(0, this.bgPressure);
    pass.dispatchWorkgroups(
      ...this.grid(),
    );
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  /** 异步读回力累加器，归一化为单步平均后清零 */
  async readForce(): Promise<ForceSample> {
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.forceAcc, 0, this.forceReadback, 0, 24);
    this.device.queue.submit([enc.finish()]);
    await this.forceReadback.mapAsync(GPUMapMode.READ);
    const raw = new Int32Array(this.forceReadback.getMappedRange().slice(0));
    this.forceReadback.unmap();
    const FP = 4096;
    const scale = 1 / Math.max(this.forceAccumSteps, 1);
    const pressure = (raw[0] / FP) * scale;
    const viscous = (raw[3] / FP) * scale;
    this.forceAccumSteps = 0;
    this.clearForceAcc();
    return { pressure, viscous, total: pressure + viscous };
  }

  /** 重置整个流场（保留当前车身） */
  resetFlow(): void {
    this.resetField();
  }

  /**
   * 读回表面相对流速（extra 缓冲），返回给定表面格的 p5/p95 分位范围。
   * 用分位数而非 min/max，避免少数极端格把色带压扁。节流调用（约每秒一次）。
   */
  async readSurfaceSpeedRange(surfaceCells: Uint32Array): Promise<{ min: number; max: number }> {
    const N = this.count;
    if (!this.speedReadback || this.speedReadback.size < N * 4) {
      this.speedReadback?.destroy();
      this.speedReadback = this.device.createBuffer({
        size: N * 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
    }
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.extra, 0, this.speedReadback, 0, N * 4);
    this.device.queue.submit([enc.finish()]);
    await this.speedReadback.mapAsync(GPUMapMode.READ);
    const all = new Float32Array(this.speedReadback.getMappedRange().slice(0));
    this.speedReadback.unmap();
    const vals: number[] = [];
    for (const cell of surfaceCells) {
      const v = all[cell];
      if (v > 1e-6) vals.push(v);
    }
    if (vals.length < 8) return { min: 0, max: 1 };
    vals.sort((a, b) => a - b);
    const pick = (q: number) => vals[Math.min(vals.length - 1, Math.floor(q * vals.length))];
    return { min: pick(0.05), max: pick(0.95) };
  }

  getFlagsBuffer(): GPUBuffer { return this.flags; }
  getMacroBuffer(): GPUBuffer { return this.macro; }
  getPressureBuffer(): GPUBuffer { return this.extra; }
  getParamsBuffer(): GPUBuffer { return this.paramsBuf; }
  getBindLayout(): GPUBindGroupLayout { return this.bindLayout; }

  destroy(): void {
    for (const b of [this.fA, this.fB, this.flags, this.macro, this.extra,
      this.forceAcc, this.forceReadback, this.paramsBuf]) {
      b.destroy();
    }
    this.speedReadback?.destroy();
  }
}
