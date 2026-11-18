/**
 * 烟线粒子 + 车身表面 + 地面的渲染器。
 * 烟线：每粒子 8 节历史，线列表 + 加性/透明混合，呈现半透明细丝。
 * 车身：表面固体格子实例化单位立方体，压力系数 Cp 在着色器里映射成色带。
 */
import smokeWgsl from './smoke.wgsl?raw';
import { SOLID } from '../lbm/constants';
import { OrbitCamera, mat4WorldFromLattice, type Mat4 } from '../render/math';

export interface SmokeSettings {
  nozzleCols: number;
  nozzleRows: number;
  trailAlpha: number;
  speedScale: number;
  showSmoke: boolean;
  showPressure: boolean;
}

const TRAIL = 8;
// 每粒子 (TRAIL-1) 段，每段 6 顶点（ribbon 四边形的两个三角形）
const SEG_VERTS = 6 * (TRAIL - 1);

interface SmokeParams {
  nx: number; ny: number; nz: number; particleCount: number;
  u0: number; speedScale: number; life: number;
  nozzleX: number; nozzleBaseY: number; nozzleBaseZ: number;
  nozzleStepY: number; nozzleStepZ: number;
  nozzleCols: number; nozzleRows: number; steps: number;
}

function smokeParamsBuffer(p: SmokeParams): ArrayBuffer {
  // 与 WGSL ParamsSmoke 对齐：u32×4 然后 f32/u32 混排，每个字段 4 字节
  const buf = new ArrayBuffer(64);
  const u32 = new Uint32Array(buf);
  const f32 = new Float32Array(buf);
  u32[0] = p.nx; u32[1] = p.ny; u32[2] = p.nz; u32[3] = p.particleCount;
  f32[4] = p.u0; f32[5] = p.speedScale; f32[6] = p.life;
  f32[7] = p.nozzleX; f32[8] = p.nozzleBaseY; f32[9] = p.nozzleBaseZ;
  f32[10] = p.nozzleStepY; f32[11] = p.nozzleStepZ;
  u32[12] = p.nozzleCols; u32[13] = p.nozzleRows; u32[14] = p.steps; u32[15] = 0;
  return buf;
}

export class SmokeRenderer {
  private device: GPUDevice;
  private nx: number;
  private ny: number;
  private nz: number;

  private particleBuf: GPUBuffer;
  private particleCount: number;
  private smokeParams: GPUBuffer;
  private camUniform: GPUBuffer;
  private surfaceBuf: GPUBuffer;
  private pressureBuf: GPUBuffer;
  private cubeVerts: GPUBuffer;
  private surfaceCount = 0;
  private surfaceCellsCpu = new Uint32Array(0);
  private speedMin = 0;
  private speedMax = 1;

  private advectPipe: GPUComputePipeline;
  private smokePipe: GPURenderPipeline;
  private surfacePipe: GPURenderPipeline;
  private groundPipe: GPURenderPipeline;
  private computeLayout: GPUBindGroupLayout;
  private renderLayout: GPUBindGroupLayout;
  private smokeRenderLayout: GPUBindGroupLayout;
  private bgCompute!: GPUBindGroup;
  private bgRender!: GPUBindGroup;
  private bgSmokeRender!: GPUBindGroup;

  readonly camera = new OrbitCamera();
  private worldFromLattice: Mat4;
  private worldScale: number;

  constructor(
    device: GPUDevice,
    format: GPUTextureFormat,
    nx: number, ny: number, nz: number,
    macroBuf: GPUBuffer,
    flagsBuf: GPUBuffer,
    pressureBuf: GPUBuffer,
  ) {
    this.device = device;
    this.nx = nx; this.ny = ny; this.nz = nz;

    // 粒子数上限：默认 2048，每粒子 9×vec4 = 144 字节
    this.particleCount = 4096;
    this.particleBuf = device.createBuffer({
      size: this.particleCount * 144,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.smokeParams = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    // Camera：2×mat4(128B) + 4×f32(16B) + vec4 右向量(16B) = 160B
    this.camUniform = device.createBuffer({ size: 160, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.surfaceBuf = device.createBuffer({
      size: Math.max(64, Math.ceil(nx * ny * nz / 8) * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.pressureBuf = pressureBuf;

    // 单位立方体顶点（pos + normal，36 顶点 × 24 字节）
    this.cubeVerts = device.createBuffer({
      size: 36 * 6 * 4,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.cubeVerts, 0, this.buildCube() as BufferSource);

    const mod = device.createShaderModule({ code: smokeWgsl });

    this.computeLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });
    // 表面管线 BGL：相机(0) + 烟线参数(3) + 粒子只读(4) + 表面格(5) + 压力(6)
    this.renderLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
        { binding: 4, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 5, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 6, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      ],
    });
    // 烟线/地面管线 BGL：相机(0) + 烟线参数(3，地面顶点用到 P.nx/nz) + 粒子只读(4)
    this.smokeRenderLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
        { binding: 4, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      ],
    });

    this.advectPipe = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.computeLayout] }),
      compute: { module: mod, entryPoint: 'advect' },
    });

    this.smokePipe = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.smokeRenderLayout] }),
      vertex: { module: mod, entryPoint: 'vsSmoke' },
      fragment: {
        module: mod, entryPoint: 'fsSmoke',
        targets: [{
          format,
          blend: {
            // 标准 alpha 混合（面状 ribbon）
            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          },
        }],
      },
      primitive: { topology: 'triangle-list' },
      depthStencil: {
        format: 'depth24plus',
        depthWriteEnabled: false,
        depthCompare: 'less',
      },
    });

    this.surfacePipe = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.renderLayout] }),
      vertex: {
        module: mod,
        entryPoint: 'vsSurface',
        buffers: [{
          arrayStride: 24,
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x3' },
            { shaderLocation: 1, offset: 12, format: 'float32x3' },
          ],
        }],
      },
      fragment: { module: mod, entryPoint: 'fsSurface', targets: [{ format }] },
      primitive: { topology: 'triangle-list' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    });

    this.groundPipe = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.smokeRenderLayout] }),
      vertex: { module: mod, entryPoint: 'vsGround' },
      fragment: { module: mod, entryPoint: 'fsGround', targets: [{ format }] },
      primitive: { topology: 'triangle-list' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    });

    this.worldScale = 14 / Math.max(nx, nz);
    this.worldFromLattice = mat4WorldFromLattice(nx, ny, nz, this.worldScale);

    this.bgCompute = device.createBindGroup({
      layout: this.computeLayout,
      entries: [
        { binding: 0, resource: { buffer: macroBuf } },
        { binding: 1, resource: { buffer: flagsBuf } },
        { binding: 2, resource: { buffer: this.particleBuf } },
        { binding: 3, resource: { buffer: this.smokeParams } },
      ],
    });
    // 表面管线：相机 + 参数 + 粒子(只读) + 表面格 + 压力
    this.bgRender = device.createBindGroup({
      layout: this.renderLayout,
      entries: [
        { binding: 0, resource: { buffer: this.camUniform } },
        { binding: 3, resource: { buffer: this.smokeParams } },
        { binding: 4, resource: { buffer: this.particleBuf } },
        { binding: 5, resource: { buffer: this.surfaceBuf } },
        { binding: 6, resource: { buffer: pressureBuf } },
      ],
    });
    // 烟线/地面管线：相机 + 参数 + 粒子(只读)
    this.bgSmokeRender = device.createBindGroup({
      layout: this.smokeRenderLayout,
      entries: [
        { binding: 0, resource: { buffer: this.camUniform } },
        { binding: 3, resource: { buffer: this.smokeParams } },
        { binding: 4, resource: { buffer: this.particleBuf } },
      ],
    });

    this.seedParticles();
  }

  /** 从 CPU 占用栅格提取表面固体格（至少有一个面邻居是流体） */
  setSurfaceCells(flags: Uint8Array): void {
    const { nx, ny, nz } = this;
    const cells: number[] = [];
    const id = (i: number, j: number, k: number) => i * (ny * nz) + j * nz + k;
    for (let i = 0; i < nx; i++) {
      for (let j = 0; j < ny; j++) {
        for (let k = 0; k < nz; k++) {
          if (flags[id(i, j, k)] !== SOLID) continue;
          let exposed = false;
          if (i === 0 || i === nx - 1 || j === 0 || j === ny - 1 || k === 0 || k === nz - 1) {
            exposed = true;
          } else {
            exposed =
              flags[id(i + 1, j, k)] === 0 || flags[id(i - 1, j, k)] === 0 ||
              flags[id(i, j + 1, k)] === 0 || flags[id(i, j - 1, k)] === 0 ||
              flags[id(i, j, k + 1)] === 0 || flags[id(i, j, k - 1)] === 0;
          }
          if (exposed) cells.push(id(i, j, k));
        }
      }
    }
    this.surfaceCount = cells.length;
    const u32 = Uint32Array.from(cells);
    if (u32.byteLength > this.surfaceBuf.size) {
      this.surfaceBuf.destroy();
      this.surfaceBuf = this.device.createBuffer({
        size: u32.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.rebuildRenderBg();
    }
    this.device.queue.writeBuffer(this.surfaceBuf, 0, u32);
    this.surfaceCellsCpu = u32;
  }

  getSurfaceCells(): Uint32Array { return this.surfaceCellsCpu; }

  /** 设置表面相对流速色带范围（CPU 按分位数统计后传入） */
  setSpeedRange(min: number, max: number): void {
    this.speedMin = min;
    this.speedMax = max;
  }

  getParticleBuffer(): GPUBuffer { return this.particleBuf; }

  private rebuildRenderBg(): void {
    this.bgRender = this.device.createBindGroup({
      layout: this.renderLayout,
      entries: [
        { binding: 0, resource: { buffer: this.camUniform } },
        { binding: 3, resource: { buffer: this.smokeParams } },
        { binding: 4, resource: { buffer: this.particleBuf } },
        { binding: 5, resource: { buffer: this.surfaceBuf } },
        { binding: 6, resource: { buffer: this.pressureBuf } },
      ],
    });
  }

  private seedParticles(): void {
    // 初始位置由 respawn 逻辑决定，这里清零即可，首帧 advect 会把负年龄粒子正常推进
    this.device.queue.writeBuffer(this.particleBuf, 0, new Uint8Array(this.particleBuf.size));
  }

  private buildCube(): Float32Array {
    // 6 面 × 4 顶点（pos, normal）
    const h = 0.5;
    const faces: Array<{ n: [number, number, number]; q: Array<[number, number, number]> }> = [
      { n: [1, 0, 0], q: [[h, -h, -h], [h, -h, h], [h, h, h], [h, h, -h]] },
      { n: [-1, 0, 0], q: [[-h, -h, h], [-h, -h, -h], [-h, h, -h], [-h, h, h]] },
      { n: [0, 1, 0], q: [[-h, h, -h], [h, h, -h], [h, h, h], [-h, h, h]] },
      { n: [0, -1, 0], q: [[-h, -h, h], [h, -h, h], [h, -h, -h], [-h, -h, -h]] },
      { n: [0, 0, 1], q: [[-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h]] },
      { n: [0, 0, -1], q: [[h, -h, -h], [-h, -h, -h], [-h, h, -h], [h, h, -h]] },
    ];
    const out: number[] = [];
    const push = (p: [number, number, number], n: [number, number, number]) =>
      out.push(p[0], p[1], p[2], n[0], n[1], n[2]);
    for (const f of faces) {
      push(f.q[0], f.n); push(f.q[1], f.n); push(f.q[2], f.n);
      push(f.q[0], f.n); push(f.q[2], f.n); push(f.q[3], f.n);
    }
    return new Float32Array(out);
  }

  /** 更新喷嘴参数与相机 uniform，执行一次粒子平流 */
  update(
    settings: SmokeSettings,
    u0: number,
    simSteps: number,
    depthView: GPUTextureView,
    colorView: GPUTextureView,
    aspect: number,
  ): void {
    const cols = settings.nozzleCols;
    const rows = settings.nozzleRows;
    const active = Math.min(this.particleCount, cols * rows * 14);
    const sp: SmokeParams = {
      nx: this.nx, ny: this.ny, nz: this.nz,
      particleCount: active,
      u0,
      speedScale: settings.speedScale,
      life: this.nx * 1.15 / Math.max(u0, 1e-3),
      nozzleX: this.nx * 0.08,
      nozzleBaseY: this.ny * 0.18,
      nozzleBaseZ: this.nz * 0.22,
      nozzleStepY: (this.ny * 0.64) / Math.max(cols - 1, 1),
      nozzleStepZ: (this.nz * 0.56) / Math.max(rows - 1, 1),
      nozzleCols: cols,
      nozzleRows: rows,
      steps: simSteps,
    };
    this.device.queue.writeBuffer(this.smokeParams, 0, smokeParamsBuffer(sp));

    const vp = this.camera.viewProj(aspect);
    const cam = new ArrayBuffer(160);
    new Float32Array(cam, 0, 16).set(vp);                 // offset 0
    new Float32Array(cam, 64, 16).set(this.worldFromLattice); // offset 64
    const f32 = new Float32Array(cam);
    f32[32] = u0;        // offset 128
    f32[33] = settings.trailAlpha;
    f32[34] = this.speedMin;
    f32[35] = this.speedMax;
    const right = this.camera.right();                    // offset 144: vec4
    f32[36] = right[0]; f32[37] = right[1]; f32[38] = right[2]; f32[39] = 1;
    this.device.queue.writeBuffer(this.camUniform, 0, cam);

    const enc = this.device.createCommandEncoder();

    if (settings.showSmoke) {
      const cp = enc.beginComputePass();
      cp.setPipeline(this.advectPipe);
      cp.setBindGroup(0, this.bgCompute);
      cp.dispatchWorkgroups(Math.ceil(active / 128));
      cp.end();
    }

    const rp = enc.beginRenderPass({
      colorAttachments: [{
        view: colorView,
        clearValue: { r: 0.05, g: 0.06, b: 0.09, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
      depthStencilAttachment: {
        view: depthView,
        depthClearValue: 1,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    });

    rp.setPipeline(this.groundPipe);
    rp.setBindGroup(0, this.bgSmokeRender);
    rp.draw(6);

    if (settings.showPressure && this.surfaceCount > 0) {
      rp.setPipeline(this.surfacePipe);
      rp.setBindGroup(0, this.bgRender);
      rp.setVertexBuffer(0, this.cubeVerts);
      rp.draw(36, this.surfaceCount);
    }

    if (settings.showSmoke) {
      rp.setPipeline(this.smokePipe);
      rp.setBindGroup(0, this.bgSmokeRender);
      rp.draw(active * SEG_VERTS);
    }

    rp.end();
    this.device.queue.submit([enc.finish()]);
  }
}
