// 场景渲染:地面网格、中截面速度云图、车身(顶点色 = 压力系数 Cp)
import type { MeshData } from './carmesh';

const COLORMAP = `
fn colormap(t0: f32) -> vec3f {
  let t = clamp(t0, 0.0, 1.0);
  var c = mix(vec3f(0.10, 0.20, 0.90), vec3f(0.10, 0.80, 0.90), clamp(t * 4.0, 0.0, 1.0));
  c = mix(c, vec3f(0.20, 0.85, 0.30), clamp(t * 4.0 - 1.0, 0.0, 1.0));
  c = mix(c, vec3f(0.98, 0.85, 0.15), clamp(t * 4.0 - 2.0, 0.0, 1.0));
  c = mix(c, vec3f(0.90, 0.15, 0.10), clamp(t * 4.0 - 3.0, 0.0, 1.0));
  return c;
}
`;

export class SceneRenderer {
  private device: GPUDevice;
  private camBuf: GPUBuffer;

  // 车身
  private carVertBuf: GPUBuffer | null = null;
  private carIdxBuf: GPUBuffer | null = null;
  private carColorBuf: GPUBuffer | null = null;
  private carIdxCount = 0;
  private carVertCount = 0;
  private carPipe: GPURenderPipeline;
  private carBG: GPUBindGroup | null = null;
  private colorPipe: GPUComputePipeline;
  private colorBG: GPUBindGroup | null = null;
  private colorParamsBuf: GPUBuffer;

  // 地面
  private groundPipe: GPURenderPipeline;
  private groundBG: GPUBindGroup;

  // 切片
  private slicePipe: GPURenderPipeline;
  private sliceBG: GPUBindGroup;
  private sliceParamsBuf: GPUBuffer;

  constructor(
    device: GPUDevice,
    private nx: number,
    private ny: number,
    private nz: number,
    camBuf: GPUBuffer,
    macroBuf: GPUBuffer,
    flagsBuf: GPUBuffer,
    format: GPUTextureFormat,
  ) {
    this.device = device;
    this.camBuf = camBuf;
    this.colorParamsBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.sliceParamsBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    // ---- 车身渲染管线(顶点色来自 compute 写入的 Cp 颜色)
    const carCode = `
struct Cam { vp: mat4x4f, eye: vec3f };
@group(0) @binding(0) var<uniform> cam: Cam;
@group(0) @binding(1) var<storage, read> colors: array<vec4f>;
struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec3f,
  @location(1) nrm: vec3f,
};
@vertex
fn vs(@location(0) pos: vec3f, @location(1) nrm: vec3f, @builtin(vertex_index) vi: u32) -> VOut {
  var o: VOut;
  o.pos = cam.vp * vec4f(pos, 1.0);
  o.color = colors[vi].rgb;
  o.nrm = nrm;
  return o;
}
@fragment
fn fs(in: VOut) -> @location(0) vec4f {
  let l = normalize(vec3f(0.45, 0.8, 0.35));
  let n = normalize(in.nrm);
  let diff = 0.35 + 0.65 * abs(dot(n, l));
  return vec4f(in.color * diff, 1.0);
}`;
    this.carPipe = device.createRenderPipeline({
      layout: 'auto',
      vertex: {
        module: device.createShaderModule({ code: carCode }),
        entryPoint: 'vs',
        buffers: [
          {
            arrayStride: 24,
            attributes: [
              { shaderLocation: 0, offset: 0, format: 'float32x3' },
              { shaderLocation: 1, offset: 12, format: 'float32x3' },
            ],
          },
        ],
      },
      fragment: {
        module: device.createShaderModule({ code: carCode }),
        entryPoint: 'fs',
        targets: [{ format }],
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    });

    // ---- 顶点着色 kernel:采样密度场 → Cp → 颜色
    const colorCode = `
const NX: u32 = ${nx}u;
const NY: u32 = ${ny}u;
const NZ: u32 = ${nz}u;
@group(0) @binding(0) var<storage, read> macro: array<vec4f>;
@group(0) @binding(1) var<storage, read> flags: array<u32>;
@group(0) @binding(2) var<storage, read> verts: array<f32>;
@group(0) @binding(3) var<storage, read_write> colors: array<vec4f>;
@group(0) @binding(4) var<uniform> params: vec4f; // x: u0
${COLORMAP}
fn cid(x: u32, y: u32, z: u32) -> u32 { return x + y * NX + z * NX * NY; }
fn rhoAt(p: vec3f) -> f32 {
  let pc = clamp(p, vec3f(1.0), vec3f(f32(NX) - 2.001, f32(NY) - 2.001, f32(NZ) - 2.001));
  let i0 = vec3u(floor(pc));
  let fr = pc - floor(pc);
  var acc = 0.0;
  for (var dz = 0u; dz < 2u; dz++) {
    for (var dy = 0u; dy < 2u; dy++) {
      for (var dx = 0u; dx < 2u; dx++) {
        let w = select(1.0 - fr.x, fr.x, dx == 1u) * select(1.0 - fr.y, fr.y, dy == 1u) * select(1.0 - fr.z, fr.z, dz == 1u);
        acc += w * macro[cid(i0.x + dx, i0.y + dy, i0.z + dz)].w;
      }
    }
  }
  return acc;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let vi = gid.x;
  if (vi >= arrayLength(&colors)) { return; }
  let p = vec3f(verts[vi * 6u], verts[vi * 6u + 1u], verts[vi * 6u + 2u]);
  let n = vec3f(verts[vi * 6u + 3u], verts[vi * 6u + 4u], verts[vi * 6u + 5u]);
  // 沿法向探到流体里取样,避免取到固体内部
  var sp = p + n * 1.3;
  var c = cid(u32(sp.x), u32(sp.y), u32(sp.z));
  if (flags[c] == 1u) { sp = p + n * 2.6; }
  let u0 = params.x;
  let cp = (rhoAt(sp) - 1.0) / (1.5 * u0 * u0);
  colors[vi] = vec4f(colormap(cp * 0.45 + 0.5), 1.0);
}`;
    this.colorPipe = device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: colorCode }), entryPoint: 'main' },
    });
    this.macroBuf = macroBuf;
    this.flagsBuf = flagsBuf;

    // ---- 地面
    const groundCode = `
struct Cam { vp: mat4x4f, eye: vec3f };
@group(0) @binding(0) var<uniform> cam: Cam;
struct VOut { @builtin(position) pos: vec4f, @location(0) wp: vec3f };
@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  var corners = array<vec2f, 6>(
    vec2f(0.0, 0.0), vec2f(${nx}.0, 0.0), vec2f(${nx}.0, ${nz}.0),
    vec2f(0.0, 0.0), vec2f(${nx}.0, ${nz}.0), vec2f(0.0, ${nz}.0));
  let c = corners[vi];
  var o: VOut;
  o.wp = vec3f(c.x, 0.5, c.y);
  o.pos = cam.vp * vec4f(o.wp, 1.0);
  return o;
}
@fragment
fn fs(in: VOut) -> @location(0) vec4f {
  let g = 8.0;
  let lx = abs(fract(in.wp.x / g) - 0.5);
  let lz = abs(fract(in.wp.z / g) - 0.5);
  let line = step(0.47, max(lx, lz));
  let base = vec3f(0.10, 0.115, 0.14);
  return vec4f(base + line * vec3f(0.05, 0.06, 0.07), 1.0);
}`;
    this.groundPipe = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: device.createShaderModule({ code: groundCode }), entryPoint: 'vs' },
      fragment: {
        module: device.createShaderModule({ code: groundCode }),
        entryPoint: 'fs',
        targets: [{ format }],
      },
      primitive: { topology: 'triangle-list' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    });
    this.groundBG = device.createBindGroup({
      layout: this.groundPipe.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: camBuf } }],
    });

    // ---- 中截面速度云图
    const sliceCode = `
const NX: u32 = ${nx}u;
const NY: u32 = ${ny}u;
const NZ: u32 = ${nz}u;
struct Cam { vp: mat4x4f, eye: vec3f };
@group(0) @binding(0) var<uniform> cam: Cam;
@group(0) @binding(1) var<storage, read> macro: array<vec4f>;
@group(0) @binding(2) var<storage, read> flags: array<u32>;
@group(0) @binding(3) var<uniform> sp: vec4f; // x: 切片 z 坐标, y: u0
${COLORMAP}
fn cid(x: u32, y: u32, z: u32) -> u32 { return x + y * NX + z * NX * NY; }
struct VOut { @builtin(position) pos: vec4f, @location(0) wp: vec3f };
@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  var corners = array<vec2f, 6>(
    vec2f(0.0, 0.0), vec2f(f32(NX), 0.0), vec2f(f32(NX), f32(NY)),
    vec2f(0.0, 0.0), vec2f(f32(NX), f32(NY)), vec2f(0.0, f32(NY)));
  let c = corners[vi];
  var o: VOut;
  o.wp = vec3f(c.x, c.y, sp.x);
  o.pos = cam.vp * vec4f(o.wp, 1.0);
  return o;
}
@fragment
fn fs(in: VOut) -> @location(0) vec4f {
  let x = u32(clamp(in.wp.x, 0.0, f32(NX) - 1.0));
  let y = u32(clamp(in.wp.y, 0.0, f32(NY) - 1.0));
  let z = u32(clamp(in.wp.z, 0.0, f32(NZ) - 1.0));
  let id = cid(x, y, z);
  if (flags[id] == 1u) { return vec4f(0.05, 0.05, 0.06, 0.95); }
  let u = macro[id].xyz;
  let speed = length(u) / sp.y;
  return vec4f(colormap(speed * 0.7), 0.82);
}`;
    this.slicePipe = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: device.createShaderModule({ code: sliceCode }), entryPoint: 'vs' },
      fragment: {
        module: device.createShaderModule({ code: sliceCode }),
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
      primitive: { topology: 'triangle-list' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less' },
    });
    this.sliceBG = device.createBindGroup({
      layout: this.slicePipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: camBuf } },
        { binding: 1, resource: { buffer: macroBuf } },
        { binding: 2, resource: { buffer: flagsBuf } },
        { binding: 3, resource: { buffer: this.sliceParamsBuf } },
      ],
    });
  }

  private macroBuf: GPUBuffer;
  private flagsBuf: GPUBuffer;

  // 换车:重建顶点/索引/颜色缓冲
  setMesh(mesh: MeshData) {
    const nVerts = mesh.positions.length / 3;
    const interleaved = new Float32Array(nVerts * 6);
    for (let i = 0; i < nVerts; i++) {
      interleaved[i * 6] = mesh.positions[i * 3];
      interleaved[i * 6 + 1] = mesh.positions[i * 3 + 1];
      interleaved[i * 6 + 2] = mesh.positions[i * 3 + 2];
      interleaved[i * 6 + 3] = mesh.normals[i * 3];
      interleaved[i * 6 + 4] = mesh.normals[i * 3 + 1];
      interleaved[i * 6 + 5] = mesh.normals[i * 3 + 2];
    }
    this.carVertBuf?.destroy();
    this.carIdxBuf?.destroy();
    this.carColorBuf?.destroy();
    this.carVertBuf = this.device.createBuffer({
      size: interleaved.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.carVertBuf, 0, interleaved);
    this.carIdxBuf = this.device.createBuffer({
      size: mesh.indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.carIdxBuf, 0, mesh.indices);
    this.carColorBuf = this.device.createBuffer({
      size: nVerts * 16,
      usage: GPUBufferUsage.STORAGE,
    });
    this.carIdxCount = mesh.indices.length;
    this.carVertCount = nVerts;

    this.carBG = this.device.createBindGroup({
      layout: this.carPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.camBuf } },
        { binding: 1, resource: { buffer: this.carColorBuf } },
      ],
    });
    this.colorBG = this.device.createBindGroup({
      layout: this.colorPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.macroBuf } },
        { binding: 1, resource: { buffer: this.flagsBuf } },
        { binding: 2, resource: { buffer: this.carVertBuf } },
        { binding: 3, resource: { buffer: this.carColorBuf } },
        { binding: 4, resource: { buffer: this.colorParamsBuf } },
      ],
    });
  }

  // 每帧:重算顶点颜色(Cp)
  encodeCarColors(encoder: GPUCommandEncoder, u0: number) {
    if (!this.colorBG) return;
    this.device.queue.writeBuffer(this.colorParamsBuf, 0, new Float32Array([u0, 0, 0, 0]));
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.colorPipe);
    pass.setBindGroup(0, this.colorBG);
    pass.dispatchWorkgroups(Math.ceil(this.carVertCount / 64));
    pass.end();
  }

  setSlice(zFrac: number, u0: number) {
    this.device.queue.writeBuffer(
      this.sliceParamsBuf,
      0,
      new Float32Array([zFrac * (this.nz - 1), u0, 0, 0]),
    );
  }

  encodeGround(pass: GPURenderPassEncoder) {
    pass.setPipeline(this.groundPipe);
    pass.setBindGroup(0, this.groundBG);
    pass.draw(6);
  }

  encodeSlice(pass: GPURenderPassEncoder) {
    pass.setPipeline(this.slicePipe);
    pass.setBindGroup(0, this.sliceBG);
    pass.draw(6);
  }

  encodeCar(pass: GPURenderPassEncoder) {
    if (!this.carBG || !this.carVertBuf || !this.carIdxBuf) return;
    pass.setPipeline(this.carPipe);
    pass.setBindGroup(0, this.carBG);
    pass.setVertexBuffer(0, this.carVertBuf);
    pass.setIndexBuffer(this.carIdxBuf, 'uint32');
    pass.drawIndexed(this.carIdxCount);
  }
}
