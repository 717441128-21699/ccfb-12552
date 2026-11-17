// WebGPU 渲染器：牙釉质用带次表面散射近似的 PBR 着色，牙龈用软组织着色。
// 每颗牙一个独立 GPUBuffer（按当前位姿每帧重建），牙龈为动态变形网格。
import {
  type Mat4, type Quat, type Vec3,
  mat4Identity, mat4LookAt, mat4Multiply, mat4Perspective, qRotate,
} from './math';
import type { ToothPart } from './mesh';

export interface RenderObject {
  part: ToothPart;
  label: number;
  positions: Float32Array; // 当前世界坐标（每帧由外部更新）
  normals: Float32Array;
  indices: Uint32Array;
  isGum?: boolean;
}

const SHADER = /* wgsl */ `
struct Camera {
  viewProj: mat4x4<f32>,
  eye: vec3<f32>,
  _pad: f32,
};
struct Light {
  dir: vec3<f32>,
  _pad: f32,
};
struct Material {
  baseColor: vec3<f32>,
  roughness: f32,
  sssAmount: f32,      // 次表面散射强度（牙釉质高、牙龈低）
  sssThickness: f32,  // 透射厚度系数
  _pad2: vec2<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> light: Light;
@group(0) @binding(2) var<uniform> material: Material;

struct VSOut {
  @builtin(position) clip: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
  @location(1) normal: vec3<f32>,
};

@vertex
fn vs(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>) -> VSOut {
  var out: VSOut;
  // 位置已在 CPU 变换到世界系，模型矩阵为单位阵
  out.worldPos = position;
  out.normal = normal;
  out.clip = camera.viewProj * vec4<f32>(position, 1.0);
  return out;
}

// 简单 GGX 漫反/高光的近似，重点在 SSS 透射项
@fragment
fn fs(in: VSOut) -> @location(0) vec4<f32> {
  let N = normalize(in.normal);
  let L = normalize(light.dir);
  let V = normalize(camera.eye - in.worldPos);
  let H = normalize(L + V);

  let ndl = max(dot(N, L), 0.0);
  let ndv = max(dot(N, V), 0.0001);

  // 漫反射（Lambert 基底）
  let diffuse = material.baseColor * (0.25 + 0.75 * ndl);

  // 高光：粗糙度越低越集中
  let shininess = mix(120.0, 12.0, material.roughness);
  let spec = pow(max(dot(N, H), 0.0), shininess) * (1.0 - material.roughness * 0.7);
  // 釉质菲涅耳边缘高光
  let fresnel = pow(1.0 - ndv, 5.0);
  spec = spec + fresnel * 0.35 * material.sssAmount;

  // 次表面散射近似：背光面的暖色透射 +  wrap 漫射（光绕过边缘）
  // 1) wrap lighting：把 [-1,1] 的点积重映射到 [0,1]，模拟光在组织内弥散
  let wrap = max((dot(N, L) + material.sssThickness) / (1.0 + material.sssThickness), 0.0);
  let sssWrap = material.baseColor * wrap * material.sssAmount * 0.6;
  // 2) 背向透射：法线朝背光时，光穿过薄釉质从内侧透出，呈暖黄
  let backlit = max(dot(-N, L), 0.0);
  let transmission = vec3<f32>(1.0, 0.85, 0.6) * backlit * material.sssAmount * 0.5;
  // 3) 边缘散射（rim），让牙冠边缘有柔和透光感
  let rim = pow(1.0 - ndv, 3.0) * material.sssAmount * 0.35;
  let sss = sssWrap + transmission + vec3<f32>(rim) * material.baseColor;

  var color = diffuse + vec3<f32>(spec) + sss;
  // 简单色调映射 + gamma
  color = color / (color + vec3<f32>(0.8));
  color = pow(color, vec3<f32>(1.0 / 2.2));
  return vec4<f32>(color, 1.0);
}
`;

export class WebGPURenderer {
  device!: GPUDevice;
  context!: GPUCanvasContext;
  format!: GPUTextureFormat;
  pipeline!: GPURenderPipeline;
  bindGroupLayout!: GPUBindGroupLayout;
  cameraBuffer!: GPUBuffer;
  lightBuffer!: GPUBuffer;
  materialBuffer!: GPUBuffer;
  bindGroup!: GPUBindGroup;
  depthTexture!: GPUTexture;
  canvas: HTMLCanvasElement;
  sssEnabled = true;

  // 轨道相机
  orbit = { yaw: 0.6, pitch: 0.35, distance: 90, target: [0, 4, 0] as Vec3 };

  private objects = new Map<number, RenderObject>();
  private gumObject: RenderObject | null = null;
  // 按对象标签缓存 GPU 缓冲，避免每帧 createBuffer 泄漏
  private gpuCache = new Map<number, { vbuf: GPUBuffer; ibuf: GPUBuffer; vcap: number; icap: number }>();
  private static readonly GUM_KEY = -1;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
  }

  static async isSupported(): Promise<boolean> {
    return typeof navigator !== 'undefined' && !!navigator.gpu && (await navigator.gpu.requestAdapter()) != null;
  }

  async init(): Promise<void> {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('找不到 WebGPU 适配器（需要 Chrome/Edge 113+ 或兼容浏览器）');
    this.device = await adapter.requestDevice();
    this.context = this.canvas.getContext('webgpu')!;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({ device: this.device, format: this.format, alphaMode: 'opaque' });

    this.cameraBuffer = this.device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.lightBuffer = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.materialBuffer = this.device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    const module = this.device.createShaderModule({ code: SHADER });
    this.bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });
    const pipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] });
    this.pipeline = this.device.createRenderPipeline({
      layout: pipelineLayout,
      vertex: {
        module,
        entryPoint: 'vs',
        buffers: [{
          arrayStride: 24,
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x3' },
            { shaderLocation: 1, offset: 12, format: 'float32x3' },
          ],
        }],
      },
      fragment: { module, entryPoint: 'fs', targets: [{ format: this.format }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: {
        depthWriteEnabled: true,
        depthCompare: 'less',
        format: 'depth24plus',
      },
    });

    this.bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.cameraBuffer } },
        { binding: 1, resource: { buffer: this.lightBuffer } },
        { binding: 2, resource: { buffer: this.materialBuffer } },
      ],
    });

    this.ensureDepth();
    this.uploadLight();
  }

  private ensureDepth(): void {
    const size = { width: this.canvas.width, height: this.canvas.height };
    if (this.depthTexture) this.depthTexture.destroy();
    this.depthTexture = this.device.createTexture({
      size,
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }

  private uploadLight(): void {
    const light = new Float32Array(4);
    light.set([0.4, -0.6, 0.7, 0]); // 方向光
    this.device.queue.writeBuffer(this.lightBuffer, 0, light);
  }

  private writeMaterial(base: Vec3, roughness: number, sss: number, thickness: number): void {
    const m = new Float32Array(12);
    const effSss = this.sssEnabled ? sss : 0;
    m.set([base[0], base[1], base[2], roughness, effSss, thickness, 0, 0, 0, 0, 0, 0]);
    this.device.queue.writeBuffer(this.materialBuffer, 0, m);
  }

  setObjects(teeth: RenderObject[], gum: RenderObject | null): void {
    this.objects.clear();
    for (const t of teeth) this.objects.set(t.label, t);
    this.gumObject = gum;
  }

  private cameraMatrix(): Mat4 {
    const { yaw, pitch, distance, target } = this.orbit;
    const eye: Vec3 = [
      target[0] + distance * Math.cos(pitch) * Math.sin(yaw),
      target[1] + distance * Math.cos(pitch) * Math.cos(yaw),
      target[2] + distance * Math.sin(pitch),
    ];
    const aspect = this.canvas.width / Math.max(1, this.canvas.height);
    const view = mat4LookAt(eye, target, [0, 0, 1]);
    const proj = mat4Perspective((45 * Math.PI) / 180, aspect, 0.1, 1000);
    const cam = new Float32Array(20);
    const viewProj = mat4Multiply(proj, view);
    cam.set(viewProj, 0);
    cam.set(eye, 16);
    this.device.queue.writeBuffer(this.cameraBuffer, 0, cam);
    return mat4Identity();
  }

  private drawObject(encoder: GPURenderPassEncoder, obj: RenderObject): void {
    const vcount = obj.positions.length / 3;
    const key = obj.isGum ? WebGPURenderer.GUM_KEY : obj.label;
    const vbytes = Math.ceil((vcount * 6 * 4) / 16) * 16;
    const ibytes = Math.ceil(obj.indices.byteLength / 16) * 16;
    let cached = this.gpuCache.get(key);
    if (!cached || cached.vcap < vbytes || cached.icap < ibytes) {
      cached?.vbuf.destroy();
      cached?.ibuf.destroy();
      cached = {
        vbuf: this.device.createBuffer({ size: vbytes, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST }),
        ibuf: this.device.createBuffer({ size: ibytes, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST }),
        vcap: vbytes,
        icap: ibytes,
      };
      this.gpuCache.set(key, cached);
    }
    const { vbuf, ibuf } = cached;
    const interleaved = new Float32Array(vcount * 6);
    for (let i = 0; i < vcount; i++) {
      interleaved[i * 6] = obj.positions[i * 3];
      interleaved[i * 6 + 1] = obj.positions[i * 3 + 1];
      interleaved[i * 6 + 2] = obj.positions[i * 3 + 2];
      interleaved[i * 6 + 3] = obj.normals[i * 3];
      interleaved[i * 6 + 4] = obj.normals[i * 3 + 1];
      interleaved[i * 6 + 5] = obj.normals[i * 3 + 2];
    }
    this.device.queue.writeBuffer(vbuf, 0, interleaved);
    this.device.queue.writeBuffer(ibuf, 0, obj.indices as BufferSource);

    if (obj.isGum) {
      this.writeMaterial([0.85, 0.45, 0.48], 0.7, 0.25, 0.3); // 牙龈：粉红软组织
    } else {
      this.writeMaterial([0.93, 0.9, 0.82], 0.25, 0.9, 0.6); // 釉质：乳白、高 SSS
    }
    encoder.setPipeline(this.pipeline);
    encoder.setBindGroup(0, this.bindGroup);
    encoder.setVertexBuffer(0, vbuf);
    encoder.setIndexBuffer(ibuf, 'uint32');
    encoder.drawIndexed(obj.indices.length);
  }

  render(): void {
    if (!this.device) return;
    if (this.canvas.width === 0 || this.canvas.height === 0) return;
    this.cameraMatrix();
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.context.getCurrentTexture().createView(),
        clearValue: { r: 0.07, g: 0.08, b: 0.1, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
      depthStencilAttachment: {
        view: this.depthTexture.createView(),
        depthClearValue: 1,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    });
    if (this.gumObject) this.drawObject(pass, this.gumObject);
    for (const obj of this.objects.values()) this.drawObject(pass, obj);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  resize(w: number, h: number): void {
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
      this.ensureDepth();
    }
  }
}

/** 把一颗牙的局部顶点按位姿变换到世界系，并同步旋转法线（法线只受旋转、不受平移）。 */
export function poseToWorld(
  part: ToothPart,
  centroid: Vec3,
  position: Vec3,
  q: Quat,
): { positions: Float32Array; normals: Float32Array } {
  const positions = new Float32Array(part.positions.length);
  const normals = new Float32Array(part.normals.length);
  for (let i = 0; i < part.vertexCount; i++) {
    const p: Vec3 = [
      part.positions[i * 3],
      part.positions[i * 3 + 1],
      part.positions[i * 3 + 2],
    ];
    const rel: Vec3 = [p[0] - centroid[0], p[1] - centroid[1], p[2] - centroid[2]];
    const rot = qRotate(q, rel);
    positions[i * 3] = position[0] + rot[0];
    positions[i * 3 + 1] = position[1] + rot[1];
    positions[i * 3 + 2] = position[2] + rot[2];
    const n: Vec3 = [
      part.normals[i * 3],
      part.normals[i * 3 + 1],
      part.normals[i * 3 + 2],
    ];
    const rn = qRotate(q, n);
    normals[i * 3] = rn[0];
    normals[i * 3 + 1] = rn[1];
    normals[i * 3 + 2] = rn[2];
  }
  return { positions, normals };
}
