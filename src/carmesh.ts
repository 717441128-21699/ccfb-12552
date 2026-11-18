// 车身网格:内置参数化车型 + GLB/STL 导入
// 坐标约定:车长沿 x(车头朝 -x 方向,即风从 -x 吹来之前车头先遇到风),y 向上,z 为车宽方向,单位米。
import { mat4FromTRS, mat4Multiply, mat4Identity, transformPoint, type Mat4 } from './math';

export interface MeshData {
  positions: Float32Array<ArrayBuffer>; // xyz 连续
  normals: Float32Array<ArrayBuffer>; // 焊接后的面积加权平滑法线
  indices: Uint32Array<ArrayBuffer>;
}

export interface CarParams {
  mirrorPos: number; // 0..1 沿车侧前后
  spoilerAngle: number; // 度,绕 z 轴
  spoilerOn: boolean;
  rideHeight: number; // 米,附加离地间隙
}

// ---------------------------------------------------------------- 网格构建器
class MeshBuilder {
  pos: number[] = [];
  idx: number[] = [];

  vert(x: number, y: number, z: number): number {
    this.pos.push(x, y, z);
    return this.pos.length / 3 - 1;
  }
  tri(a: number, b: number, c: number) {
    this.idx.push(a, b, c);
  }
  quad(a: number, b: number, c: number, d: number) {
    this.tri(a, b, c);
    this.tri(a, c, d);
  }

  // 沿 x 的放样体:每个站位一个椭圆截面
  loft(stations: { x: number; w: number; yBot: number; yTop: number }[], n = 14) {
    const rings: number[][] = [];
    for (const s of stations) {
      const yc = (s.yBot + s.yTop) / 2, h = (s.yTop - s.yBot) / 2;
      const ring: number[] = [];
      for (let i = 0; i < n; i++) {
        const t = (i / n) * Math.PI * 2;
        ring.push(this.vert(s.x, yc + h * Math.sin(t), s.w * Math.cos(t)));
      }
      rings.push(ring);
    }
    for (let r = 0; r < rings.length - 1; r++)
      for (let i = 0; i < n; i++) {
        const j = (i + 1) % n;
        this.quad(rings[r][i], rings[r][j], rings[r + 1][j], rings[r + 1][i]);
      }
    // 端盖
    for (const [ring, x, flip] of [
      [rings[0], stations[0].x, true],
      [rings[rings.length - 1], stations[stations.length - 1].x, false],
    ] as const) {
      const cy = ring.reduce((s, v) => s + this.pos[v * 3 + 1], 0) / ring.length;
      const c = this.vert(x, cy, 0);
      for (let i = 0; i < n; i++) {
        const j = (i + 1) % n;
        if (flip) this.tri(c, ring[j], ring[i]);
        else this.tri(c, ring[i], ring[j]);
      }
    }
  }

  // 盒子,可绕 z 轴旋转(用于尾翼)
  box(cx: number, cy: number, cz: number, sx: number, sy: number, sz: number, rotZ = 0) {
    const c = Math.cos(rotZ), s = Math.sin(rotZ);
    const base: number[] = [];
    for (const [dx, dy, dz] of [
      [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
      [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ]) {
      const px = (dx * sx) / 2, py = (dy * sy) / 2;
      base.push(this.vert(cx + px * c - py * s, cy + px * s + py * c, cz + (dz * sz) / 2));
    }
    const F = [[0, 1, 2, 3], [5, 4, 7, 6], [4, 0, 3, 7], [1, 5, 6, 2], [3, 2, 6, 7], [4, 5, 1, 0]];
    for (const f of F) this.quad(base[f[0]], base[f[1]], base[f[2]], base[f[3]]);
  }

  // 沿 z 的圆柱(车轮)
  cylinder(cx: number, cy: number, cz: number, r: number, len: number, n = 14) {
    const r0: number[] = [], r1: number[] = [];
    for (let i = 0; i < n; i++) {
      const t = (i / n) * Math.PI * 2;
      r0.push(this.vert(cx + r * Math.cos(t), cy + r * Math.sin(t), cz - len / 2));
      r1.push(this.vert(cx + r * Math.cos(t), cy + r * Math.sin(t), cz + len / 2));
    }
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      this.quad(r0[i], r0[j], r1[j], r1[i]);
    }
    const c0 = this.vert(cx, cy, cz - len / 2), c1 = this.vert(cx, cy, cz + len / 2);
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      this.tri(c0, r0[j], r0[i]);
      this.tri(c1, r1[i], r1[j]);
    }
  }

  build(): MeshData {
    const positions = new Float32Array(this.pos);
    const indices = new Uint32Array(this.idx);
    const normals = computeSmoothNormals(positions, indices);
    return { positions, normals, indices };
  }
}

// 按位置焊接顶点并计算面积加权平滑法线
export function computeSmoothNormals(
  positions: Float32Array<ArrayBuffer>,
  indices: Uint32Array<ArrayBuffer>,
): Float32Array<ArrayBuffer> {
  const n = positions.length / 3;
  const remap = new Map<string, number>();
  const toCanon = new Uint32Array(n);
  const canonPos: number[] = [];
  for (let i = 0; i < n; i++) {
    const key = `${positions[i * 3].toFixed(4)},${positions[i * 3 + 1].toFixed(4)},${positions[i * 3 + 2].toFixed(4)}`;
    let c = remap.get(key);
    if (c === undefined) {
      c = canonPos.length / 3;
      remap.set(key, c);
      canonPos.push(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
    }
    toCanon[i] = c;
  }
  const acc = new Float32Array(canonPos.length);
  for (let t = 0; t < indices.length; t += 3) {
    const a = toCanon[indices[t]], b = toCanon[indices[t + 1]], c = toCanon[indices[t + 2]];
    const ax = canonPos[a * 3], ay = canonPos[a * 3 + 1], az = canonPos[a * 3 + 2];
    const e1x = canonPos[b * 3] - ax, e1y = canonPos[b * 3 + 1] - ay, e1z = canonPos[b * 3 + 2] - az;
    const e2x = canonPos[c * 3] - ax, e2y = canonPos[c * 3 + 1] - ay, e2z = canonPos[c * 3 + 2] - az;
    const nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
    for (const v of [a, b, c]) {
      acc[v * 3] += nx; acc[v * 3 + 1] += ny; acc[v * 3 + 2] += nz;
    }
  }
  const normals = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const c = toCanon[i];
    const l = Math.hypot(acc[c * 3], acc[c * 3 + 1], acc[c * 3 + 2]) || 1;
    normals[i * 3] = acc[c * 3] / l;
    normals[i * 3 + 1] = acc[c * 3 + 1] / l;
    normals[i * 3 + 2] = acc[c * 3 + 2] / l;
  }
  return normals;
}

// ---------------------------------------------------------------- 内置参数化车型
export function buildProceduralCar(p: CarParams): MeshData {
  const mb = new MeshBuilder();
  const rh = p.rideHeight;

  // 车身主断面(车头在 -x 方向)
  mb.loft([
    { x: -2.2, w: 0.55, yBot: 0.35 + rh, yTop: 0.95 + rh },
    { x: -1.9, w: 0.8, yBot: 0.28 + rh, yTop: 1.05 + rh },
    { x: -1.5, w: 0.88, yBot: 0.22 + rh, yTop: 1.1 + rh },
    { x: -0.8, w: 0.9, yBot: 0.18 + rh, yTop: 1.12 + rh },
    { x: 0.0, w: 0.9, yBot: 0.16 + rh, yTop: 1.1 + rh },
    { x: 0.8, w: 0.88, yBot: 0.18 + rh, yTop: 0.95 + rh },
    { x: 1.4, w: 0.84, yBot: 0.22 + rh, yTop: 0.8 + rh },
    { x: 1.9, w: 0.75, yBot: 0.28 + rh, yTop: 0.68 + rh },
    { x: 2.2, w: 0.6, yBot: 0.35 + rh, yTop: 0.6 + rh },
  ]);
  // 座舱
  mb.loft([
    { x: -1.6, w: 0.55, yBot: 1.0 + rh, yTop: 1.1 + rh },
    { x: -1.2, w: 0.72, yBot: 1.05 + rh, yTop: 1.35 + rh },
    { x: -0.5, w: 0.78, yBot: 1.08 + rh, yTop: 1.45 + rh },
    { x: 0.2, w: 0.76, yBot: 1.05 + rh, yTop: 1.44 + rh },
    { x: 0.8, w: 0.68, yBot: 0.95 + rh, yTop: 1.25 + rh },
    { x: 0.95, w: 0.6, yBot: 0.9 + rh, yTop: 1.05 + rh },
  ]);
  // 车轮(不随离地间隙移动)
  for (const wx of [-1.45, 1.45])
    for (const wz of [-0.72, 0.72]) mb.cylinder(wx, 0.33, wz, 0.33, 0.22);

  // 后视镜:位置可调
  const mx = 0.5 + p.mirrorPos * 0.5; // 0.5..1.0
  for (const side of [-1, 1]) {
    const mz = side * 1.0;
    mb.box(mx, 0.98 + rh, side * 0.93, 0.04, 0.03, 0.16); // 镜杆
    mb.box(mx, 1.0 + rh, mz, 0.07, 0.09, 0.16); // 镜壳
  }

  // 尾翼:角度可调
  if (p.spoilerOn) {
    const ang = (-p.spoilerAngle * Math.PI) / 180;
    for (const side of [-1, 1]) mb.box(-1.98, 1.05 + rh, side * 0.42, 0.06, 0.28, 0.05);
    mb.box(-2.05, 1.2 + rh, 0, 0.32, 0.025, 1.35, ang);
  }
  return mb.build();
}

// ---------------------------------------------------------------- 坐标归一化
// 把任意导入模型摆到约定位姿:最长水平轴对齐 x,底面贴 y=0,中心在原点
export function normalizeMesh(mesh: MeshData, flipX: boolean): MeshData {
  const pos = mesh.positions;
  let minX = 1e9, maxX = -1e9, minY = 1e9, maxY = -1e9, minZ = 1e9, maxZ = -1e9;
  for (let i = 0; i < pos.length; i += 3) {
    minX = Math.min(minX, pos[i]); maxX = Math.max(maxX, pos[i]);
    minY = Math.min(minY, pos[i + 1]); maxY = Math.max(maxY, pos[i + 1]);
    minZ = Math.min(minZ, pos[i + 2]); maxZ = Math.max(maxZ, pos[i + 2]);
  }
  const spanX = maxX - minX, spanZ = maxZ - minZ;
  const rotate = spanZ > spanX; // 长轴在 z 上则绕 y 转 90°
  const len = Math.max(spanX, spanZ);
  const s = 4.4 / len; // 统一缩放到 4.4 m 车长
  const cx = (minX + maxX) / 2, cz = (minZ + maxZ) / 2;
  const out = new Float32Array(pos.length);
  for (let i = 0; i < pos.length; i += 3) {
    let x = (pos[i] - cx) * s;
    const y = (pos[i + 1] - minY) * s;
    let z = (pos[i + 2] - cz) * s;
    if (rotate) [x, z] = [z, -x];
    if (flipX) x = -x;
    out[i] = x; out[i + 1] = y; out[i + 2] = z;
  }
  const normals = computeSmoothNormals(out, mesh.indices);
  return { positions: out, normals, indices: mesh.indices };
}

// 米制网格 → 格子坐标;flipX 使车头(本地 +x)朝向网格 -x 的来流方向
export function meshToGrid(
  mesh: MeshData,
  scale: number,
  ox: number,
  oy: number,
  oz: number,
  flipX = true,
): MeshData {
  const n = mesh.positions.length;
  const pos = new Float32Array(n);
  const sx = flipX ? -scale : scale;
  for (let i = 0; i < n; i += 3) {
    pos[i] = mesh.positions[i] * sx + ox;
    pos[i + 1] = mesh.positions[i + 1] * scale + oy;
    pos[i + 2] = mesh.positions[i + 2] * scale + oz;
  }
  return { positions: pos, normals: mesh.normals, indices: mesh.indices };
}

// ---------------------------------------------------------------- 文件加载
export async function loadMeshFile(file: File): Promise<MeshData> {
  const buf = await file.arrayBuffer();
  if (/\.stl$/i.test(file.name)) return parseSTL(buf);
  if (/\.glb$/i.test(file.name)) return parseGLB(buf);
  throw new Error('仅支持 .glb / .stl 文件');
}

function parseSTL(buf: ArrayBuffer): MeshData {
  const bytes = new Uint8Array(buf);
  const head = new TextDecoder().decode(bytes.slice(0, 512));
  const isAscii = head.trimStart().startsWith('solid') && head.includes('facet');
  const positions: number[] = [];
  if (isAscii) {
    const text = new TextDecoder().decode(bytes);
    const re = /vertex\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)/g;
    let m: RegExpExecArray | null;
    while ((m = re.exec(text))) positions.push(+m[1], +m[2], +m[3]);
  } else {
    const dv = new DataView(buf);
    const count = dv.getUint32(80, true);
    for (let t = 0; t < count; t++) {
      const o = 84 + t * 50 + 12; // 跳过法线
      for (let v = 0; v < 3; v++)
        positions.push(dv.getFloat32(o + v * 12, true), dv.getFloat32(o + v * 12 + 4, true), dv.getFloat32(o + v * 12 + 8, true));
    }
  }
  const pos = new Float32Array(positions);
  const indices = new Uint32Array(pos.length / 3);
  for (let i = 0; i < indices.length; i++) indices[i] = i;
  const normals = computeSmoothNormals(pos, indices);
  return { positions: pos, normals, indices };
}

function parseGLB(buf: ArrayBuffer): MeshData {
  const dv = new DataView(buf);
  if (dv.getUint32(0, true) !== 0x46546c67) throw new Error('不是有效的 GLB 文件');
  const jsonLen = dv.getUint32(12, true);
  const json = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 20, jsonLen)));
  const binStart = 20 + jsonLen + 8;

  const readAccessor = (ai: number): { data: Float32Array | Uint32Array | Uint16Array; count: number; comps: number } => {
    const acc = json.accessors[ai];
    const bv = json.bufferViews[acc.bufferView];
    const start = binStart + (bv.byteOffset ?? 0) + (acc.byteOffset ?? 0);
    const comps = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 }[acc.type as string] ?? 3;
    const stride = bv.byteStride ?? 0;
    let data: Float32Array | Uint32Array | Uint16Array;
    if (acc.componentType === 5126) {
      data = new Float32Array(acc.count * comps);
      if (stride === comps * 4 || stride === 0) {
        (data as Float32Array).set(new Float32Array(buf, start, acc.count * comps));
      } else {
        for (let i = 0; i < acc.count; i++)
          for (let c = 0; c < comps; c++)
            (data as Float32Array)[i * comps + c] = dv.getFloat32(start + i * stride + c * 4, true);
      }
    } else if (acc.componentType === 5125) {
      data = new Uint32Array(buf.slice(start, start + acc.count * 4));
    } else if (acc.componentType === 5123) {
      data = new Uint16Array(buf.slice(start, start + acc.count * 2));
    } else if (acc.componentType === 5121) {
      data = new Uint16Array(new Uint8Array(buf.slice(start, start + acc.count)));
    } else {
      throw new Error(`不支持的 accessor 类型 ${acc.componentType}`);
    }
    return { data, count: acc.count, comps };
  };

  const positions: number[] = [];
  const indices: number[] = [];

  const walkNode = (ni: number, parent: Mat4) => {
    const node = json.nodes[ni];
    let local: Mat4;
    if (node.matrix) {
      local = new Float32Array(node.matrix);
    } else {
      local = mat4FromTRS(
        node.translation ?? [0, 0, 0],
        node.rotation ?? [0, 0, 0, 1],
        node.scale ?? [1, 1, 1],
      );
    }
    const world = mat4Multiply(parent, local);
    if (node.mesh !== undefined) {
      for (const prim of json.meshes[node.mesh].primitives) {
        if ((prim.mode ?? 4) !== 4 || prim.attributes.POSITION === undefined) continue;
        const pa = readAccessor(prim.attributes.POSITION);
        const base = positions.length / 3;
        for (let i = 0; i < pa.count; i++) {
          const p = transformPoint(world, pa.data[i * 3], pa.data[i * 3 + 1], pa.data[i * 3 + 2]);
          positions.push(p[0], p[1], p[2]);
        }
        if (prim.indices !== undefined) {
          const ia = readAccessor(prim.indices);
          for (let i = 0; i < ia.count; i++) indices.push(base + ia.data[i]);
        } else {
          for (let i = 0; i < pa.count; i++) indices.push(base + i);
        }
      }
    }
    for (const c of node.children ?? []) walkNode(c, world);
  };

  const scene = json.scenes[json.scene ?? 0];
  for (const ni of scene.nodes) walkNode(ni, mat4Identity());
  if (positions.length === 0) throw new Error('GLB 中没有可用的三角网格');

  const pos = new Float32Array(positions);
  const idx = new Uint32Array(indices);
  const normals = computeSmoothNormals(pos, idx);
  return { positions: pos, normals, indices: idx };
}
