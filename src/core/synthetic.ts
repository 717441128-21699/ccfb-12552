// 程序化合成牙列：用于无厂家文件时的演示，以及去釉算法的几何真值验证。
// 牙体用带圆角的盒体（包围盒精确可控），牙龈用牙颈轮廓放样的带状网格，
// 输出带面标签的 RawMesh，再走与真实 STL 完全相同的焊接/分割管线。
import type { RawMesh, ToothPart } from './mesh';
import { makeRng } from './math';

interface Builder {
  positions: number[];
  indices: number[];
  faceLabels: number[];
  addVertex(x: number, y: number, z: number): number;
  addTri(a: number, b: number, c: number, label: number): void;
}

function newBuilder(): Builder {
  const positions: number[] = [];
  const indices: number[] = [];
  const faceLabels: number[] = [];
  return {
    positions,
    indices,
    faceLabels,
    addVertex(x, y, z) {
      positions.push(x, y, z);
      return positions.length / 3 - 1;
    },
    addTri(a, b, c, label) {
      indices.push(a, b, c);
      faceLabels.push(label);
    },
  };
}

export interface BuiltRaw {
  raw: RawMesh;
  /** 生成参数（含几何真值），供测试断言 */
  meta: Record<string, unknown>;
}

function finish(b: Builder, meta: Record<string, unknown>): BuiltRaw {
  return {
    raw: {
      positions: Float32Array.from(b.positions),
      indices: Uint32Array.from(b.indices),
      faceLabels: Uint16Array.from(b.faceLabels),
    },
    meta,
  };
}

/**
 * 轴对齐圆角盒牙体（盒体 + 倒角顶点近似）。
 * 包围盒严格为 [cx-hw, cx+hw] × [cy-hb, cy+hb] × [cz, cz+h]。
 */
export function addBoxTooth(
  b: Builder,
  label: number,
  cx: number, cy: number,
  hw: number, hb: number, height: number,
  cz: number,
): void {
  const x0 = cx - hw, x1 = cx + hw;
  const y0 = cy - hb, y1 = cy + hb;
  const z0 = cz, z1 = cz + height;
  const p = [
    b.addVertex(x0, y0, z0), b.addVertex(x1, y0, z0),
    b.addVertex(x1, y1, z0), b.addVertex(x0, y1, z0),
    b.addVertex(x0, y0, z1), b.addVertex(x1, y0, z1),
    b.addVertex(x1, y1, z1), b.addVertex(x0, y1, z1),
  ];
  const tris: [number, number, number][] = [
    [0, 2, 1], [0, 3, 2], // 底
    [4, 5, 6], [4, 6, 7], // 顶（咬合面）
    [0, 1, 5], [0, 5, 4], // -y
    [1, 2, 6], [1, 6, 5], // +x
    [2, 3, 7], [2, 7, 6], // +y
    [3, 0, 4], [3, 4, 7], // -x
  ];
  for (const [a, c, d] of tris) b.addTri(p[a], p[c], p[d], label);
}

/** 合并多个 BuiltRaw（直接拼接，标签保持不变） */
export function mergeRaws(parts: BuiltRaw[]): BuiltRaw {
  const positions: number[] = [];
  const indices: number[] = [];
  const faceLabels: number[] = [];
  const meta: { parts: unknown[] } = { parts: [] };
  let offset = 0;
  for (const part of parts) {
    positions.push(...part.raw.positions);
    for (const i of part.raw.indices) indices.push(i + offset);
    faceLabels.push(...part.raw.faceLabels);
    offset += part.raw.positions.length / 3;
    meta.parts.push(part.meta);
  }
  return {
    raw: {
      positions: Float32Array.from(positions),
      indices: Uint32Array.from(indices),
      faceLabels: Uint16Array.from(faceLabels),
    },
    meta,
  };
}

/**
 * 两颗已知间距的测试牙：两个等宽盒体沿 x 轴排列。
 * @param width 每颗牙 x 方向宽度（mm）
 * @param gap 两牙之间的带符号间隙（mm，负值=重叠）
 * 真值：重叠 = -gap（gap<0 时），每面分配 = -gap/2。
 */
export function buildTwoToothTestModel(width = 4, depth = 4, height = 8, gap = -0.6): BuiltRaw {
  const b = newBuilder();
  // 牙 A：[-width, 0]；牙 B：[gap, gap+width]（A 右表面 x=0，B 左表面 x=gap）
  addBoxTooth(b, 11, -width / 2, 0, width / 2, depth / 2, height, 0);
  addBoxTooth(b, 21, gap + width / 2, 0, width / 2, depth / 2, height, 0);
  return finish(b, {
    kind: 'two-tooth',
    width,
    depth,
    height,
    gap,
    truth: {
      overlap: Math.max(0, -gap),
      stripPerSurface: Math.max(0, -gap) / 2,
      crownWidth: width,
    },
  });
}

/**
 * 演示上颌牙列：14 颗盒体牙排在椭圆牙弓上，外加一条与牙颈共享顶点的牙龈带。
 * 标签使用 FDI：右上 11..17、左上 21..27。
 */
export function buildDemoArch(): BuiltRaw {
  const b = newBuilder();
  const rng = makeRng(42);
  const labels = [17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27];
  const teeth: {
    label: number; cx: number; cy: number; hw: number; hb: number; h: number;
  }[] = [];

  const archA = 26; // 椭圆半轴 x
  const archB = 20; // 椭圆半轴 y
  labels.forEach((label, i) => {
    // 沿半椭圆参数 t 从 0..π 均匀取 14 个点
    const t = Math.PI * (i / (labels.length - 1));
    const cx = archA * Math.cos(t);
    const cy = archB * Math.sin(t) - 6;
    const idx = label % 10;
    // 前牙窄、磨牙宽
    const hw = idx <= 2 ? 2.1 + rng() * 0.2 : idx <= 5 ? 2.6 : 3.1;
    const hb = idx <= 2 ? 1.9 : 2.6;
    const h = 7.5 + rng() * 1.2;
    teeth.push({ label, cx, cy, hw, hb, h });
    addBoxTooth(b, label, cx, cy, hw, hb, h, -2);
  });

  // 牙龈拓扑（沿牙弓的规则栅格带），设计目标：所有边等长、无跨牙长边、
  // 牙颈边界与牙龈共享顶点、远端有固定锚点环，给变形提供平滑消散路径。
  //
  // 结构：取每颗牙颈圈的颊侧两角(corner0,1)与舌侧两角(corner2,3)，分别组成
  // 颊侧/舌侧两条沿牙弓的脊线；牙与牙之间在脊线上插入等距牙龈顶点（牙间乳头）。
  // 每条脊线再向径向（离开牙弓中心）规则外扩 fixedLayers 层，最外层是固定黏膜。
  const baseIndex = (toothOrdinal: number, corner: number) => toothOrdinal * 8 + corner;
  const archCenterY = teeth.reduce((s, t) => s + t.cy, 0) / teeth.length;
  const radialDir = (x: number, y: number): [number, number] => {
    let dx = x;
    let dy = y - archCenterY;
    const l = Math.hypot(dx, dy) || 1;
    return [dx / l, dy / l];
  };

  // 沿牙弓把一组锚点（颈圈角）重采样成等距脊线，牙间插入牙龈自有乳头点
  const resampleSpine = (anchors: { x: number; y: number; toothVertex?: number }[], spacing: number, z: number) => {
    const spine: { v: number; toothVertex?: number }[] = [];
    for (let i = 0; i < anchors.length; i++) {
      const a = anchors[i];
      if (a.toothVertex !== undefined) {
        spine.push({ v: a.toothVertex, toothVertex: a.toothVertex }); // 复用牙颈顶点
      } else {
        spine.push({ v: b.addVertex(a.x, a.y, z) });
      }
      if (i === anchors.length - 1) break;
      const c = anchors[i + 1];
      const gap = Math.hypot(c.x - a.x, c.y - a.y);
      const pieces = Math.max(1, Math.round(gap / spacing));
      for (let s = 1; s < pieces; s++) {
        const t = s / pieces;
        spine.push({ v: b.addVertex(a.x + (c.x - a.x) * t, a.y + (c.y - a.y) * t, z) });
      }
    }
    return spine;
  };

  const spineSpacing = 1.5;
  // 颊侧脊线锚点：每牙 corner0、corner1（沿牙弓前进方向的两个颊侧颈圈角）
  const buccalAnchors = teeth.flatMap((_t, ord) => [
    { x: b.positions[baseIndex(ord, 0) * 3], y: b.positions[baseIndex(ord, 0) * 3 + 1], toothVertex: baseIndex(ord, 0) },
    { x: b.positions[baseIndex(ord, 1) * 3], y: b.positions[baseIndex(ord, 1) * 3 + 1], toothVertex: baseIndex(ord, 1) },
  ]);
  // 舌侧脊线锚点：corner3、corner2
  const lingualAnchors = teeth.flatMap((_t, ord) => [
    { x: b.positions[baseIndex(ord, 3) * 3], y: b.positions[baseIndex(ord, 3) * 3 + 1], toothVertex: baseIndex(ord, 3) },
    { x: b.positions[baseIndex(ord, 2) * 3], y: b.positions[baseIndex(ord, 2) * 3 + 1], toothVertex: baseIndex(ord, 2) },
  ]);
  const buccalSpine = resampleSpine(buccalAnchors, spineSpacing, -2.2);
  const lingualSpine = resampleSpine(lingualAnchors, spineSpacing, -2.2);

  // 每条脊线径向规则外扩 fixedLayers 层，层间距 layerStep，最外层固定
  const fixedLayers = 12;
  const layerStep = 0.7;
  const buildBand = (spine: { v: number; toothVertex?: number }[]) => {
    let prev = spine.map((s) => s.v);
    for (let layer = 1; layer <= fixedLayers; layer++) {
      const ring: number[] = prev.map((vi) => {
        if (layer === 1 && spine.some((s) => s.v === vi && s.toothVertex !== undefined)) {
          // 第一层不复制牙颈顶点本身，而是为它建外扩点
        }
        const x = b.positions[vi * 3];
        const y = b.positions[vi * 3 + 1];
        const [dx, dy] = radialDir(x, y);
        return b.addVertex(x + dx * layerStep, y + dy * layerStep, -2.2 - layer * 0.18);
      });
      const m = ring.length;
      for (let i = 0; i < m - 1; i++) {
        b.addTri(prev[i], ring[i], ring[i + 1], 0);
        b.addTri(prev[i], ring[i + 1], prev[i + 1], 0);
      }
      prev = ring;
    }
  };
  buildBand(buccalSpine);
  buildBand(lingualSpine);

  // 牙弓两端封口：把颊/舌脊线的首尾端点用三角连起来（短边，局部连接）
  const connectEnds = (bs: { v: number }[], ls: { v: number }[]) => {
    // 脊线第一个/最后一个锚点对应同牙的不同角，距离为牙宽（短边），直接连面
    b.addTri(bs[0].v, ls[0].v, ls[Math.min(1, ls.length - 1)].v, 0);
    b.addTri(bs[0].v, ls[Math.min(1, ls.length - 1)].v, bs[Math.min(1, bs.length - 1)].v, 0);
    const bn = bs.length - 1;
    const ln = ls.length - 1;
    b.addTri(bs[bn].v, ls[ln].v, ls[Math.max(0, ln - 1)].v, 0);
    b.addTri(bs[bn].v, ls[Math.max(0, ln - 1)].v, bs[Math.max(0, bn - 1)].v, 0);
  };
  connectEnds(buccalSpine, lingualSpine);

  return finish(b, { kind: 'demo-arch', toothCount: teeth.length, teeth });
}

/** 把 RawMesh 按标签拆成独立盒体部件（测试辅助，不走焊接） */
export function rawToParts(raw: RawMesh): ToothPart[] {
  const labels = [...new Set(raw.faceLabels)].filter((l) => l > 0).sort((a, b) => a - b);
  return labels.map((label) => {
    const vmap = new Map<number, number>();
    const positions: number[] = [];
    const indices: number[] = [];
    for (let f = 0; f < raw.faceLabels.length; f++) {
      if (raw.faceLabels[f] !== label) continue;
      for (let k = 0; k < 3; k++) {
        const src = raw.indices[f * 3 + k];
        let dst = vmap.get(src);
        if (dst === undefined) {
          dst = positions.length / 3;
          vmap.set(src, dst);
          positions.push(
            raw.positions[src * 3],
            raw.positions[src * 3 + 1],
            raw.positions[src * 3 + 2],
          );
        }
        indices.push(dst);
      }
    }
    const pos = Float32Array.from(positions);
    return {
      label,
      name: `#${label}`,
      positions: pos,
      normals: new Float32Array(pos.length),
      indices: Uint32Array.from(indices),
      vertexCount: pos.length / 3,
      faceCount: indices.length / 3,
    };
  });
}
