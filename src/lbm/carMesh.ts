/**
 * 内置参数化"车身"：由若干盒体/椭球/圆柱拼成的简化轿车，
 * 供造型师在没有外部模型时直接调参（后视镜位置、尾翼角度/高度）。
 * 单位为任意模型单位，体素化阶段统一缩放。
 */
import type { TriangleMesh } from './voxelize';

export interface CarParams {
  /** 后视镜相对车身侧面的外移量（模型单位） */
  mirrorOffset: number;
  /** 尾翼攻角，度 */
  wingAngleDeg: number;
  /** 尾翼安装高度 */
  wingHeight: number;
}

export class MeshBuilder {
  positions: number[] = [];

  /** 直接追加一个三角形（三个顶点的扁平 xyz 坐标） */
  private tri(
    ax: number, ay: number, az: number,
    bx: number, by: number, bz: number,
    cx: number, cy: number, cz: number,
  ): void {
    this.positions.push(
      ax, ay, az,
      bx, by, bz,
      cx, cy, cz,
    );
  }

  /** 由 8 个角点构建盒体/台体，每面两个三角形 */
  private boxFromCorners(p: Array<[number, number, number]>): void {
    const f = (a: number, b: number, c: number, d: number) => {
      this.tri(...p[a], ...p[b], ...p[c]);
      this.tri(...p[a], ...p[c], ...p[d]);
    };
    f(0, 3, 2, 1); f(4, 5, 6, 7);
    f(0, 1, 5, 4); f(2, 3, 7, 6);
    f(1, 2, 6, 5); f(0, 4, 7, 3);
  }

  /** 轴对齐盒体，中心点 + 半尺寸 */
  addBox(cx: number, cy: number, cz: number, hx: number, hy: number, hz: number): void {
    const p: Array<[number, number, number]> = [
      [cx - hx, cy - hy, cz - hz], [cx + hx, cy - hy, cz - hz],
      [cx + hx, cy + hy, cz - hz], [cx - hx, cy + hy, cz - hz],
      [cx - hx, cy - hy, cz + hz], [cx + hx, cy - hy, cz + hz],
      [cx + hx, cy + hy, cz + hz], [cx - hx, cy + hy, cz + hz],
    ];
    this.boxFromCorners(p);
  }

  /** 梯形台（四座标分别给半宽/半高），沿 x 方向，用于座舱 */
  addFrustum(
    x0: number, x1: number,
    halfW0: number, halfW1: number,
    y0: number, y1: number,
  ): void {
    const p: Array<[number, number, number]> = [
      [x0, y0, -halfW0], [x0, y0, halfW0], [x0, y1, halfW0], [x0, y1, -halfW0],
      [x1, y0, -halfW1], [x1, y0, halfW1], [x1, y1, halfW1], [x1, y1, -halfW1],
    ];
    this.boxFromCorners(p);
  }

  /** 圆柱体，轴沿 z（车轮） */
  addCylinderZ(cx: number, cy: number, cz: number, r: number, halfLen: number, seg = 14): void {
    const ring0: Array<[number, number, number]> = [];
    const ring1: Array<[number, number, number]> = [];
    for (let s = 0; s < seg; s++) {
      const a = (s / seg) * Math.PI * 2;
      const rx = r * Math.cos(a), ry = r * Math.sin(a);
      ring0.push([cx + rx, cy + ry, cz - halfLen]);
      ring1.push([cx + rx, cy + ry, cz + halfLen]);
    }
    const c0: [number, number, number] = [cx, cy, cz - halfLen];
    const c1: [number, number, number] = [cx, cy, cz + halfLen];
    for (let s = 0; s < seg; s++) {
      const n = (s + 1) % seg;
      this.tri(...ring0[s], ...ring1[s], ...ring1[n]);
      this.tri(...ring0[s], ...ring1[n], ...ring0[n]);
      this.tri(...c0, ...ring0[s], ...ring0[n]);
      this.tri(...c1, ...ring1[n], ...ring1[s]);
    }
  }

  /** 绕 z 轴旋转的盒体（尾翼），angleDeg 为弦线相对水平面的倾角 */
  addRotatedBoxZ(
    cx: number, cy: number, cz: number,
    hx: number, hy: number, hz: number,
    angleDeg: number,
  ): void {
    const a = (angleDeg * Math.PI) / 180;
    const cos = Math.cos(a), sin = Math.sin(a);
    const local: Array<[number, number, number]> = [
      [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
      [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
    ];
    const p: Array<[number, number, number]> = local.map(([x, y, z]) => [
      cx + x * cos - y * sin,
      cy + x * sin + y * cos,
      cz + z,
    ]);
    this.boxFromCorners(p);
  }

  build(): TriangleMesh {
    return { positions: new Float32Array(this.positions) };
  }
}

export function buildCarMesh(params: CarParams): TriangleMesh {
  const m = new MeshBuilder();

  // 下车身（长 4.4 / 宽 1.8 / 高 0.7），底面 y=0
  m.addBox(0, 0.42, 0, 2.2, 0.42, 0.9);
  // 车头发动机盖略收
  m.addBox(1.55, 0.86, 0, 0.65, 0.12, 0.82);
  // 后备箱盖
  m.addBox(-1.7, 0.88, 0, 0.5, 0.1, 0.82);
  // 座舱梯形台（前宽后窄、向上收）
  m.addFrustum(-1.15, 0.95, 0.78, 0.66, 0.86, 1.42);
  // 车顶小盖，消除台阶锯齿
  m.addBox(-0.1, 1.45, 0, 1.0, 0.06, 0.64);

  // 四个车轮（半径 0.34），车身两侧
  const wheelX = [1.35, -1.4];
  const wheelZ = 0.86;
  for (const wx of wheelX) {
    m.addCylinderZ(wx, 0.34, -wheelZ, 0.34, 0.18);
    m.addCylinderZ(wx, 0.34, wheelZ, 0.34, 0.18);
  }

  // 后视镜：前门两侧小盒，横向位置由 mirrorOffset 控制
  const mirrorZ = 0.92 + params.mirrorOffset * 0.06;
  m.addBox(0.55, 1.02, -mirrorZ, 0.14, 0.1, 0.1);
  m.addBox(0.55, 1.02, mirrorZ, 0.14, 0.1, 0.1);
  // 镜柄
  m.addBox(0.62, 0.98, -(0.82 + params.mirrorOffset * 0.03), 0.08, 0.04, 0.12);
  m.addBox(0.62, 0.98, (0.82 + params.mirrorOffset * 0.03), 0.08, 0.04, 0.12);

  // 尾翼：两翼片支架 + 水平翼面（绕 z 旋转）
  const wingY = 1.18 + params.wingHeight * 0.07;
  const wingX = -2.05;
  m.addBox(wingX + 0.12, (0.92 + wingY) / 2, -0.55, 0.05, (wingY - 0.92) / 2, 0.04);
  m.addBox(wingX + 0.12, (0.92 + wingY) / 2, 0.55, 0.05, (wingY - 0.92) / 2, 0.04);
  m.addRotatedBoxZ(wingX, wingY, 0, 0.42, 0.035, 0.85, params.wingAngleDeg);

  return m.build();
}

/** 极简 Wavefront OBJ 解析：支持 v / f（含 vt、vn 索引），多边形扇面三角化 */
export function parseOBJ(text: string): TriangleMesh {
  const verts: number[] = [];
  const positions: number[] = [];
  const lines = text.split(/\r?\n/);
  for (const line of lines) {
    if (line.startsWith('v ')) {
      const [, x, y, z] = line.trim().split(/\s+/);
      verts.push(Number(x), Number(y), Number(z));
    } else if (line.startsWith('f ')) {
      const refs = line.trim().split(/\s+/).slice(1).map((tok) => {
        const vi = Number(tok.split('/')[0]);
        return vi > 0 ? vi - 1 : verts.length / 3 + vi;
      });
      for (let i = 1; i < refs.length - 1; i++) {
        for (const vi of [refs[0], refs[i], refs[i + 1]]) {
          positions.push(verts[vi * 3], verts[vi * 3 + 1], verts[vi * 3 + 2]);
        }
      }
    }
  }
  return { positions: new Float32Array(positions) };
}
