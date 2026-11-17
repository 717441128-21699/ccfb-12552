// 三角网格数据结构。位置用扁平 Float32Array，索引为 Uint32Array。
// 标签挂在三角面上（口扫分割标签通常逐面给出）：0 = 牙龈，1..32 = FDI 牙位。

export interface RawMesh {
  positions: Float32Array; // 顶点坐标（可能未焊接，每三角形 3 个独立顶点）
  indices: Uint32Array; // 三角形索引
  faceLabels: Uint16Array; // 每个三角形一个标签
}

export interface Mesh {
  positions: Float32Array; // 焊接后的顶点坐标
  normals: Float32Array; // 平滑顶点法线
  indices: Uint32Array;
  faceLabels: Uint16Array;
  vertexCount: number;
  faceCount: number;
  /** 焊接映射：原始顶点序号 -> 焊接后顶点序号 */
  weldMap: Uint32Array;
}

export interface ToothPart {
  label: number; // FDI 牙位
  name: string;
  positions: Float32Array; // 该部件局部（原始世界坐标）顶点
  normals: Float32Array;
  indices: Uint32Array;
  vertexCount: number;
  faceCount: number;
}

export interface SegmentedModel {
  teeth: ToothPart[];
  gum: ToothPart; // label = 0
  /** 全模型焊接网格（牙龈 + 所有牙），用于牙龈变形与整体渲染 */
  fullMesh: Mesh;
}

/** FDI 两位牙位代号 -> 中文名 */
export function toothName(label: number): string {
  if (label <= 0) return '牙龈';
  const quadrant = Math.floor(label / 10);
  const idx = label % 10;
  const arch = quadrant === 1 || quadrant === 4 ? '上颌' : '下颌';
  const names = ['', '中切牙', '侧切牙', '尖牙', '第一前磨牙', '第二前磨牙', '第一磨牙', '第二磨牙', '第三磨牙'];
  const side = quadrant === 1 || quadrant === 2 ? '右' : '左';
  return `${arch}${side}${names[idx] ?? `#${idx}`}(${label})`;
}
