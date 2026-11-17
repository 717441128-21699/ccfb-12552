// 按预标好的面标签把焊接网格拆成单颗牙 + 牙龈。
import type { Mesh, SegmentedModel, ToothPart } from './mesh';
import { extractPart } from './stl';
import { toothName } from './mesh';

export function segmentModel(fullMesh: Mesh): SegmentedModel {
  const labelSet = new Set<number>();
  for (let f = 0; f < fullMesh.faceCount; f++) labelSet.add(fullMesh.faceLabels[f]);

  const teeth: ToothPart[] = [];
  let gum: ToothPart;

  if (labelSet.has(0)) {
    gum = extractPart(fullMesh, 0);
    gum.name = '牙龈';
  } else {
    // 没有牙龈标签时给一个空部件，渲染与变形逻辑统一处理
    gum = {
      label: 0, name: '牙龈',
      positions: new Float32Array(0), normals: new Float32Array(0),
      indices: new Uint32Array(0), vertexCount: 0, faceCount: 0,
    };
  }

  const toothLabels = [...labelSet].filter((l) => l > 0).sort((a, b) => a - b);
  for (const label of toothLabels) {
    const part = extractPart(fullMesh, label);
    part.name = toothName(label);
    teeth.push(part);
  }

  return { teeth, gum, fullMesh };
}

/** 每个标签的面数统计，供 UI 展示 */
export function labelStatistics(mesh: Mesh): Map<number, number> {
  const stats = new Map<number, number>();
  for (let f = 0; f < mesh.faceCount; f++) {
    const l = mesh.faceLabels[f];
    stats.set(l, (stats.get(l) ?? 0) + 1);
  }
  return stats;
}
