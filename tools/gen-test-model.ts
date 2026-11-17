// 导出"两颗已知间距测试牙"为二进制 STL + 面标签旁车文件，
// 用于在外部软件中复算去釉量、与本工具结果做几何真值比对（容差 0.05mm）。
// 用法：npm run gen:testmodel
import { writeFileSync, mkdirSync } from 'node:fs';
import { buildTwoToothTestModel } from '../src/core/synthetic';
import { encodeBinarySTL } from '../src/app';

const built = buildTwoToothTestModel(4, 4, 8, -0.6);
mkdirSync('out', { recursive: true });

const blob = encodeBinarySTL([
  { positions: built.raw.positions, indices: built.raw.indices },
]);
const buf = Buffer.from(await blob.arrayBuffer());
writeFileSync('out/two-tooth-test.stl', buf);

// 面标签：每三角形一行（0=牙龈，这里两颗牙分别为 11、21）
const labels = [...built.raw.faceLabels].join('\n');
writeFileSync('out/two-tooth-test.lab', labels + '\n');

const truth = (built.meta as { truth: { overlap: number; stripPerSurface: number } }).truth;
console.log('已生成 out/two-tooth-test.stl 与 .lab');
console.log('几何真值：重叠', truth.overlap, 'mm；每邻面应去釉', truth.stripPerSurface, 'mm');
console.log('验收：本工具计算的去釉量与真值偏差应 <= 0.05mm');
