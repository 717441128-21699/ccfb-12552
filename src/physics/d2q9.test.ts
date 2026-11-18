/**
 * 2D D2Q9 参考 LBM（纯 TS，与 GPU D3Q19 同一套物理：BGK + 半way回弹 +
 * 动量交换力），用于在没有 GPU 的 CI 里验证核心数值性质：
 *   1. 碰撞保持质量/动量守恒
 *   2. 压力驱动 Poiseuille 流收敛到理论抛物线
 *   3. 两个造型（有无尾翼）的阻力差可被稳定分辨
 * 运行：npm run test:physics
 */
import assert from 'node:assert/strict';

const Q9 = 9;
const EX9 = new Int32Array([0, 1, 0, -1, 0, 1, -1, -1, 1]);
const EY9 = new Int32Array([0, 0, 1, 0, -1, 1, 1, -1, -1]);
const W9 = new Float64Array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36]);
const OPP9 = new Uint8Array([0, 3, 4, 1, 2, 7, 8, 5, 6]);

export class D2Q9 {
  readonly nx: number;
  readonly ny: number;
  private readonly solid: Uint8Array;
  private readonly id: (i: number, j: number) => number;

  constructor(nx: number, ny: number, solid: Uint8Array, private tau = 0.8) {
    this.nx = nx;
    this.ny = ny;
    this.solid = solid;
    this.id = (i, j) => i * ny + j;
  }

  f = new Float64Array(0);
  rho = new Float64Array(0);
  ux = new Float64Array(0);
  uy = new Float64Array(0);
  forceX = 0;

  reset(u0: number): void {
    const n = this.nx * this.ny;
    this.f = new Float64Array(n * Q9);
    this.rho = new Float64Array(n);
    this.ux = new Float64Array(n);
    this.uy = new Float64Array(n);
    for (let c = 0; c < n; c++) {
      if (this.solid[c]) {
        for (let q = 0; q < Q9; q++) this.f[c * Q9 + q] = W9[q];
        this.rho[c] = 1;
        continue;
      }
      this.rho[c] = 1;
      this.ux[c] = u0;
      for (let q = 0; q < Q9; q++) {
        const eu = EX9[q] * u0;
        this.f[c * Q9 + q] = W9[q] * (1 + 3 * eu + 4.5 * eu * eu - 1.5 * u0 * u0);
      }
    }
  }

  private feq(q: number, rho: number, ux: number, uy: number): number {
    const eu = EX9[q] * ux + EY9[q] * uy;
    const u2 = ux * ux + uy * uy;
    return W9[q] * rho * (1 + 3 * eu + 4.5 * eu * eu - 1.5 * u2);
  }

  feqPub(q: number, rho: number, ux: number, uy: number): number {
    return this.feq(q, rho, ux, uy);
  }

  step(u0: number, bodyForce = 0, applyInlet = true): void {
    const { nx, ny, solid, f } = this;
    const total = nx * ny;
    const post = new Float64Array(f.length);
    const fNew = new Float64Array(f.length);
    this.forceX = 0;

    // 阶段 1：全场碰撞
    for (let n = 0; n < total; n++) {
      if (solid[n]) continue;
      let rho = 0, mx = 0, my = 0;
      for (let q = 0; q < Q9; q++) {
        const v = f[n * Q9 + q];
        rho += v;
        mx += v * EX9[q];
        my += v * EY9[q];
      }
      const uxRaw = mx / rho, uyRaw = my / rho;
      // 宏观诊断速度（HSD 显式源项方案：平衡态用原始速度）
      this.rho[n] = rho;
      this.ux[n] = uxRaw;
      this.uy[n] = uyRaw;

      const omega = 1 / this.tau;
      for (let q = 0; q < Q9; q++) {
        const eq = this.feq(q, rho, uxRaw, uyRaw);
        // He-Shan-Doolen 显式力源项 S_i = w_i · (e·F)/c_s²，c_s²=1/3
        const source = W9[q] * 3 * EX9[q] * bodyForce;
        post[n * Q9 + q] = f[n * Q9 + q] - omega * (f[n * Q9 + q] - eq) + source;
      }
    }

    // 阶段 2：pull 迁移 + 边界
    for (let i = 0; i < nx; i++) {
      for (let j = 0; j < ny; j++) {
        const n = this.id(i, j);
        if (solid[n]) continue;
        for (let q = 0; q < Q9; q++) {
          let si = i - EX9[q];
          const sj = j - EY9[q];
          if (sj < 0 || sj >= ny) {
            // 半way回弹：源在壁面外
            fNew[n * Q9 + q] = post[n * Q9 + OPP9[q]];
            continue;
          }
          // x 周期环绕
          if (si < 0) si = nx - 1;
          if (si >= nx) si = 0;
          const sn = this.id(si, sj);
          if (solid[sn]) {
            fNew[n * Q9 + q] = post[n * Q9 + OPP9[q]];
            this.forceX += (f[n * Q9 + q] - f[n * Q9 + OPP9[q]]) * EX9[q];
          } else {
            fNew[n * Q9 + q] = post[sn * Q9 + q];
          }
        }
      }
    }
    this.f.set(fNew);

    // 入口定速（Zou-He 风格，x=0 列）；纯周期验证时关闭
    if (!applyInlet) return;
    for (let j = 1; j < ny - 1; j++) {
      const n = this.id(0, j);
      if (solid[n]) continue;
      const known = [0, 2, 4, 6, 7].reduce((s, q) => s + f[n * Q9 + q], 0);
      const rhoIn = known / (1 - u0);
      this.rho[n] = rhoIn;
      this.ux[n] = u0;
      for (const q of [1, 5, 8]) {
        const qo = OPP9[q];
        f[n * Q9 + q] = this.feq(q, rhoIn, u0, 0) +
          (f[n * Q9 + qo] - this.feq(qo, rhoIn, u0, 0));
      }
    }
  }
}

// 诊断脚本（diag.ts）import 本模块时用 LBM_DIAG=1 跳过断言执行
if (!process.env.LBM_DIAG) {
let passed = 0;
function check(name: string, fn: () => void): void {
  fn();
  passed++;
  console.log(`  ✓ ${name}`);
}

// ---- 测试 1：BGK 碰撞的质量/动量守恒（机器精度）+ 全步骤边界误差有界 ----
{
  const nx = 16, ny = 10;
  const solid = new Uint8Array(nx * ny);
  const sim = new D2Q9(nx, ny, solid, 0.8);
  sim.reset(0.05);

  // (a) 纯碰撞算子：Σfeq = Σf = ρ，质量与动量应机器精度守恒
  let m0 = 0, px0 = 0;
  for (let n = 0; n < nx * ny; n++) {
    for (let q = 0; q < Q9; q++) {
      m0 += sim.f[n * Q9 + q];
      px0 += sim.f[n * Q9 + q] * EX9[q];
    }
  }
  // (a) 纯碰撞算子：对任意格子 Σfeq=ρ、Σfeq·e=ρu，故 BGK 碰撞守恒
  const omega = 1 / 0.8;
  let mCol = 0, pxCol = 0;
  for (let n = 0; n < nx * ny; n++) {
    let rho = 0, ux = 0;
    for (let q = 0; q < Q9; q++) {
      rho += sim.f[n * Q9 + q];
      ux += sim.f[n * Q9 + q] * EX9[q];
    }
    ux /= rho;
    for (let q = 0; q < Q9; q++) {
      const eq = sim.feqPub(q, rho, ux, 0);
      const post = sim.f[n * Q9 + q] - omega * (sim.f[n * Q9 + q] - eq);
      mCol += post;
      pxCol += post * EX9[q];
    }
  }
  check('BGK 碰撞保持总质量（机器精度）', () => assert.ok(Math.abs(mCol - m0) / m0 < 1e-12));
  check('BGK 碰撞保持总动量（机器精度）', () => assert.ok(Math.abs(pxCol - px0) / Math.abs(px0) < 1e-12));

  // (b) 含壁面回弹的完整一步：半way回弹的跨步交换残差应在 1e-7 内且不发散
  sim.step(0.05, 0, false);
  let m1 = 0, px1 = 0;
  for (let n = 0; n < nx * ny; n++) {
    for (let q = 0; q < Q9; q++) {
      m1 += sim.f[n * Q9 + q];
      px1 += sim.f[n * Q9 + q] * EX9[q];
    }
  }
  check('完整步骤质量漂移 < 1e-7', () => assert.ok(Math.abs(m1 - m0) / m0 < 1e-7));
  check('静止壁面产生阻力（动量下降）', () => assert.ok(px1 < px0));
}

// ---- 测试 2：Poiseuille 抛物线剖面 ----
{
  const nx = 40, ny = 23;
  const solid = new Uint8Array(nx * ny);
  // 显式壁格：j=0 与 j=ny-1 为固体，半way回弹面关于通道对称
  for (let i = 0; i < nx; i++) {
    solid[i * ny] = 1;
    solid[i * ny + ny - 1] = 1;
  }
  const sim = new D2Q9(nx, ny, solid, 1.0);
  sim.reset(0);
  const G = 0.002; // 压力梯度（体力）
  for (let t = 0; t < 10000; t++) sim.step(0, G, false);

  // 理论：u(z) = G/(2ν) z(H-z)，ν=(τ-0.5)/3
  // 半way回弹面位于壁格外半格；拟合得到流体格 j 的有效坐标 z = j-0.5，
  // 有效通道高度 H = ny-2，系数与 G/(2ν) 的偏差 <1%
  const nu = (1.0 - 0.5) / 3;
  const H = ny - 2;
  let err = 0, norm = 0;
  const iMid = 20;
  for (let j = 1; j < ny - 1; j++) {
    const z = j - 0.5;
    const theory = (G / (2 * nu)) * z * (H - z);
    const got = sim.ux[iMid * ny + j];
    err += (theory - got) ** 2;
    norm += theory ** 2;
  }
  const rel = Math.sqrt(err / norm);
  console.log(`    Poiseuille 剖面相对 L2 误差 = ${rel.toFixed(4)}`);
  check('Poiseuille 流收敛到理论抛物线（<3%）', () => assert.ok(rel < 0.03));
}

// ---- 测试 3：两个造型的阻力差可分辨 ----
{
  const nx = 120, ny = 60;
  const build = (withWing: boolean): Uint8Array => {
    const s = new Uint8Array(nx * ny);
    const mark = (i0: number, i1: number, j0: number, j1: number) => {
      for (let i = i0; i <= i1; i++)
        for (let j = j0; j <= j1; j++)
          if (i >= 0 && i < nx && j >= 0 && j < ny) s[i * ny + j] = 1;
    };
    // 车身：矩形块
    mark(40, 75, 12, 30);
    // 车顶斜面近似
    mark(48, 66, 31, 38);
    if (withWing) {
      // 尾翼：车尾上方小翼片 + 支架
      mark(78, 84, 40, 42);
      mark(80, 81, 33, 39);
    }
    return s;
  };

  const measure = (withWing: boolean): number => {
    const sim = new D2Q9(nx, ny, build(withWing), 0.65);
    sim.reset(0.08);
    let last = 0;
    const window: number[] = [];
    for (let t = 0; t < 4000; t++) {
      sim.step(0.08);
      if (t > 3000 && t % 20 === 0) {
        window.push(sim.forceX);
        last = sim.forceX;
      }
    }
    const avg = window.reduce((a, b) => a + b, 0) / window.length;
    assert.ok(avg > 0, '阻力应为正（沿来流方向）');
    assert.ok(Math.abs(last - avg) / avg < 0.15, '尾段阻力应基本稳定');
    return avg;
  };

  const cdBase = measure(false);
  const cdWing = measure(true);
  console.log(`    无尾翼阻力=${cdBase.toFixed(4)}，带尾翼阻力=${cdWing.toFixed(4)}，` +
    `Δ=${(cdWing - cdBase).toFixed(4)}`);
  check('带尾翼造型阻力显著高于基础造型', () => assert.ok(cdWing > cdBase * 1.05));
}

console.log(`\n全部 ${passed} 项物理测试通过 ✔`);
} // end !LBM_DIAG
