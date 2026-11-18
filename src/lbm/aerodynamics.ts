/**
 * 气动力后处理：从体素栅格估计迎风参考面积，把 lattice 动量交换力
 * 换算成阻力系数 Cd，并维护滑动平均与 A/B 方案对比。
 */
import { SOLID } from './constants';
import type { ForceSample } from '../gpu/LbmSolver';

/** 迎风投影面积（格子²）：沿 x 方向投影，每个 (y,z) 列只要出现固体就计 1 */
export function frontalArea(flags: Uint8Array, nx: number, ny: number, nz: number): number {
  let area = 0;
  for (let j = 0; j < ny; j++) {
    for (let k = 0; k < nz; k++) {
      let hit = false;
      for (let i = 0; i < nx; i++) {
        if (flags[i * (ny * nz) + j * nz + k] === SOLID) { hit = true; break; }
      }
      if (hit) area++;
    }
  }
  return area;
}

export interface CdResult {
  cd: number;
  cdPressure: number;
  cdViscous: number;
  pressureRatio: number;
  force: ForceSample;
}

/**
 * Cd = F / (0.5 · ρ₀ · u₀² · A)，lattice 单位下 ρ₀ ≈ 1。
 * 同一参考面积下两个方案的 Cd 差即阻力差；也同时输出 lattice 力差。
 */
export function computeCd(force: ForceSample, u0: number, refArea: number): CdResult {
  const dyn = 0.5 * u0 * u0 * Math.max(refArea, 1);
  return {
    cd: force.total / dyn,
    cdPressure: force.pressure / dyn,
    cdViscous: force.viscous / dyn,
    pressureRatio: force.total !== 0 ? force.pressure / force.total : 0,
    force,
  };
}

/** 指数滑动平均，window 越大越平滑 */
export class CdAverager {
  private avg: CdResult | null = null;
  private readonly alpha: number;

  constructor(window = 120) {
    this.alpha = 2 / (window + 1);
  }

  push(r: CdResult): CdResult {
    if (!this.avg) {
      this.avg = r;
      return r;
    }
    const a = this.alpha;
    this.avg = {
      cd: lerp(this.avg.cd, r.cd, a),
      cdPressure: lerp(this.avg.cdPressure, r.cdPressure, a),
      cdViscous: lerp(this.avg.cdViscous, r.cdViscous, a),
      pressureRatio: lerp(this.avg.pressureRatio, r.pressureRatio, a),
      force: {
        pressure: lerp(this.avg.force.pressure, r.force.pressure, a),
        viscous: lerp(this.avg.force.viscous, r.force.viscous, a),
        total: 0,
      },
    };
    this.avg.force.total = this.avg.force.pressure + this.avg.force.viscous;
    return this.avg;
  }

  reset(): void { this.avg = null; }
  value(): CdResult | null { return this.avg; }
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

export interface DesignSnapshot {
  name: string;
  cd: number;
  cdPressure: number;
  cdViscous: number;
  refArea: number;
  params: Record<string, number>;
  solidCount: number;
  atStep: number;
}

export function diffSnapshots(a: DesignSnapshot, b: DesignSnapshot): string {
  const dCd = b.cd - a.cd;
  const pct = a.cd !== 0 ? (dCd / Math.abs(a.cd)) * 100 : 0;
  const better = dCd < 0 ? 'B' : 'A';
  return [
    `方案 A：Cd = ${a.cd.toFixed(4)}（压差 ${(a.cdPressure).toFixed(4)} / 摩擦 ${a.cdViscous.toFixed(4)}）`,
    `方案 B：Cd = ${b.cd.toFixed(4)}（压差 ${(b.cdPressure).toFixed(4)} / 摩擦 ${b.cdViscous.toFixed(4)}）`,
    `ΔCd(B−A) = ${dCd >= 0 ? '+' : ''}${dCd.toFixed(4)}（${pct >= 0 ? '+' : ''}${pct.toFixed(1)}%）`,
    `参考面积 A = ${a.refArea} / ${b.refArea} 格²，阻力更小：方案 ${better}`,
  ].join('\n');
}
