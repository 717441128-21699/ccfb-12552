/**
 * 虚拟风洞主程序：WebGPU 初始化、LBM 推进循环、体素化热启动、
 * 烟线/表面渲染调度、气动力监控与 A/B 方案对比。
 */
import { LbmSolver } from './gpu/LbmSolver';
import { SmokeRenderer, type SmokeSettings } from './gpu/SmokeRenderer';
import { requestWindTunnelDevice } from './gpu/device';
import { voxelizeMesh, type TriangleMesh, type VoxelGrid } from './lbm/voxelize';
import { buildCarMesh, parseOBJ, type CarParams } from './lbm/carMesh';
import {
  CdAverager,
  computeCd,
  diffSnapshots,
  frontalArea,
  type DesignSnapshot,
} from './lbm/aerodynamics';

interface GridDims { nx: number; ny: number; nz: number; }

function parseGrid(s: string): GridDims {
  const [nx, ny, nz] = s.split('x').map(Number);
  return { nx, ny, nz };
}

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing #${id}`);
  return el as T;
};

class WindTunnelApp {
  private solver!: LbmSolver;
  private renderer!: SmokeRenderer;
  private depthTex!: GPUTexture;
  private device!: GPUDevice;
  private context!: GPUCanvasContext;
  private format!: GPUTextureFormat;

  private grid: GridDims = parseGrid($<HTMLSelectElement>('grid-size').value);
  private voxel: VoxelGrid | null = null;
  private importedMesh: TriangleMesh | null = null;
  private refArea = 1;
  private running = true;
  private stepsPerFrame = 2;
  private simSteps = 0;

  private readonly cdAvg = new CdAverager();
  private snapshotA: DesignSnapshot | null = null;
  private snapshotB: DesignSnapshot | null = null;

  private fpsT0 = performance.now();
  private fpsSteps0 = 0;
  private forceT0 = performance.now();
  private speedT0 = 0;

  private settings: SmokeSettings = {
    nozzleCols: 16,
    nozzleRows: 10,
    trailAlpha: 0.7,
    speedScale: 1.5,
    showSmoke: true,
    showPressure: true,
  };

  async init(): Promise<void> {
    const status = $<HTMLDivElement>('status');
    if (!navigator.gpu) {
      status.textContent = '当前浏览器不支持 WebGPU，请用 Chrome/Edge 113+ 并启用硬件加速。';
      return;
    }
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: 'high-performance',
    });
    if (!adapter) {
      status.textContent = '未找到可用 GPU 适配器。';
      return;
    }
    this.device = await requestWindTunnelDevice(adapter);
    this.device.lost.then((info) => {
      $<HTMLDivElement>('status').textContent = `GPU 设备丢失：${info.message}`;
    });

    const canvas = $<HTMLCanvasElement>('gpu-canvas');
    this.context = canvas.getContext('webgpu')!;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({
      device: this.device,
      format: this.format,
      alphaMode: 'opaque',
    });

    this.buildSolver();
    this.bindUI();
    this.loop();
    status.textContent = `就绪：${this.grid.nx}×${this.grid.ny}×${this.grid.nz} 格，` +
      `${(this.grid.nx * this.grid.ny * this.grid.nz / 1e6).toFixed(2)}M 格子`;
  }

  private buildSolver(): void {
    const u0 = Number($<HTMLInputElement>('u0').value);
    const tau = Number($<HTMLInputElement>('tau').value);
    const smag = Number($<HTMLInputElement>('smag').value);

    this.solver?.destroy();
    this.solver = new LbmSolver(this.device, this.context, {
      ...this.grid, u0, tau, smagorinsky: smag,
    });

    const canvas = $<HTMLCanvasElement>('gpu-canvas');
    this.depthTex?.destroy();
    this.depthTex = this.device.createTexture({
      size: [canvas.clientWidth || 1280, canvas.clientHeight || 800],
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

    this.renderer = new SmokeRenderer(
      this.device, this.format,
      this.grid.nx, this.grid.ny, this.grid.nz,
      this.solver.getMacroBuffer(),
      this.solver.getFlagsBuffer(),
      this.solver.getPressureBuffer(),
    );
    this.renderer.camera.attach(canvas);
    this.cdAvg.reset();
    this.simSteps = 0;

    // 用当前造型参数构建并体素化
    this.rebuildVoxels(true);
  }

  private currentCarParams(): CarParams {
    return {
      mirrorOffset: Number($<HTMLInputElement>('mirror').value),
      wingAngleDeg: Number($<HTMLInputElement>('wing').value),
      wingHeight: Number($<HTMLInputElement>('wing-h').value),
    };
  }

  private rebuildVoxels(first: boolean): void {
    const mesh = this.importedMesh ?? buildCarMesh(this.currentCarParams());
    this.voxel = voxelizeMesh(mesh, this.grid);
    this.refArea = frontalArea(
      this.voxel.flags, this.grid.nx, this.grid.ny, this.grid.nz,
    );
    if (first) {
      this.solver.uploadInitialVoxels(this.voxel.flags);
    } else {
      void this.solver.revoxelize(this.voxel.flags).then((r) => {
        $<HTMLDivElement>('voxel-info').textContent =
          `固体 ${this.voxel!.solidCount.toLocaleString()} 格，` +
          `本次热更新 ${r.changed.toLocaleString()} 格，流场未重置`;
      });
    }
    this.renderer.setSurfaceCells(this.voxel.flags);
    $<HTMLDivElement>('voxel-info').textContent =
      `固体 ${this.voxel.solidCount.toLocaleString()} 格，迎风面积 ${this.refArea} 格²`;
  }

  private bindUI(): void {
    const bindRange = (id: string, suffix = '') => {
      const input = $<HTMLInputElement>(id);
      const out = $(id + '-val');
      const sync = () => { out.textContent = input.value + suffix; };
      input.addEventListener('input', () => {
        sync();
        if (id === 'u0' || id === 'tau' || id === 'smag') {
          this.solver.setRuntime({
            u0: Number($<HTMLInputElement>('u0').value),
            tau: Number($<HTMLInputElement>('tau').value),
            smagorinsky: Number($<HTMLInputElement>('smag').value),
          });
        }
      });
      sync();
    };
    bindRange('u0');
    bindRange('tau');
    bindRange('smag');
    bindRange('mirror');
    bindRange('wing', '°');
    bindRange('wing-h');
    bindRange('smoke-n');
    bindRange('trail');

    $<HTMLInputElement>('smoke-n').addEventListener('input', (e) => {
      const n = Number((e.target as HTMLInputElement).value);
      this.settings.nozzleCols = n;
      this.settings.nozzleRows = Math.max(4, Math.round(n / 2));
    });
    $<HTMLInputElement>('trail').addEventListener('input', (e) => {
      // 拖尾长度映射到透明度（固定 8 节历史，靠浓度表现长短观感）
      const t = Number((e.target as HTMLInputElement).value);
      this.settings.trailAlpha = 0.2 + (t / 64) * 0.6;
    });
    $<HTMLInputElement>('smoke-on').addEventListener('change', (e) => {
      this.settings.showSmoke = (e.target as HTMLInputElement).checked;
    });
    $<HTMLInputElement>('pressure-on').addEventListener('change', (e) => {
      this.settings.showPressure = (e.target as HTMLInputElement).checked;
    });

    $<HTMLButtonElement>('btn-play').addEventListener('click', (e) => {
      this.running = !this.running;
      (e.target as HTMLButtonElement).textContent = this.running ? '⏸ 暂停' : '▶ 继续';
    });
    $<HTMLButtonElement>('btn-reset').addEventListener('click', () => {
      this.solver.resetFlow();
      this.cdAvg.reset();
      this.simSteps = 0;
    });
    $<HTMLButtonElement>('btn-revoxel').addEventListener('click', () => {
      if (this.importedMesh) {
        $<HTMLDivElement>('status').textContent = '已导入外部 OBJ：滑块只对内置参数车生效，请移除 OBJ 后再调造型。';
      }
      this.rebuildVoxels(false);
    });

    $<HTMLSelectElement>('grid-size').addEventListener('change', (e) => {
      this.grid = parseGrid((e.target as HTMLSelectElement).value);
      this.buildSolver();
    });

    $<HTMLInputElement>('obj-file').addEventListener('change', async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;
      const text = await file.text();
      this.importedMesh = parseOBJ(text);
      $<HTMLDivElement>('status').textContent =
        `已导入 ${file.name}（${this.importedMesh.positions.length / 9 | 0} 三角形），重新体素化中…`;
      this.rebuildVoxels(false);
    });

    const snapshot = (which: 'A' | 'B') => {
      const avg = this.cdAvg.value();
      if (!avg || !this.voxel) return;
      const snap: DesignSnapshot = {
        name: which,
        cd: avg.cd,
        cdPressure: avg.cdPressure,
        cdViscous: avg.cdViscous,
        refArea: this.refArea,
        params: { ...this.currentCarParams() },
        solidCount: this.voxel.solidCount,
        atStep: this.simSteps,
      };
      if (which === 'A') this.snapshotA = snap;
      else this.snapshotB = snap;
      this.renderDiff();
    };
    $<HTMLButtonElement>('btn-snapshot-a').addEventListener('click', () => snapshot('A'));
    $<HTMLButtonElement>('btn-snapshot-b').addEventListener('click', () => snapshot('B'));

    $<HTMLButtonElement>('btn-view-side').addEventListener('click', () => this.renderer.camera.preset('side'));
    $<HTMLButtonElement>('btn-view-top').addEventListener('click', () => this.renderer.camera.preset('top'));
    $<HTMLButtonElement>('btn-view-3d').addEventListener('click', () => this.renderer.camera.preset('3d'));

    window.addEventListener('resize', () => this.resizeDepth());
  }

  private resizeDepth(): void {
    const canvas = $<HTMLCanvasElement>('gpu-canvas');
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (w === 0 || h === 0) return;
    this.depthTex.destroy();
    this.depthTex = this.device.createTexture({
      size: [w, h],
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }

  private renderDiff(): void {
    const el = $<HTMLDivElement>('ab-result');
    if (this.snapshotA && this.snapshotB) {
      el.textContent = diffSnapshots(this.snapshotA, this.snapshotB);
      el.className = this.snapshotB.cd <= this.snapshotA.cd ? 'good' : 'bad';
    } else {
      el.textContent = '保存两个方案后自动给出阻力差';
      el.className = 'muted';
    }
  }

  private loop = (): void => {
    requestAnimationFrame(this.loop);
    const canvas = $<HTMLCanvasElement>('gpu-canvas');
    if (canvas.clientWidth === 0) return;
    if (canvas.width !== canvas.clientWidth * devicePixelRatio ||
        canvas.height !== canvas.clientHeight * devicePixelRatio) {
      canvas.width = canvas.clientWidth * devicePixelRatio;
      canvas.height = canvas.clientHeight * devicePixelRatio;
      this.resizeDepth();
    }

    if (this.running) {
      // 力累加器跨帧累积，采样（updateMetrics 里）后才清零
      this.solver.step(this.stepsPerFrame);
      this.simSteps += this.stepsPerFrame;
      this.solver.updateSurfacePressure();
    }

    const u0 = Number($<HTMLInputElement>('u0').value);
    this.renderer.update(
      this.settings,
      u0,
      this.simSteps,
      this.depthTex.createView(),
      this.context.getCurrentTexture().createView(),
      canvas.width / canvas.height,
    );

    this.updateMetrics(u0);
  };

  private updateMetrics(u0: number): void {
    const now = performance.now();
    $<HTMLDivElement>('m-steps').textContent = String(this.simSteps);

    if (now - this.fpsT0 >= 1000) {
      const sps = ((this.simSteps - this.fpsSteps0) * 1000) / (now - this.fpsT0);
      $<HTMLDivElement>('m-sps').textContent = sps.toFixed(1);
      this.fpsT0 = now;
      this.fpsSteps0 = this.simSteps;
    }

    // 约每 0.6 秒读回一次气动力（mapAsync 异步，不阻塞渲染循环）
    if (this.running && now - this.forceT0 >= 600) {
      this.forceT0 = now;
      void this.solver.readForce().then((force) => {
        const r = computeCd(force, u0, this.refArea);
        const avg = this.cdAvg.push(r);
        $<HTMLDivElement>('m-cd').textContent = r.cd.toFixed(3);
        $<HTMLDivElement>('m-cd-avg').textContent = avg.cd.toFixed(3);
        $<HTMLDivElement>('m-pressure-ratio').textContent =
          `${(avg.pressureRatio * 100).toFixed(0)}%`;
      });
    }

    // 约每秒更新表面流速色带范围（分位数归一化，避免单色）
    if (this.running && now - this.speedT0 >= 1000) {
      this.speedT0 = now;
      const cells = this.renderer.getSurfaceCells();
      if (cells.length > 0) {
        void this.solver.readSurfaceSpeedRange(cells).then((range) => {
          this.renderer.setSpeedRange(range.min, range.max);
        });
      }
    }
  }
}

new WindTunnelApp().init().catch((err) => {
  console.error(err);
  $<HTMLDivElement>('status').textContent = `初始化失败：${err instanceof Error ? err.message : String(err)}`;
});
