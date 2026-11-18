// 主程序:初始化 GPU / 求解器 / 渲染,驱动主循环与交互
import { initGPU } from './gpu';
import { LBM } from './lbm';
import { SmokeSystem } from './particles';
import { SceneRenderer } from './render';
import { OrbitCamera } from './camera';
import { UI, fatal } from './ui';
import {
  buildProceduralCar,
  loadMeshFile,
  normalizeMesh,
  meshToGrid,
  type MeshData,
} from './carmesh';
import { voxelize } from './voxelize';

// 网格候选:默认 256×96×96 ≈ 236 万格子;显存/限额不够时自动降档
const GRID_CANDIDATES: [number, number, number][] = [
  [256, 96, 96],
  [224, 96, 96],
  [256, 88, 96],
];
const CELLS_PER_M = 25; // 4.4 m 车长 → 110 格子
const CAR_X = 100; // 车中心在网格中的 x 位置
const CAR_Y = 1.5; // 车底(车轮着地点)在网格中的 y 位置

async function main() {
  const canvas = document.getElementById('gpu') as HTMLCanvasElement;

  // 依次尝试网格规模,直到设备限额满足
  let device: GPUDevice | null = null;
  let grid: [number, number, number] = GRID_CANDIDATES[0];
  let lastErr: unknown = null;
  for (const g of GRID_CANDIDATES) {
    const need = g[0] * g[1] * g[2] * 19 * 4;
    try {
      device = await initGPU(need);
      grid = g;
      break;
    } catch (e) {
      lastErr = e;
    }
  }
  if (!device) throw lastErr;
  const [nx, ny, nz] = grid;

  const context = canvas.getContext('webgpu')!;
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });

  const solver = new LBM(device, nx, ny, nz);
  const camBuf = device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const camera = new OrbitCamera(canvas);
  camera.target = [CAR_X, 18, nz / 2];

  const renderer = new SceneRenderer(device, nx, ny, nz, camBuf, solver.macroBuf, solver.flagsBuf, format);
  const smoke = new SmokeSystem(device, nx, ny, nz, solver.macroBuf, solver.flagsBuf, camBuf, format);

  // ---------------- 车型状态 ----------------
  let carArea = 1; // 正面投影面积(格子²)
  let useProcedural = true;
  let rawLoaded: MeshData | null = null; // 导入的原始模型
  let flipX = false;
  let revoxelizeTimer: ReturnType<typeof setTimeout> | null = null;

  const applyMesh = (meshMeters: MeshData, flip: boolean) => {
    const gridMesh = meshToGrid(meshMeters, CELLS_PER_M, CAR_X, CAR_Y, nz / 2, flip);
    const { mask, area } = voxelize(gridMesh, nx, ny, nz);
    carArea = Math.max(1, area);
    solver.updateObstacles(mask); // 流场接着算,不重置
    renderer.setMesh(gridMesh);
  };

  const applyLoaded = () => {
    if (!rawLoaded) return;
    applyMesh(normalizeMesh(rawLoaded, flipX), false);
  };

  const rebuildProcedural = () => {
    applyMesh(buildProceduralCar(ui.getCarParams()), true);
  };

  const scheduleRevoxelize = () => {
    if (revoxelizeTimer) clearTimeout(revoxelizeTimer);
    revoxelizeTimer = setTimeout(() => {
      revoxelizeTimer = null;
      if (useProcedural) rebuildProcedural();
    }, 150); // 拖动滑杆时防抖
  };

  // ---------------- UI ----------------
  let paused = false;
  let smokeOn = true;
  let sliceOn = true;
  let sliceFrac = 0.5;
  let cdA: number | null = null;
  let cdB: number | null = null;
  let cdSmooth: number | null = null;
  let clSmooth: number | null = null;

  const ui = new UI({
    onWind: (u0) => solver.setFlow(u0, solver.nu),
    onViscosity: (nu) => solver.setFlow(solver.u0, nu),
    onSmoke: (on) => (smokeOn = on),
    onSlice: (on) => (sliceOn = on),
    onSlicePos: (f) => (sliceFrac = f),
    onPause: (p) => (paused = p),
    onReset: () => {
      solver.resetFlow();
      cdSmooth = clSmooth = null;
      ui.toast('流场已重置为均匀来流');
    },
    onCarParam: () => scheduleRevoxelize(),
    onSaveA: () => {
      if (cdSmooth === null) return ui.toast('Cd 还没有读数,等流场发展一下');
      cdA = cdSmooth;
      ui.setCompare(cdA, cdB);
      ui.toast(`方案 A:Cd = ${cdA.toFixed(3)}`);
    },
    onSaveB: () => {
      if (cdSmooth === null) return ui.toast('Cd 还没有读数,等流场发展一下');
      cdB = cdSmooth;
      ui.setCompare(cdA, cdB);
      ui.toast(`方案 B:Cd = ${cdB.toFixed(3)}`);
    },
    onLoadFile: async (f) => {
      try {
        ui.toast(`正在解析 ${f.name}…`);
        rawLoaded = await loadMeshFile(f);
        useProcedural = false;
        ui.setProceduralEnabled(false);
        ui.toast(`体素化中(${(rawLoaded.indices.length / 3).toLocaleString()} 三角形)…`);
        await new Promise((r) => setTimeout(r, 30)); // 让 toast 先渲染
        applyLoaded();
        ui.toast('模型已加载,流场继续推进');
      } catch (e) {
        ui.toast(`加载失败:${(e as Error).message}`, 5000);
      }
    },
    onFlip: () => {
      flipX = !flipX;
      if (!useProcedural) applyLoaded();
      ui.toast('模型已掉头');
    },
    onRestoreDefault: () => {
      useProcedural = true;
      rawLoaded = null;
      ui.setProceduralEnabled(true);
      rebuildProcedural();
      ui.toast('已恢复内置参数化车型');
    },
  });

  // 初始车型
  rebuildProcedural();
  smoke.prime();

  // ---------------- 深度缓冲与尺寸 ----------------
  let depthTex: GPUTexture | null = null;
  const fitCanvas = () => {
    const w = Math.max(1, Math.floor(canvas.clientWidth * devicePixelRatio));
    const h = Math.max(1, Math.floor(canvas.clientHeight * devicePixelRatio));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
      depthTex?.destroy();
      depthTex = device.createTexture({
        size: [w, h],
        format: 'depth24plus',
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
    }
  };

  // ---------------- 主循环 ----------------
  let stepsPerFrame = 4;
  let emaFrame = 16.7;
  let lastT = performance.now();
  let statT = 0;
  let stepsPerSec = 0;

  const carLenCells = 4.4 * CELLS_PER_M;

  const frame = () => {
    const now = performance.now();
    const dtMs = now - lastT;
    lastT = now;
    emaFrame = emaFrame * 0.9 + dtMs * 0.1;

    // 自适应步数:帧耗时富余(顶着 vsync)就加步,紧张就减步
    if (!paused) {
      if (emaFrame > 24 && stepsPerFrame > 1) stepsPerFrame--;
      else if (emaFrame < 19 && stepsPerFrame < 64) stepsPerFrame++;
    }
    stepsPerSec = paused ? 0 : (stepsPerFrame * 1000) / emaFrame;

    fitCanvas();
    const aspect = canvas.width / canvas.height;
    device.queue.writeBuffer(camBuf, 0, camera.viewProj(aspect));

    const encoder = device.createCommandEncoder();
    if (!paused) {
      solver.encodeSteps(encoder, stepsPerFrame);
      smoke.encodeAdvect(encoder, stepsPerFrame);
      renderer.encodeCarColors(encoder, solver.u0);
    }
    const view = context.getCurrentTexture().createView();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view,
          clearValue: { r: 0.045, g: 0.055, b: 0.075, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
      depthStencilAttachment: {
        view: depthTex!.createView(),
        depthClearValue: 1,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    });
    renderer.encodeGround(pass);
    if (sliceOn) {
      renderer.setSlice(sliceFrac, solver.u0);
      renderer.encodeSlice(pass);
    }
    renderer.encodeCar(pass);
    if (smokeOn) smoke.encodeRender(pass);
    pass.end();
    device.queue.submit([encoder.finish()]);

    // 读回上一帧的力 → Cd / Cl(EMA 平滑)
    if (!paused) {
      solver.readForce().then((f) => {
        if (!f) return;
        const q = 0.5 * solver.u0 * solver.u0 * carArea;
        const cd = f.fx / q;
        const cl = f.fy / q;
        cdSmooth = cdSmooth === null ? cd : cdSmooth * 0.92 + cd * 0.08;
        clSmooth = clSmooth === null ? cl : clSmooth * 0.92 + cl * 0.08;
        ui.setCd(cdSmooth, clSmooth);
      });
    }

    // 状态栏
    if (now - statT > 300) {
      statT = now;
      const nu = solver.nu;
      const re = (solver.u0 * carLenCells) / nu;
      ui.setStats({
        grid: `${nx}×${ny}×${nz} = ${((nx * ny * nz) / 1e6).toFixed(2)}M`,
        stepsPerSec: stepsPerSec.toFixed(0),
        re: re.toExponential(2),
        area: `${carArea} 格² (≈${(carArea / (CELLS_PER_M * CELLS_PER_M)).toFixed(2)} m²)`,
      });
    }

    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

main().catch((e) => {
  console.error(e);
  fatal(`初始化失败\n\n${(e as Error).message ?? e}`);
});
