/**
 * 按适配器能力申请 LBM 所需的提升限额（大存储缓冲绑定、足够的缓冲尺寸）。
 * 取"适配器上限"与"需求"的较小值，保证在不同显卡上都能成功申请。
 */
export interface WindTunnelLimits {
  maxStorageBufferBindingSize: number;
  maxBufferSize: number;
}

/** 2M 格 D3Q19 双份分布函数缓冲约需 310MB，留余量按 512MB 申请 */
const DESIRED_STORAGE_BINDING = 512 * 1024 * 1024;
const DESIRED_BUFFER = 512 * 1024 * 1024;

export async function requestWindTunnelDevice(
  adapter: GPUAdapter,
): Promise<GPUDevice> {
  const a = adapter.limits;
  const requiredLimits: Record<string, number> = {
    maxStorageBufferBindingSize: Math.min(a.maxStorageBufferBindingSize, DESIRED_STORAGE_BINDING),
    maxBufferSize: Math.min(a.maxBufferSize, DESIRED_BUFFER),
  };
  try {
    return await adapter.requestDevice({ requiredLimits });
  } catch {
    // 某些驱动对组合限额敏感，回退到默认限额再试一次
    return adapter.requestDevice();
  }
}

/** 判断给定网格的单份 f 缓冲是否在设备限额内 */
export function gridFits(device: GPUDevice, cellCount: number, q = 19): boolean {
  const fBytes = cellCount * q * 4;
  return fBytes <= device.limits.maxStorageBufferBindingSize &&
    fBytes <= device.limits.maxBufferSize;
}
