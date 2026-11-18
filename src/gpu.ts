// WebGPU 设备初始化:申请大 storage buffer 绑定限额(分布函数单缓冲约 180MB)
export async function initGPU(neededBindingBytes: number): Promise<GPUDevice> {
  if (!navigator.gpu) {
    throw new Error('当前浏览器不支持 WebGPU。请使用 Chrome / Edge 113+ 并确认已启用硬件加速。');
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('找不到可用的 GPU 适配器。');

  const mb = (n: number) => (n / 1048576).toFixed(0) + ' MB';
  if (adapter.limits.maxStorageBufferBindingSize < neededBindingBytes) {
    throw new Error(
      `显卡的 maxStorageBufferBindingSize 为 ${mb(adapter.limits.maxStorageBufferBindingSize)},` +
        `小于求解器所需的 ${mb(neededBindingBytes)}。请更换显卡或减小网格规模。`,
    );
  }
  const requiredLimits: Record<string, number> = {
    maxStorageBufferBindingSize: neededBindingBytes,
  };
  if (adapter.limits.maxBufferSize >= neededBindingBytes) {
    requiredLimits.maxBufferSize = neededBindingBytes;
  }
  const device = await adapter.requestDevice({ requiredLimits });
  device.lost.then((info) => {
    if (info.reason !== 'destroyed') console.error('WebGPU device lost:', info.message);
  });
  return device;
}
