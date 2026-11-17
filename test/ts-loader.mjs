// 仅用于 Node 测试：把无扩展名的相对导入解析到 .ts 文件。
// Vite/浏览器构建走 bundler 解析，不经过此钩子。
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

export async function resolve(specifier, context, nextResolve) {
  if (specifier.startsWith('.') && !/\.[a-zA-Z0-9]+$/.test(specifier) && context.parentURL) {
    const candidates = [specifier + '.ts', specifier + '/index.ts'];
    for (const candidate of candidates) {
      const url = new URL(candidate, context.parentURL);
      if (url.protocol === 'file:' && existsSync(fileURLToPath(url))) {
        return nextResolve(url.href, context);
      }
    }
  }
  return nextResolve(specifier, context);
}
