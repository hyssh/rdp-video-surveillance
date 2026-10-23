import { defineConfig, loadEnv, type PluginOption } from 'vite';
import react from '@vitejs/plugin-react';
import { mockMiddleware } from './mock/middleware';

const API_PATHS = ['/stats', '/videos', '/frames', '/jobs', '/v2'];

export default defineConfig(({ mode }) => {
  const env = { ...loadEnv(mode, process.cwd(), ''), ...process.env };
  const mock = env.VITE_MOCK === '1';
  const plugins: PluginOption[] = [react()];
  if (mock) {
    plugins.push({
      name: 'mock-api',
      configureServer(server) {
        server.middlewares.use(mockMiddleware());
      },
    });
  }
  const proxy = mock
    ? undefined
    : Object.fromEntries(API_PATHS.map((p) => [p, { target: 'http://localhost:8082', changeOrigin: true }]));
  return {
    base: '/ui/',
    plugins,
    build: { outDir: '../static/app', emptyOutDir: true, chunkSizeWarningLimit: 900 },
    server: { port: 5173, proxy },
  };
});
