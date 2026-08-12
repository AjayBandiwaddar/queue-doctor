import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  base: '/queue-doctor/',
  plugins: [react()],
  build: {
    outDir: 'dist'
  }
});
