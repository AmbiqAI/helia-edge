import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir:'./tests', use:{baseURL:'http://127.0.0.1:8772/helia-edge/'},
  webServer:{command:'node node_modules/@ambiqai/helia-ui/scripts/serve-dist.mjs --port 8772 --base /helia-edge --dist dist', url:'http://127.0.0.1:8772/helia-edge/', reuseExistingServer:false},
});
