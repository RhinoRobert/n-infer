// What n-infer sees on this machine, and what it would choose.
//   node examples/hardware.js
import { backend, hardware } from '../lib/index.js';

const info = hardware();
console.log(`build:  ${backend}`);
console.log(`cpu:    ${info.physicalCores} cores / ${info.logicalCores} threads`);
console.log(`simd:   ${info.cpuFeatures.join(' ')}`);
for (const device of info.devices) {
  const gib = (mib) => (mib / 1024).toFixed(1);
  console.log(
    `device: ${device.description.trim()} [${device.backend}] ${gib(device.freeMib)} of ${gib(device.totalMib)} GiB free`,
  );
}
