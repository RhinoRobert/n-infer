// Run inside a project that has n-infer installed from its packed tarballs (a "clean-room" install).
// Prints which binary was selected and proves it can run a model.
//
//   node check-install.mjs <model.gguf>
import { LocalModel, backend, hardware } from 'n-infer';

const [modelPath] = process.argv.slice(2);
const warnings = [];
process.on('warning', (warning) =>
  warnings.push(`${warning.name}: ${warning.message.split('\n')[0]}`),
);

const info = hardware();
const result = { backend, device: info.devices.map((d) => d.description.trim())[0], warnings };

if (modelPath) {
  const model = await new LocalModel({ modelPath, contextSize: 256 }).init();
  result.generated =
    (await model.generate('Once upon a time', { maxTokens: 8, temperature: 0 })).length > 0;
  model.close();
}
console.log(JSON.stringify(result));
