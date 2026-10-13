import fs from 'node:fs';
import path from 'node:path';
import { resizeUint8Bicubic } from '../../frontend/src/lib/bicubicResize.js';

const directory = path.resolve(process.argv[2]);
const fixtures = JSON.parse(fs.readFileSync(path.join(directory, 'fixtures/resize/examples.json'), 'utf8'));
const results = fixtures.map(item => {
    const input = fs.readFileSync(path.join(directory, 'fixtures/resize', item.input));
    const expected = fs.readFileSync(path.join(directory, 'fixtures/resize', item.output));
    const actual = resizeUint8Bicubic(input, item.width, item.height, item.output_width, item.output_height);
    let errors = 0, maximum = 0;
    for (let i = 0; i < actual.length; i++) {
        const delta = Math.abs(actual[i] - expected[i]);
        errors += Number(delta > 0); maximum = Math.max(maximum, delta);
    }
    return { id: item.id, channel_count: actual.length, different_channels: errors, maximum_pixel_error: maximum };
});
const report = { native_processor: 'Torchvision BICUBIC, antialias=True, uint8 CPU, PyTorch 2.8',
    cases: results.length, passed: results.every(r => r.different_channels === 0), results };
fs.writeFileSync(path.join(directory, 'validation_resize.json'), JSON.stringify(report, null, 2));
console.log(JSON.stringify(report));
if (!report.passed) process.exitCode = 1;
