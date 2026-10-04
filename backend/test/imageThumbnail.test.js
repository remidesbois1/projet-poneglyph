const assert = require('node:assert/strict');
const test = require('node:test');
const sharp = require('sharp');
const {
  createImageThumbnail,
  getThumbnailWidth,
  createPageMiniature,
} = require('../src/utils/imageThumbnail');

test('thumbnail widths are bounded to safe server-side limits', () => {
  assert.equal(getThumbnailWidth('640'), 640);
  assert.equal(getThumbnailWidth('4'), 96);
  assert.equal(getThumbnailWidth('99999'), 1600);
  assert.equal(getThumbnailWidth('invalid'), 640);
});

test('thumbnails use high-quality server-side resizing without enlargement', async () => {
  const source = await sharp({
    create: {
      width: 1600,
      height: 1200,
      channels: 3,
      background: { r: 30, g: 60, b: 90 },
    },
  }).png().toBuffer();

  const thumbnail = await createImageThumbnail(source, { width: 640 });
  const metadata = await sharp(thumbnail).metadata();

  assert.ok(['avif', 'heif'].includes(metadata.format));
  assert.equal(metadata.width, 640);
  assert.equal(metadata.height, 480);
});

test('public miniatures have fixed bounds for portrait, spreads and tall pages without enlargement', async () => {
  for (const [width, height, expectedWidth, expectedHeight] of [
    [1200, 1800, 85, 128],
    [2400, 1600, 192, 128],
    [6000, 1000, 192, 32],
    [1000, 6000, 21, 128],
    [40, 60, 40, 60],
  ]) {
    const source = await sharp({ create: { width, height, channels: 3, background: '#aabbcc' } }).png().toBuffer();
    // Options supplied by a caller cannot turn this public miniature into a readable full-size image.
    const miniature = await createPageMiniature(source, { width: 6000, height: 6000 });
    const metadata = await sharp(miniature).metadata();
    assert.ok(['avif', 'heif'].includes(metadata.format));
    assert.equal(metadata.width, expectedWidth);
    assert.equal(metadata.height, expectedHeight);
  }
});

test('public miniatures respect orientation before applying their bounds', async () => {
  const source = await sharp({ create: { width: 1800, height: 1200, channels: 3, background: '#445566' } })
    .withMetadata({ orientation: 6 }).jpeg().toBuffer();
  const metadata = await sharp(await createPageMiniature(source)).metadata();
  assert.equal(metadata.width, 85);
  assert.equal(metadata.height, 128);
  assert.equal(metadata.orientation, undefined);
});
