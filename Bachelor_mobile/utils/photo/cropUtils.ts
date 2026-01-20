/**
 * Clamps image translation based on crop box and screen constraints
 * Ensures the image always covers the crop box and sticks to screen edges when larger than screen
 */
export const getClampedImageTranslation = (
  proposedX: number,
  proposedY: number,
  scale: number,
  viewportW: number,
  viewportH: number,
  cropX: number,
  cropY: number,
  cropW: number,
  cropH: number,
  baseRenderedW: number,
  baseRenderedH: number
) => {
  'worklet';
  const scaledImageW = baseRenderedW * scale;
  const scaledImageH = baseRenderedH * scale;

  const cx = viewportW / 2;
  const cy = viewportH / 2;

  // 1. Crop Constraints (Image must cover Crop Box)
  const maxTxCrop = cropX - cx + (scaledImageW / 2);
  const minTxCrop = (cropX + cropW) - cx - (scaledImageW / 2);
  const maxTyCrop = cropY - cy + (scaledImageH / 2);
  const minTyCrop = (cropY + cropH) - cy - (scaledImageH / 2);

  // 2. Screen Constraints (Image must stick to Screen sides if wide enough)
  const maxTxScreen = (scaledImageW / 2) - cx;
  const minTxScreen = viewportW - cx - (scaledImageW / 2);
  const maxTyScreen = (scaledImageH / 2) - cy;
  const minTyScreen = viewportH - cy - (scaledImageH / 2);

  let finalMaxTx = maxTxCrop;
  let finalMinTx = minTxCrop;

  if (scaledImageW >= viewportW) {
     finalMaxTx = Math.min(finalMaxTx, maxTxScreen);
     finalMinTx = Math.max(finalMinTx, minTxScreen);
  }

  let finalMaxTy = maxTyCrop;
  let finalMinTy = minTyCrop;

  if (scaledImageH >= viewportH) {
    finalMaxTy = Math.min(finalMaxTy, maxTyScreen);
    finalMinTy = Math.max(finalMinTy, minTyScreen);
  }

  return {
    translateX: Math.min(Math.max(proposedX, finalMinTx), finalMaxTx),
    translateY: Math.min(Math.max(proposedY, finalMinTy), finalMaxTy),
  };
};

/**
 * Gets valid bounds for crop box based on current image transform
 */
export const getValidCropBounds = (
  currentScale: number,
  tx: number,
  ty: number,
  viewportW: number,
  viewportH: number,
  baseRenderedW: number,
  baseRenderedH: number
) => {
  'worklet';
  const vw = viewportW;
  const vh = viewportH;
  const imgW = baseRenderedW * currentScale;
  const imgH = baseRenderedH * currentScale;
  const cx = (vw / 2) + tx;
  const cy = (vh / 2) + ty;
  const imgMinX = cx - (imgW / 2);
  const imgMaxX = cx + (imgW / 2);
  const imgMinY = cy - (imgH / 2);
  const imgMaxY = cy + (imgH / 2);
  return {
    minX: Math.max(imgMinX, 0),
    maxX: Math.min(imgMaxX, vw),
    minY: Math.max(imgMinY, 0),
    maxY: Math.min(imgMaxY, vh)
  };
};
