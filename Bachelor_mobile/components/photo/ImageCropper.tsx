import React, { useState, useEffect, useCallback } from 'react';
import { StyleSheet, View, Alert, Image, Dimensions } from 'react-native';
import * as ImageManipulator from 'expo-image-manipulator';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import Animated, { 
  useSharedValue, 
  useAnimatedStyle, 
} from 'react-native-reanimated';
import { Gesture, GestureDetector } from 'react-native-gesture-handler';

import { Theme } from '@/constants/Theme';
import { TopBar, Button, LoadingOverlay, BottomSheet, IconButton } from '@/components/ui';
import type { SnapPoint } from '@/components/ui';
import { CameraGridOverlay, type GridType } from './CameraGridOverlay';
import { CameraSettingsSheet } from './CameraSettingsSheet';
import { getClampedImageTranslation } from '@/utils/photo/cropUtils';

const AnimatedImage = Animated.createAnimatedComponent(Image);

const { height: SCREEN_HEIGHT } = Dimensions.get('window');

const BOTTOM_SHEET_COLLAPSED_HEIGHT = SCREEN_HEIGHT * 0.225;
const INITIAL_SCALE = 1.0;
const MAX_SCALE = 5.0;
const MIN_CROP_SIZE = 80;

interface ImageCropperProps {
  originalUri: string;
  imageWidth: number | null;
  imageHeight: number | null;
  onRetake: () => void;
  onCropComplete: (croppedUri: string) => void;
  onImageLoaded?: () => void;
}

export default function ImageCropper({ originalUri, imageWidth, imageHeight, onRetake, onCropComplete, onImageLoaded }: ImageCropperProps) {
  const insets = useSafeAreaInsets();
  const [isProcessing, setIsProcessing] = useState(false);
  const [isImageLoading, setIsImageLoading] = useState(false); // Start false until layout is ready
  const [gridType, setGridType] = useState<GridType>('square');
  const [sheetSnapPoint, setSheetSnapPoint] = useState<SnapPoint>('collapsed');
  
  // Layout state: We do NOT render the image/crop interface until we know this size
  const [viewport, setViewport] = useState({ width: 0, height: 0 });

  const topBarHeight = insets.top + Theme.spacing.md + Theme.layout.height.topBar;
  const [imageDims, setImageDims] = useState<{imageWidth: number, imageHeight: number} | null>(null);

  // -- Shared Values --
  const scale = useSharedValue(INITIAL_SCALE);
  const translateX = useSharedValue(0);
  const translateY = useSharedValue(0);
  const savedScale = useSharedValue(INITIAL_SCALE);
  const savedTranslateX = useSharedValue(0);
  const savedTranslateY = useSharedValue(0);

  const viewportW = useSharedValue(0);
  const viewportH = useSharedValue(0);
  
  const baseRenderedW = useSharedValue(0);
  const baseRenderedH = useSharedValue(0);

  const cropX = useSharedValue(0); 
  const cropY = useSharedValue(0);
  const cropWidth = useSharedValue(0);
  const cropHeight = useSharedValue(0);
  
  const startCropX = useSharedValue(0);
  const startCropY = useSharedValue(0);
  const startCropW = useSharedValue(0);
  const startCropH = useSharedValue(0);

  useEffect(() => {
    if (!originalUri || viewport.width === 0 || viewport.height === 0) return;
    setIsImageLoading(true);

    Image.getSize(
      originalUri,
      (originalW, originalH) => {
        const width = imageWidth ?? originalW;
        const height = imageHeight ?? originalH;
        const imageAspect = width / height;
        const screenAspect = viewport.width / viewport.height;
        

        let renderedW_calc: number;
        let renderedH_calc: number;

        if (imageAspect > screenAspect) {
          renderedW_calc = viewport.width;
          renderedH_calc = viewport.width / imageAspect;
        } else {
          renderedH_calc = viewport.height;
          renderedW_calc = viewport.height * imageAspect;
        }
        
        
        setImageDims({ imageWidth: width, imageHeight: height });

        scale.value = 1.0;
        translateX.value = 0;
        translateY.value = 0;
        savedScale.value = 1.0;
        savedTranslateX.value = 0;
        savedTranslateY.value = 0;

        viewportW.value = viewport.width;
        viewportH.value = viewport.height;
        baseRenderedW.value = renderedW_calc;
        baseRenderedH.value = renderedH_calc;

        const initialCropW = renderedW_calc;
        const initialCropH = renderedH_calc; 
        
        cropWidth.value = initialCropW;
        cropHeight.value = initialCropH;
        cropX.value = (viewport.width - renderedW_calc) / 2;
        cropY.value = (viewport.height - renderedH_calc) / 2;


        setIsImageLoading(false);
        if (onImageLoaded) onImageLoaded();
      },
      () => {
        Alert.alert('Load Error', 'Failed to retrieve image dimensions.');
        setIsImageLoading(false);
        onRetake();
      }
    );
  }, [originalUri, viewport.width, viewport.height, imageWidth, imageHeight]);

  const imagePanGesture = Gesture.Pan()
    .onUpdate((event) => {
      'worklet';
      const proposedX = savedTranslateX.value + event.translationX;
      const proposedY = savedTranslateY.value + event.translationY;
      const clamped = getClampedImageTranslation(proposedX, proposedY, scale.value, viewportW.value, viewportH.value, cropX.value, cropY.value, cropWidth.value, cropHeight.value, baseRenderedW.value, baseRenderedH.value);
      translateX.value = clamped.translateX;
      translateY.value = clamped.translateY;
    })
    .onEnd(() => { 'worklet'; savedTranslateX.value = translateX.value; savedTranslateY.value = translateY.value; });

  const imagePinchGesture = Gesture.Pinch()
    .onUpdate((event) => {
      'worklet';
      const proposedScale = savedScale.value * event.scale;
      const minScaleW = cropWidth.value / baseRenderedW.value;
      const minScaleH = cropHeight.value / baseRenderedH.value;
      const newScale = Math.min(Math.max(proposedScale, Math.max(minScaleW, minScaleH, 1.0)), MAX_SCALE);
      scale.value = newScale;
      const clamped = getClampedImageTranslation(translateX.value, translateY.value, newScale, viewportW.value, viewportH.value, cropX.value, cropY.value, cropWidth.value, cropHeight.value, baseRenderedW.value, baseRenderedH.value);
      translateX.value = clamped.translateX;
      translateY.value = clamped.translateY;
    })
    .onEnd(() => { 'worklet'; savedScale.value = scale.value; savedTranslateX.value = translateX.value; savedTranslateY.value = translateY.value; });

  const imageGestures = Gesture.Simultaneous(imagePanGesture, imagePinchGesture);

  const getValidCropBounds = (currentScale: number, tx: number, ty: number) => {
    'worklet';
    const vw = viewportW.value;
    const vh = viewportH.value;
    const imgW = baseRenderedW.value * currentScale;
    const imgH = baseRenderedH.value * currentScale;
    const cx = (vw / 2) + tx;
    const cy = (vh / 2) + ty;
    const imgMinX = cx - (imgW / 2);
    const imgMaxX = cx + (imgW / 2);
    const imgMinY = cy - (imgH / 2);
    const imgMaxY = cy + (imgH / 2);
    return { minX: Math.max(imgMinX, 0), maxX: Math.min(imgMaxX, vw), minY: Math.max(imgMinY, 0), maxY: Math.min(imgMaxY, vh) };
  };

  const setupCropStartValues = () => { 'worklet'; startCropX.value = cropX.value; startCropY.value = cropY.value; startCropW.value = cropWidth.value; startCropH.value = cropHeight.value; };

  const CenterMoveGesture = Gesture.Pan().onStart(setupCropStartValues).onUpdate((e) => {
      'worklet';
      const bounds = getValidCropBounds(scale.value, translateX.value, translateY.value);
      let newX = startCropX.value + e.translationX;
      let newY = startCropY.value + e.translationY;
      newX = Math.min(Math.max(newX, bounds.minX), bounds.maxX - startCropW.value);
      newY = Math.min(Math.max(newY, bounds.minY), bounds.maxY - startCropH.value);
      cropX.value = newX; cropY.value = newY;
  });

  const TopLeftResize = Gesture.Pan().onStart(setupCropStartValues).onUpdate((e) => {
     'worklet';
     const bounds = getValidCropBounds(scale.value, translateX.value, translateY.value);
     let newX = Math.max(startCropX.value + e.translationX, bounds.minX);
     let newY = Math.max(startCropY.value + e.translationY, bounds.minY);
     const r = startCropX.value + startCropW.value; const b = startCropY.value + startCropH.value;
     let newW = r - newX; let newH = b - newY;
     if (newW < MIN_CROP_SIZE) { newW = MIN_CROP_SIZE; newX = r - MIN_CROP_SIZE; }
     if (newH < MIN_CROP_SIZE) { newH = MIN_CROP_SIZE; newY = b - MIN_CROP_SIZE; }
     cropX.value = newX; cropY.value = newY; cropWidth.value = newW; cropHeight.value = newH;
  });
  
  const TopRightResize = Gesture.Pan().onStart(setupCropStartValues).onUpdate((e) => {
    'worklet';
    const bounds = getValidCropBounds(scale.value, translateX.value, translateY.value);
    let newY = Math.max(startCropY.value + e.translationY, bounds.minY);
    let newW = Math.min(startCropW.value + e.translationX, bounds.maxX - startCropX.value);
    const b = startCropY.value + startCropH.value;
    let newH = b - newY;
    if (newH < MIN_CROP_SIZE) { newH = MIN_CROP_SIZE; newY = b - MIN_CROP_SIZE; }
    if (newW < MIN_CROP_SIZE) newW = MIN_CROP_SIZE;
    cropY.value = newY; cropWidth.value = newW; cropHeight.value = newH;
  });
  const BottomLeftResize = Gesture.Pan().onStart(setupCropStartValues).onUpdate((e) => {
    'worklet';
    const bounds = getValidCropBounds(scale.value, translateX.value, translateY.value);
    let newX = Math.max(startCropX.value + e.translationX, bounds.minX);
    let newH = Math.min(startCropH.value + e.translationY, bounds.maxY - startCropY.value);
    const r = startCropX.value + startCropW.value;
    let newW = r - newX;
    if (newW < MIN_CROP_SIZE) { newW = MIN_CROP_SIZE; newX = r - MIN_CROP_SIZE; }
    if (newH < MIN_CROP_SIZE) newH = MIN_CROP_SIZE;
    cropX.value = newX; cropWidth.value = newW; cropHeight.value = newH;
  });
  const BottomRightResize = Gesture.Pan().onStart(setupCropStartValues).onUpdate((e) => {
    'worklet';
    const bounds = getValidCropBounds(scale.value, translateX.value, translateY.value);
    let newW = Math.min(startCropW.value + e.translationX, bounds.maxX - startCropX.value);
    let newH = Math.min(startCropH.value + e.translationY, bounds.maxY - startCropY.value);
    if (newW < MIN_CROP_SIZE) newW = MIN_CROP_SIZE;
    if (newH < MIN_CROP_SIZE) newH = MIN_CROP_SIZE;
    cropWidth.value = newW; cropHeight.value = newH;
  });

  const imageAnimatedStyle = useAnimatedStyle(() => ({
    width: baseRenderedW.value,
    height: baseRenderedH.value,
    transform: [
      { translateX: translateX.value },
      { translateY: translateY.value },
      { scale: scale.value },
    ],
  }));

  const cropFrameStyle = useAnimatedStyle(() => ({
    width: cropWidth.value, height: cropHeight.value, left: cropX.value, top: cropY.value,
  }));
  const maskStyle = useAnimatedStyle(() => ({
    width: cropWidth.value, height: cropHeight.value, left: cropX.value, top: cropY.value,
    borderLeftWidth: 2000, borderRightWidth: 2000, borderTopWidth: 2000, borderBottomWidth: 2000,
    borderColor: 'rgba(0,0,0,0.6)', position: 'absolute', transform: [{ translateX: -2000 }, { translateY: -2000 }]
  }));

  const handleCrop = useCallback(async () => {
    if (!imageDims) return;
    try {
      setIsProcessing(true);
      const finalScale = scale.value;
      const finalTx = translateX.value;
      const finalTy = translateY.value;
      
      const imgVisualX = (viewport.width - (baseRenderedW.value * finalScale)) / 2 + finalTx;
      const imgVisualY = (viewport.height - (baseRenderedH.value * finalScale)) / 2 + finalTy;

      const relativeX = cropX.value - imgVisualX;
      const relativeY = cropY.value - imgVisualY;

      // Map back to original image
      const ratio = imageDims.imageWidth / (baseRenderedW.value * finalScale);
      const cropData = {
        originX: Math.round(Math.max(0, relativeX * ratio)),
        originY: Math.round(Math.max(0, relativeY * ratio)),
        width: Math.round(cropWidth.value * ratio),
        height: Math.round(cropHeight.value * ratio),
      };

      const manipResult = await ImageManipulator.manipulateAsync(
        originalUri,
        [{ crop: cropData }],
        { compress: 0.9, format: ImageManipulator.SaveFormat.JPEG }
      );
      onCropComplete(manipResult.uri);
    } catch (error) {
      Alert.alert('Error', 'Failed to crop');
    } finally {
      setIsProcessing(false);
    }
  }, [imageDims, originalUri, onCropComplete, viewport]);

  return (
    <View style={styles.container}>
      <View style={styles.topBarContainer}>
        <TopBar 
          title="Crop Image" variant="glass" onBack={onRetake} noPaddingBottom  
          rightElement={<IconButton icon="settings-outline" onPress={() => setSheetSnapPoint(sheetSnapPoint === 'collapsed' ? 'half' : 'collapsed')} size="medium" variant="ghost" />} 
        />
      </View>

      <View 
        style={[styles.viewportContainer, { marginTop: topBarHeight, marginBottom: BOTTOM_SHEET_COLLAPSED_HEIGHT }]}
        onLayout={(e) => {
            const { width, height } = e.nativeEvent.layout;
            if (Math.abs(width - viewport.width) > 1 || Math.abs(height - viewport.height) > 1) {
                setViewport({ width, height });
            }
        }}
      >
        {viewport.width > 0 && (
            <GestureDetector gesture={imageGestures}>
                <Animated.View style={{ flex: 1, width: '100%', height: '100%' }}>
                    <View style={styles.imageViewport}>
                        <AnimatedImage
                            source={{ uri: originalUri }}
                            resizeMode="stretch" 
                            style={imageAnimatedStyle} 
                        />
                    </View>

                    <Animated.View style={maskStyle} pointerEvents="none" />

                    <Animated.View style={[styles.cropFrame, cropFrameStyle]}>
                        <CameraGridOverlay gridType={gridType} width={100} height={100} style={{width: '100%', height: '100%'}} />
                        <GestureDetector gesture={CenterMoveGesture}><View style={styles.centerMoveHitBox} /></GestureDetector>
                        <GestureDetector gesture={TopLeftResize}><View style={[styles.cornerHitBox, { top: -20, left: -20 }]}><View style={[styles.cornerHandle, styles.topLeft]} /></View></GestureDetector>
                        <GestureDetector gesture={TopRightResize}><View style={[styles.cornerHitBox, { top: -20, right: -20 }]}><View style={[styles.cornerHandle, styles.topRight]} /></View></GestureDetector>
                        <GestureDetector gesture={BottomLeftResize}><View style={[styles.cornerHitBox, { bottom: -20, left: -20 }]}><View style={[styles.cornerHandle, styles.bottomLeft]} /></View></GestureDetector>
                        <GestureDetector gesture={BottomRightResize}><View style={[styles.cornerHitBox, { bottom: -20, right: -20 }]}><View style={[styles.cornerHandle, styles.bottomRight]} /></View></GestureDetector>
                    </Animated.View>

                </Animated.View>
            </GestureDetector>
        )}
      </View>

      <LoadingOverlay visible={isImageLoading || isProcessing} message={isProcessing ? 'Cropping...' : 'Loading...'} variant={isProcessing ? 'processing' : 'default'} />

      <BottomSheet snapPoint={sheetSnapPoint} onSnapPointChange={setSheetSnapPoint} showBackdrop={false}>
        {sheetSnapPoint === 'half' ? (
          <CameraSettingsSheet gridType={gridType} onGridTypeChange={setGridType} />
        ) : (
          <View style={styles.bottomSheetContent}>
            <View style={styles.buttonsRow}>
              <Button title="Retake" onPress={onRetake} variant="secondary" icon="close-circle-outline" />
              <Button title="Confirm" onPress={handleCrop} variant="primary" icon="checkmark-circle-outline" loading={isProcessing} />
            </View>
          </View>
        )}
      </BottomSheet>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  topBarContainer: { position: 'absolute', top: 0, left: 0, right: 0, zIndex: 100 },
  viewportContainer: {
    flex: 1, 
    backgroundColor: '#000',
    overflow: 'hidden',
  },
  imageViewport: {
    flex: 1,
    width: '100%',
    height: '100%',
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
  },
  cropFrame: { position: 'absolute', borderWidth: 1, borderColor: 'rgba(255,255,255,0.8)' },
  centerMoveHitBox: { width: '100%', height: '100%', position: 'absolute', backgroundColor: 'transparent', zIndex: 10 },
  cornerHitBox: { position: 'absolute', width: 60, height: 60, justifyContent: 'center', alignItems: 'center', zIndex: 200 },
  cornerHandle: { width: 20, height: 20, borderColor: '#FFFFFF', borderWidth: 3 },
  topLeft: { borderBottomWidth: 0, borderRightWidth: 0, marginTop: 20, marginLeft: 20 }, 
  topRight: { borderBottomWidth: 0, borderLeftWidth: 0, marginTop: 20, marginRight: 20 },
  bottomLeft: { borderTopWidth: 0, borderRightWidth: 0, marginBottom: 20, marginLeft: 20 },
  bottomRight: { borderTopWidth: 0, borderLeftWidth: 0, marginBottom: 20, marginRight: 20 },
  bottomSheetContent: { paddingTop: Theme.spacing.md, paddingBottom: Theme.spacing.lg },
  buttonsRow: { flexDirection: 'row', gap: Theme.spacing.sm, justifyContent: 'space-between' },
});