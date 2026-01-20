import React from 'react';
import { View, StyleSheet, Dimensions } from 'react-native';

export type GridType = 'none' | 'thirds' | 'golden' | 'square';

interface CameraGridOverlayProps {
  gridType: GridType;
  width?: number;
  height?: number;
}

export function CameraGridOverlay({ gridType, width, height }: CameraGridOverlayProps) {
  if (gridType === 'none') return null;

  const { width: screenWidth, height: screenHeight } = Dimensions.get('window');
  const containerWidth = width ?? screenWidth;
  const containerHeight = height ?? screenHeight;

  const renderThirdsGrid = () => (
    <>
      {/* Vertical lines */}
      <View style={[styles.line, styles.verticalLine, { left: containerWidth * 0.333 }]} />
      <View style={[styles.line, styles.verticalLine, { left: containerWidth * 0.666 }]} />

      {/* Horizontal lines */}
      <View style={[styles.line, styles.horizontalLine, { top: containerHeight * 0.333 }]} />
      <View style={[styles.line, styles.horizontalLine, { top: containerHeight * 0.666 }]} />
    </>
  );

  const renderGoldenRatioGrid = () => {
    const goldenRatio = 0.618;
    return (
      <>
        {/* Vertical lines */}
        <View style={[styles.line, styles.verticalLine, { left: containerWidth * goldenRatio }]} />
        <View
          style={[styles.line, styles.verticalLine, { left: containerWidth * (1 - goldenRatio) }]}
        />

        {/* Horizontal lines */}
        <View style={[styles.line, styles.horizontalLine, { top: containerHeight * goldenRatio }]} />
        <View
          style={[styles.line, styles.horizontalLine, { top: containerHeight * (1 - goldenRatio) }]}
        />
      </>
    );
  };

  const renderSquareGrid = () => {
    const gridSize = 4;
    const cellWidth = containerWidth / gridSize;
    const cellHeight = containerHeight / gridSize;

    const lines = [];

    // Vertical lines
    for (let i = 1; i < gridSize; i++) {
      lines.push(
        <View
          key={`v-${i}`}
          style={[styles.line, styles.verticalLine, { left: cellWidth * i }]}
        />
      );
    }

    // Horizontal lines
    for (let i = 1; i < gridSize; i++) {
      lines.push(
        <View
          key={`h-${i}`}
          style={[styles.line, styles.horizontalLine, { top: cellHeight * i }]}
        />
      );
    }

    return <>{lines}</>;
  };

  return (
    <View style={styles.container} pointerEvents="none">
      {gridType === 'thirds' && renderThirdsGrid()}
      {gridType === 'golden' && renderGoldenRatioGrid()}
      {gridType === 'square' && renderSquareGrid()}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    ...StyleSheet.absoluteFillObject,
    zIndex: 5,
  },
  line: {
    position: 'absolute',
    backgroundColor: 'rgba(255, 255, 255, 0.3)',
  },
  verticalLine: {
    width: 1,
    height: '100%',
  },
  horizontalLine: {
    height: 1,
    width: '100%',
  },
});
