import React, { ReactNode, useRef, useEffect } from 'react';
import {
  View,
  StyleSheet,
  Dimensions,
  TouchableWithoutFeedback,
  ViewStyle,
} from 'react-native';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withSpring,
  interpolate,
} from 'react-native-reanimated';
import { Gesture, GestureDetector } from 'react-native-gesture-handler';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Theme } from '@/constants/Theme';

const { height: SCREEN_HEIGHT } = Dimensions.get('window');

export type SnapPoint = 'collapsed' | 'half' | 'expanded';

interface BottomSheetProps {
  children: ReactNode;
  snapPoint?: SnapPoint;
  onSnapPointChange?: (snapPoint: SnapPoint) => void;
  showBackdrop?: boolean;
  enablePanGesture?: boolean;
  style?: ViewStyle;
}

const SNAP_POINTS: Record<SnapPoint, number> = {
  collapsed: 0.225,  // Base level 
  half: 0.5,        // Middle position
  expanded: 0.85,    // Full expansion
};

const SPRING_CONFIG = {
  damping: Theme.animation.easing.spring.damping,
  stiffness: Theme.animation.easing.spring.stiffness,
  mass: Theme.animation.easing.spring.mass,
};

export function BottomSheet({
  children,
  snapPoint = 'collapsed',
  onSnapPointChange,
  showBackdrop = true,
  enablePanGesture = true,
  style,
}: BottomSheetProps) {
  const insets = useSafeAreaInsets();

  // Initialize with 'collapsed' to avoid recreating shared value
  const translateY = useSharedValue(SCREEN_HEIGHT * (1 - SNAP_POINTS['collapsed']));
  const startY = useSharedValue(0);
  const prevSnapPoint = useRef<SnapPoint | null>(null);

  const snapToPoint = (point: SnapPoint, callback?: () => void) => {
    'worklet';
    const targetY = SCREEN_HEIGHT * (1 - SNAP_POINTS[point]);
    translateY.value = withSpring(targetY, SPRING_CONFIG, () => {});
  };

  // Handle external snap point changes (e.g., from props) and initial position
  useEffect(() => {
    if (prevSnapPoint.current !== snapPoint) {
      prevSnapPoint.current = snapPoint;
      const targetY = SCREEN_HEIGHT * (1 - SNAP_POINTS[snapPoint]);
      translateY.value = withSpring(targetY, SPRING_CONFIG);
    }
  }, [snapPoint, translateY]);

  const findNearestSnapPoint = (currentY: number): SnapPoint => {
    'worklet';
    const currentRatio = 1 - currentY / SCREEN_HEIGHT;

    let nearestPoint: SnapPoint = 'collapsed';
    let minDiff = Infinity;

    (Object.keys(SNAP_POINTS) as SnapPoint[]).forEach((point) => {
      const diff = Math.abs(SNAP_POINTS[point] - currentRatio);
      if (diff < minDiff) {
        minDiff = diff;
        nearestPoint = point;
      }
    });

    return nearestPoint;
  };

  const panGesture = Gesture.Pan()
    .enabled(enablePanGesture)
    .onStart(() => {
      startY.value = translateY.value;
    })
    .onUpdate((event) => {
      const newY = startY.value + event.translationY;
      // Constrain within bounds - cannot go below collapsed or above expanded
      const minY = SCREEN_HEIGHT * (1 - SNAP_POINTS.expanded);
      const maxY = SCREEN_HEIGHT * (1 - SNAP_POINTS.collapsed);

      translateY.value = Math.max(minY, Math.min(maxY, newY));
    })
    .onEnd(() => {
      // Snap to nearest point
      const nearestPoint = findNearestSnapPoint(translateY.value);
      snapToPoint(nearestPoint, () => {
        if (onSnapPointChange) {
          onSnapPointChange(nearestPoint);
        }
      });
    });

  const sheetAnimatedStyle = useAnimatedStyle(() => {
    // Calculate current ratio (0 = collapsed, 1 = expanded)
    const currentRatio = 1 - translateY.value / SCREEN_HEIGHT;
    
    // Interpolate border radius: 0 when collapsed (ratio < 0.3), full radius when expanded (ratio > 0.4)
    const borderRadius = interpolate(
      currentRatio,
      [0, 0.3, 0.4, 1],
      [0, 0, Theme.radius.lg, Theme.radius.lg],
      'clamp'
    );
    
    return {
      transform: [{ translateY: translateY.value }],
      borderTopLeftRadius: borderRadius,
      borderTopRightRadius: borderRadius,
    };
  });

  const backdropAnimatedStyle = useAnimatedStyle(() => {
    const opacity = 1 - translateY.value / SCREEN_HEIGHT;
    return {
      opacity: opacity * 0.5,
    };
  });

  const handleBackdropPress = () => {
    // Snap to collapsed when backdrop is pressed
    snapToPoint('collapsed', () => {
      if (onSnapPointChange) {
        onSnapPointChange('collapsed');
      }
    });
  };

  return (
    <>
      {/* Backdrop */}
      {showBackdrop && (
        <Animated.View style={[styles.backdrop, backdropAnimatedStyle]} pointerEvents="auto">
          <TouchableWithoutFeedback onPress={handleBackdropPress}>
            <View style={StyleSheet.absoluteFill} />
          </TouchableWithoutFeedback>
        </Animated.View>
      )}

      {/* Bottom Sheet */}
      <GestureDetector gesture={panGesture}>
        <Animated.View
          style={[
            styles.sheet,
            {
              height: SCREEN_HEIGHT,
              paddingBottom: insets.bottom,
            },
            sheetAnimatedStyle,
            style,
          ]}
        >
          {/* Handle */}
          {enablePanGesture && (
            <View style={styles.handleContainer}>
              <View style={styles.handle} />
            </View>
          )}

          {/* Content */}
          <View style={styles.content}>{children}</View>
        </Animated.View>
      </GestureDetector>
    </>
  );
}

const styles = StyleSheet.create({
  backdrop: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: '#000',
    zIndex: 998,
  },
  sheet: {
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: Theme.colors.background.elevated,
    // Border radius is now animated based on snap point
    borderTopLeftRadius: 0,
    borderTopRightRadius: 0,
    zIndex: 999,
    ...Theme.shadows.strong,
  },
  handleContainer: {
    alignItems: 'center',
    paddingVertical: Theme.spacing.md,
  },
  handle: {
    width: 40,
    height: 4,
    borderRadius: 2,
    backgroundColor: Theme.colors.text.tertiary,
  },
  content: {
    flex: 1,
    paddingHorizontal: Theme.spacing.base,
  },
});
