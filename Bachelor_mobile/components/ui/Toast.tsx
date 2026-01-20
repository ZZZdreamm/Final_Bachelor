import React, { useEffect } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withSpring,
  withDelay,
  runOnJS,
} from 'react-native-reanimated';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { Theme } from '@/constants/Theme';

export type ToastType = 'success' | 'error' | 'warning' | 'info';

interface ToastProps {
  message: string;
  type?: ToastType;
  duration?: number;
  visible: boolean;
  onHide: () => void;
}

export function Toast({
  message,
  type = 'info',
  duration = 3000,
  visible,
  onHide,
}: ToastProps) {
  const insets = useSafeAreaInsets();
  const translateY = useSharedValue(-200);

  const getBackgroundColor = (): string => {
    switch (type) {
      case 'success':
        return Theme.colors.accent.success;
      case 'error':
        return Theme.colors.accent.error;
      case 'warning':
        return Theme.colors.accent.warning;
      case 'info':
        return Theme.colors.accent.primary;
    }
  };

  const getIcon = (): keyof typeof Ionicons.glyphMap => {
    switch (type) {
      case 'success':
        return 'checkmark-circle';
      case 'error':
        return 'close-circle';
      case 'warning':
        return 'warning';
      case 'info':
        return 'information-circle';
    }
  };

  useEffect(() => {
    if (visible) {
      // Slide down
      translateY.value = withSpring(0, {
        damping: Theme.animation.easing.spring.damping,
        stiffness: Theme.animation.easing.spring.stiffness,
      });

      // Auto hide after duration
      const timer = setTimeout(() => {
        translateY.value = withSpring(-200, {
          damping: Theme.animation.easing.spring.damping,
          stiffness: Theme.animation.easing.spring.stiffness,
        }, () => {
          runOnJS(onHide)();
        });
      }, duration);

      return () => clearTimeout(timer);
    }
  }, [visible, duration, onHide]);

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [{ translateY: translateY.value }],
  }));

  if (!visible) return null;

  return (
    <Animated.View
      style={[
        styles.container,
        {
          top: insets.top + Theme.spacing.md,
          backgroundColor: getBackgroundColor(),
        },
        animatedStyle,
      ]}
    >
      <Ionicons name={getIcon()} size={24} color="#FFFFFF" style={styles.icon} />
      <Text style={styles.message} numberOfLines={3}>
        {message}
      </Text>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  container: {
    position: 'absolute',
    left: Theme.spacing.base,
    right: Theme.spacing.base,
    zIndex: 10000,
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: Theme.spacing.md,
    paddingHorizontal: Theme.spacing.base,
    borderRadius: Theme.radius.md,
    ...Theme.shadows.strong,
  },
  icon: {
    marginRight: Theme.spacing.sm,
  },
  message: {
    flex: 1,
    fontSize: Theme.typography.size.base,
    fontWeight: Theme.typography.weight.medium,
    color: '#FFFFFF',
  },
});
