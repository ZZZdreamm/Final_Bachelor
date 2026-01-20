import React, { ReactNode } from 'react';
import {
  View,
  StyleSheet,
  ViewStyle,
  TouchableOpacity,
} from 'react-native';
import { Theme } from '@/constants/Theme';

type CardPadding = 'none' | 'small' | 'medium' | 'large';
type CardVariant = 'elevated' | 'outlined' | 'glass';

interface CardProps {
  children: ReactNode;
  padding?: CardPadding;
  variant?: CardVariant;
  onPress?: () => void;
  style?: ViewStyle;
}

export function Card({
  children,
  padding = 'medium',
  variant = 'elevated',
  onPress,
  style,
}: CardProps) {
  const getPadding = (): number => {
    switch (padding) {
      case 'none':
        return 0;
      case 'small':
        return Theme.spacing.md;
      case 'medium':
        return Theme.spacing.base;
      case 'large':
        return Theme.spacing.lg;
    }
  };

  const getBackgroundColor = (): string => {
    switch (variant) {
      case 'elevated':
        return Theme.colors.background.secondary;
      case 'outlined':
        return Theme.colors.background.primary;
      case 'glass':
        return Theme.colors.surface.glass;
    }
  };

  const containerStyle: ViewStyle[] = [
    styles.base,
    {
      backgroundColor: getBackgroundColor(),
      padding: getPadding(),
      borderRadius: Theme.radius.base,
    },
    variant === 'outlined' && {
      borderWidth: 1,
      borderColor: Theme.colors.border.default,
    },
    variant === 'elevated' && Theme.shadows.medium,
    style,
  ];

  if (onPress) {
    return (
      <TouchableOpacity
        style={containerStyle}
        onPress={onPress}
        activeOpacity={0.7}
      >
        {children}
      </TouchableOpacity>
    );
  }

  return <View style={containerStyle}>{children}</View>;
}

const styles = StyleSheet.create({
  base: {
    overflow: 'hidden',
  },
});
