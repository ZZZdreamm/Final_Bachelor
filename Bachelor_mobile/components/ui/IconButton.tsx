import React from 'react';
import {
  TouchableOpacity,
  StyleSheet,
  ViewStyle,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { Theme } from '@/constants/Theme';

type IconButtonSize = 'small' | 'medium' | 'large';
type IconButtonVariant = 'glass' | 'solid' | 'ghost';

interface IconButtonProps {
  icon: keyof typeof Ionicons.glyphMap;
  onPress: () => void;
  size?: IconButtonSize;
  variant?: IconButtonVariant;
  disabled?: boolean;
  style?: ViewStyle;
  color?: string;
}

export function IconButton({
  icon,
  onPress,
  size = 'medium',
  variant = 'glass',
  disabled = false,
  style,
  color,
}: IconButtonProps) {
  const getDimensions = (): number => {
    switch (size) {
      case 'small':
        return 36;
      case 'medium':
        return 44;
      case 'large':
        return 56;
    }
  };

  const getIconSize = (): number => {
    switch (size) {
      case 'small':
        return Theme.layout.iconSize.sm;
      case 'medium':
        return Theme.layout.iconSize.base;
      case 'large':
        return Theme.layout.iconSize.lg;
    }
  };

  const getBackgroundColor = (): string => {
    if (disabled) {
      return Theme.colors.surface.glass;
    }

    switch (variant) {
      case 'glass':
        return Theme.colors.surface.glass;
      case 'solid':
        return Theme.colors.background.elevated;
      case 'ghost':
        return 'transparent';
    }
  };

  const iconColor = color || (disabled ? Theme.colors.text.tertiary : Theme.colors.text.primary);

  const buttonSize = getDimensions();

  const containerStyle: ViewStyle[] = [
    styles.base,
    {
      width: buttonSize,
      height: buttonSize,
      borderRadius: buttonSize / 2,
      backgroundColor: getBackgroundColor(),
      opacity: disabled ? Theme.opacity.disabled : Theme.opacity.full,
    },
    variant === 'ghost' && {
      borderWidth: 1,
      borderColor: Theme.colors.border.default,
    },
    !disabled && variant !== 'ghost' && Theme.shadows.subtle,
    style,
  ];

  return (
    <TouchableOpacity
      style={containerStyle}
      onPress={onPress}
      disabled={disabled}
      activeOpacity={0.7}
    >
      <Ionicons name={icon} size={getIconSize()} color={iconColor} />
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  base: {
    justifyContent: 'center',
    alignItems: 'center',
  },
});
