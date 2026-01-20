import React from 'react';
import {
  TouchableOpacity,
  Text,
  ActivityIndicator,
  StyleSheet,
  ViewStyle,
  TextStyle,
  View,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { Theme } from '@/constants/Theme';

type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger';
type ButtonSize = 'small' | 'medium' | 'large';
type IconPosition = 'left' | 'right';

interface ButtonProps {
  title: string;
  onPress: () => void;
  variant?: ButtonVariant;
  size?: ButtonSize;
  disabled?: boolean;
  loading?: boolean;
  icon?: keyof typeof Ionicons.glyphMap;
  iconPosition?: IconPosition;
  fullWidth?: boolean;
  style?: ViewStyle;
}

export function Button({
  title,
  onPress,
  variant = 'primary',
  size = 'medium',
  disabled = false,
  loading = false,
  icon,
  iconPosition = 'left',
  fullWidth = false,
  style,
}: ButtonProps) {
  const isDisabled = disabled || loading;

  const getBackgroundColor = (): string => {
    if (isDisabled) {
      switch (variant) {
        case 'primary':
          return `${Theme.colors.accent.primary}66`; // 40% opacity
        case 'secondary':
          return `${Theme.colors.background.elevated}66`;
        case 'danger':
          return `${Theme.colors.accent.error}66`;
        case 'ghost':
          return 'transparent';
      }
    }

    switch (variant) {
      case 'primary':
        return Theme.colors.accent.primary;
      case 'secondary':
        return Theme.colors.background.elevated;
      case 'danger':
        return Theme.colors.accent.error;
      case 'ghost':
        return 'transparent';
    }
  };

  const getTextColor = (): string => {
    if (isDisabled) {
      return Theme.colors.text.tertiary;
    }

    switch (variant) {
      case 'ghost':
      case 'secondary':
        return Theme.colors.text.primary;
      default:
        return '#FFFFFF';
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

  const containerStyle: ViewStyle[] = [
    styles.base,
    {
      backgroundColor: getBackgroundColor(),
      height: Theme.layout.height.button[size],
      borderRadius: Theme.radius.md,
      paddingHorizontal: size === 'small' ? Theme.spacing.md : Theme.spacing.lg,
      opacity: isDisabled ? Theme.opacity.disabled : Theme.opacity.full,
    },
    // Ghost variant has no border - just transparent background
    fullWidth && { width: '100%' },
    !isDisabled && Theme.shadows.medium,
    style,
  ];

  const textStyle: TextStyle[] = [
    styles.text,
    {
      color: getTextColor(),
      fontSize: size === 'small' ? Theme.typography.size.sm : Theme.typography.size.base,
      fontWeight: Theme.typography.weight.semibold,
    },
  ];

  const renderIcon = () => {
    if (!icon) return null;

    return (
      <Ionicons
        name={icon}
        size={getIconSize()}
        color={getTextColor()}
        style={iconPosition === 'left' ? styles.iconLeft : styles.iconRight}
      />
    );
  };

  return (
    <TouchableOpacity
      style={containerStyle}
      onPress={onPress}
      disabled={isDisabled}
      activeOpacity={0.7}
    >
      {iconPosition === 'left' && renderIcon()}

      {loading ? (
        <ActivityIndicator size="small" color={getTextColor()} />
      ) : (
        <Text style={textStyle}>{title}</Text>
      )}

      {iconPosition === 'right' && !loading && renderIcon()}
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  base: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: Theme.spacing.sm,
  },
  text: {
    textAlign: 'center',
  },
  iconLeft: {
    marginRight: Theme.spacing.xs,
  },
  iconRight: {
    marginLeft: Theme.spacing.xs,
  },
});
