import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  ViewStyle,
  TextStyle,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { Theme } from '@/constants/Theme';

type BadgeVariant = 'default' | 'success' | 'warning' | 'error' | 'info';
type BadgeSize = 'small' | 'medium' | 'large';

interface BadgeProps {
  text: string;
  icon?: keyof typeof Ionicons.glyphMap;
  variant?: BadgeVariant;
  size?: BadgeSize;
  style?: ViewStyle;
  maxWidth?: number;
}

export function Badge({
  text,
  icon,
  variant = 'default',
  size = 'medium',
  style,
  maxWidth,
}: BadgeProps) {
  const getBackgroundColor = (): string => {
    switch (variant) {
      case 'success':
        return `${Theme.colors.accent.success}33`; // 20% opacity
      case 'warning':
        return `${Theme.colors.accent.warning}33`;
      case 'error':
        return `${Theme.colors.accent.error}33`;
      case 'info':
        return `${Theme.colors.accent.primary}33`;
      case 'default':
        return Theme.colors.surface.glass;
    }
  };

  const getTextColor = (): string => {
    switch (variant) {
      case 'success':
        return Theme.colors.accent.success;
      case 'warning':
        return Theme.colors.accent.warning;
      case 'error':
        return Theme.colors.accent.error;
      case 'info':
        return Theme.colors.accent.primary;
      case 'default':
        return Theme.colors.text.primary;
    }
  };

  const getPadding = (): { vertical: number; horizontal: number } => {
    switch (size) {
      case 'small':
        return { vertical: Theme.spacing.xs, horizontal: Theme.spacing.sm };
      case 'medium':
        return { vertical: Theme.spacing.sm, horizontal: Theme.spacing.md };
      case 'large':
        return { vertical: Theme.spacing.md, horizontal: Theme.spacing.base };
    }
  };

  const getFontSize = (): number => {
    switch (size) {
      case 'small':
        return Theme.typography.size.xs;
      case 'medium':
        return Theme.typography.size.sm;
      case 'large':
        return Theme.typography.size.base;
    }
  };

  const getIconSize = (): number => {
    switch (size) {
      case 'small':
        return Theme.layout.iconSize.xs;
      case 'medium':
        return Theme.layout.iconSize.sm;
      case 'large':
        return Theme.layout.iconSize.base;
    }
  };

  const padding = getPadding();

  const containerStyle: ViewStyle[] = [
    styles.container,
    {
      backgroundColor: getBackgroundColor(),
      paddingVertical: padding.vertical,
      paddingHorizontal: padding.horizontal,
      borderRadius: Theme.radius.md,
      maxWidth: maxWidth,
    },
    style,
  ];

  const textStyle: TextStyle = {
    color: getTextColor(),
    fontSize: getFontSize(),
    fontWeight: Theme.typography.weight.medium,
  };

  return (
    <View style={containerStyle}>
      {icon && (
        <Ionicons
          name={icon}
          size={getIconSize()}
          color={getTextColor()}
          style={styles.icon}
        />
      )}
      <Text style={textStyle} numberOfLines={2}>{text}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    alignSelf: 'flex-start',
  },
  icon: {
    marginRight: Theme.spacing.xs,
  },
});
