import React, { ReactNode } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ViewStyle,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Theme } from '@/constants/Theme';
import { IconButton } from './IconButton';

interface TopBarProps {
  title: string;
  onBack?: () => void;
  rightElement?: ReactNode;
  variant?: 'solid' | 'glass' | 'transparent';
  style?: ViewStyle;
  noPaddingBottom?: boolean;
}

export function TopBar({
  title,
  onBack,
  rightElement,
  variant = 'glass',
  style,
  noPaddingBottom = false,
}: TopBarProps) {
  const insets = useSafeAreaInsets();

  const getBackgroundColor = (): string => {
    switch (variant) {
      case 'solid':
        return Theme.colors.background.secondary;
      case 'glass':
        return Theme.colors.surface.glass;
      case 'transparent':
        return 'transparent';
    }
  };

  const containerStyle: ViewStyle[] = [
    styles.container,
    {
      paddingTop: insets.top + Theme.spacing.md,
      paddingBottom: noPaddingBottom ? 0 : Theme.spacing.md,
      backgroundColor: getBackgroundColor(),
      borderBottomColor: variant === 'solid' ? Theme.colors.border.subtle : 'transparent',
      borderBottomWidth: variant === 'solid' ? 1 : 0,
    },
    variant === 'glass' && styles.glass,
    style,
  ];

  return (
    <View style={containerStyle}>
      <View style={styles.content}>
        <View style={styles.leftSection}>
          {onBack ? (
            <IconButton
              icon="arrow-back"
              onPress={onBack}
              size="medium"
              variant={variant === 'transparent' ? 'glass' : 'ghost'}
            />
          ) : (
            <View style={styles.placeholder} />
          )}
        </View>

        <View style={styles.centerSection}>
          <Text style={styles.title} numberOfLines={1}>
            {title}
          </Text>
        </View>

        <View style={styles.rightSection}>
          {rightElement || <View style={styles.placeholder} />}
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    zIndex: 10,
  },
  glass: {
    backdropFilter: 'blur(10px)',
  },
  content: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: Theme.spacing.base,
    height: Theme.layout.height.topBar,
  },
  leftSection: {
    width: 44,
    alignItems: 'flex-start',
  },
  centerSection: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: Theme.spacing.md,
    marginRight: 20
  },
  rightSection: {
    width: 54,
    alignItems: 'flex-end',
    marginRight: 8,
    position: 'absolute',
    right: 16,
  },
  title: {
    fontSize: Theme.typography.size.lg,
    fontWeight: Theme.typography.weight.semibold,
    color: Theme.colors.text.primary,
    textAlign: 'center',
  },
  placeholder: {
    width: 44,
    height: 44,
  },
});
