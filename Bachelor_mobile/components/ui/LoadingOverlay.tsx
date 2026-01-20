import React from 'react';
import {
  View,
  Text,
  ActivityIndicator,
  StyleSheet,
  ViewStyle,
  TouchableOpacity,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { Theme } from '@/constants/Theme';

interface LoadingOverlayProps {
  message?: string;
  subMessage?: string;
  variant?: 'default' | 'success' | 'processing';
  visible?: boolean;
  style?: ViewStyle;
  onCancel?: () => void;
  showCancelButton?: boolean;
}

export function LoadingOverlay({
  message = 'Loading...',
  subMessage,
  variant = 'default',
  visible = true,
  style,
  onCancel,
  showCancelButton = false,
}: LoadingOverlayProps) {
  if (!visible) return null;

  const getIndicatorColor = (): string => {
    switch (variant) {
      case 'success':
        return Theme.colors.accent.success;
      case 'processing':
        return Theme.colors.accent.primary;
      case 'default':
        return Theme.colors.text.primary;
    }
  };

  return (
    <View style={[styles.container, style]}>
      <View style={styles.content}>
        <ActivityIndicator size="large" color={getIndicatorColor()} />

        {message && <Text style={styles.message}>{message}</Text>}

        {subMessage && <Text style={styles.subMessage}>{subMessage}</Text>}

        {showCancelButton && onCancel && (
          <TouchableOpacity style={styles.cancelButton} onPress={onCancel}>
            <Ionicons name="close-circle" size={20} color={Theme.colors.text.secondary} />
            <Text style={styles.cancelText}>Cancel</Text>
          </TouchableOpacity>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: Theme.colors.surface.overlayStrong,
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 9999,
  },
  content: {
    alignItems: 'center',
    padding: Theme.spacing.xl,
    borderRadius: Theme.radius.base,
    backgroundColor: Theme.colors.background.elevated,
    minWidth: 200,
    ...Theme.shadows.strong,
  },
  message: {
    marginTop: Theme.spacing.base,
    fontSize: Theme.typography.size.lg,
    fontWeight: Theme.typography.weight.semibold,
    color: Theme.colors.text.primary,
    textAlign: 'center',
  },
  subMessage: {
    marginTop: Theme.spacing.sm,
    fontSize: Theme.typography.size.sm,
    color: Theme.colors.text.secondary,
    textAlign: 'center',
  },
  cancelButton: {
    marginTop: Theme.spacing.lg,
    paddingVertical: Theme.spacing.md,
    paddingHorizontal: Theme.spacing.lg,
    borderRadius: Theme.radius.md,
    backgroundColor: Theme.colors.surface.glass,
    flexDirection: 'row',
    alignItems: 'center',
    gap: Theme.spacing.sm,
  },
  cancelText: {
    fontSize: Theme.typography.size.base,
    fontWeight: Theme.typography.weight.medium,
    color: Theme.colors.text.secondary,
  },
});
