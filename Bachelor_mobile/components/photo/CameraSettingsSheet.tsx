import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { Theme } from '@/constants/Theme';
import type { GridType } from './CameraGridOverlay';

interface CameraSettingsSheetProps {
  gridType: GridType;
  onGridTypeChange: (type: GridType) => void;
}

export function CameraSettingsSheet({
  gridType,
  onGridTypeChange,
}: CameraSettingsSheetProps) {
  const gridOptions: { label: string; value: GridType }[] = [
    { label: 'Off', value: 'none' },
    { label: 'Rule of Thirds', value: 'thirds' },
    { label: 'Golden Ratio', value: 'golden' },
    { label: 'Square', value: 'square' },
  ];

  return (
    <View style={styles.container}>
      <Text style={styles.sectionTitle}>Camera Settings</Text>

      {/* Grid Overlay */}
      <View style={styles.section}>
        <Text style={styles.label}>Grid Overlay</Text>
        <View style={styles.optionsGrid}>
          {gridOptions.map((option) => (
            <TouchableOpacity
              key={option.value}
              style={[
                styles.option,
                gridType === option.value && styles.optionActive,
              ]}
              onPress={() => onGridTypeChange(option.value)}
            >
              <Text
                style={[
                  styles.optionText,
                  gridType === option.value && styles.optionTextActive,
                ]}
              >
                {option.label}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    paddingVertical: Theme.spacing.base,
  },
  sectionTitle: {
    fontSize: Theme.typography.size.xl,
    fontWeight: Theme.typography.weight.semibold,
    color: Theme.colors.text.primary,
    marginBottom: Theme.spacing.lg,
  },
  section: {
    marginBottom: Theme.spacing.lg,
  },
  label: {
    fontSize: Theme.typography.size.base,
    fontWeight: Theme.typography.weight.medium,
    color: Theme.colors.text.secondary,
    marginBottom: Theme.spacing.md,
  },
  optionsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: Theme.spacing.sm,
  },
  option: {
    paddingVertical: Theme.spacing.sm,
    paddingHorizontal: Theme.spacing.base,
    borderRadius: Theme.radius.md,
    backgroundColor: Theme.colors.surface.glass,
    borderWidth: 1,
    borderColor: Theme.colors.border.default,
    minWidth: '30%',
    alignItems: 'center',
  },
  optionActive: {
    backgroundColor: Theme.colors.accent.primary,
    borderColor: Theme.colors.accent.primary,
  },
  optionText: {
    fontSize: Theme.typography.size.sm,
    color: Theme.colors.text.primary,
    fontWeight: Theme.typography.weight.medium,
  },
  optionTextActive: {
    color: '#FFFFFF',
    fontWeight: Theme.typography.weight.semibold,
  },
});
