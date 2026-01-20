import { Link } from 'expo-router';
import React from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  Platform,
  BackHandler,
  ScrollView,
} from 'react-native';
import Ionicons from '@expo/vector-icons/Ionicons';
import { StatusBar } from 'expo-status-bar';
import Animated, { FadeInDown, FadeInUp } from 'react-native-reanimated';
import { Theme } from '@/constants/Theme';
import { Card } from '@/components/ui';

export default function HomeScreen() {
  const handleExitApp = () => {
    if (Platform.OS === 'android') {
      BackHandler.exitApp();
    } else if (Platform.OS === 'web') {
      window.close();
    }
    // iOS doesn't support programmatic app exit
  };

  return (
    <View style={styles.container}>
      <StatusBar style="light" />

      <ScrollView
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        <Animated.View
          entering={FadeInUp.duration(600).delay(100)}
          style={styles.heroSection}
        >
          <View style={styles.iconContainer}>
            <Ionicons name="business" size={64} color={Theme.colors.accent.primary} />
          </View>

          <Text style={styles.title}>Story Build</Text>
          <Text style={styles.subtitle}>
            Capture buildings and discover their 3D models
          </Text>
        </Animated.View>

        <Animated.View
          entering={FadeInDown.duration(600).delay(300)}
          style={styles.actionSection}
        >
          <Link href="/(tabs)/photo" asChild>
            <TouchableOpacity activeOpacity={0.9}>
              <Card variant="elevated" padding="large" style={styles.mainCard}>
                <View style={styles.cardIcon}>
                  <Ionicons name="camera" size={48} color={Theme.colors.accent.primary} />
                </View>
                <Text style={styles.cardTitle}>Take Photo</Text>
                <Text style={styles.cardDescription}>
                  Point your camera at a building to get started
                </Text>
                <View style={styles.cardArrow}>
                  <Ionicons
                    name="arrow-forward"
                    size={24}
                    color={Theme.colors.accent.primary}
                  />
                </View>
              </Card>
            </TouchableOpacity>
          </Link>
        </Animated.View>

        <Animated.View
          entering={FadeInDown.duration(600).delay(500)}
          style={styles.guideSection}
        >
          <Text style={styles.guideTitle}>How it works</Text>

          <View style={styles.stepsList}>
            <View style={styles.stepItem}>
              <View style={styles.stepNumber}>
                <Text style={styles.stepNumberText}>1</Text>
              </View>
              <View style={styles.stepContent}>
                <Text style={styles.stepTitle}>Capture</Text>
                <Text style={styles.stepDescription}>
                  Take a photo of any building
                </Text>
              </View>
            </View>

            <View style={styles.stepItem}>
              <View style={styles.stepNumber}>
                <Text style={styles.stepNumberText}>2</Text>
              </View>
              <View style={styles.stepContent}>
                <Text style={styles.stepTitle}>Crop & Send</Text>
                <Text style={styles.stepDescription}>
                  Adjust your photo and send for analysis
                </Text>
              </View>
            </View>

            <View style={styles.stepItem}>
              <View style={styles.stepNumber}>
                <Text style={styles.stepNumberText}>3</Text>
              </View>
              <View style={styles.stepContent}>
                <Text style={styles.stepTitle}>View in 3D</Text>
                <Text style={styles.stepDescription}>
                  Explore the building in 3D
                </Text>
              </View>
            </View>
          </View>
        </Animated.View>

        <View style={styles.footer}>
          <Text style={styles.footerText}>Version 1.0.0</Text>

          {Platform.OS !== 'ios' && (
            <TouchableOpacity style={styles.exitButton} onPress={handleExitApp}>
              <Ionicons name="exit-outline" size={18} color={Theme.colors.accent.error} />
              <Text style={styles.exitButtonText}>Exit App</Text>
            </TouchableOpacity>
          )}
        </View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Theme.colors.background.primary,
  },
  scrollContent: {
    paddingVertical: Theme.spacing['2xl'],
    paddingHorizontal: Theme.spacing.lg,
  },
  heroSection: {
    alignItems: 'center',
    marginBottom: Theme.spacing['2xl'],
  },
  iconContainer: {
    width: 120,
    height: 120,
    borderRadius: 60,
    backgroundColor: Theme.colors.surface.glass,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: Theme.spacing.lg,
    ...Theme.shadows.medium,
  },
  title: {
    fontSize: Theme.typography.size['5xl'],
    fontWeight: Theme.typography.weight.bold,
    color: Theme.colors.text.primary,
    marginBottom: Theme.spacing.md,
    textAlign: 'center',
  },
  subtitle: {
    fontSize: Theme.typography.size.lg,
    color: Theme.colors.text.secondary,
    textAlign: 'center',
    lineHeight: Theme.typography.lineHeight.relaxed * Theme.typography.size.lg,
    paddingHorizontal: Theme.spacing.lg,
  },
  actionSection: {
    marginBottom: Theme.spacing['2xl'],
  },
  mainCard: {
    alignItems: 'center',
    paddingVertical: Theme.spacing.xl,
  },
  cardIcon: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: `${Theme.colors.accent.primary}1A`,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: Theme.spacing.base,
  },
  cardTitle: {
    fontSize: Theme.typography.size['2xl'],
    fontWeight: Theme.typography.weight.bold,
    color: Theme.colors.text.primary,
    marginBottom: Theme.spacing.sm,
  },
  cardDescription: {
    fontSize: Theme.typography.size.base,
    color: Theme.colors.text.secondary,
    textAlign: 'center',
    marginBottom: Theme.spacing.base,
  },
  cardArrow: {
    marginTop: Theme.spacing.sm,
  },
  guideSection: {
    marginBottom: Theme.spacing['2xl'],
  },
  guideTitle: {
    fontSize: Theme.typography.size.xl,
    fontWeight: Theme.typography.weight.semibold,
    color: Theme.colors.text.primary,
    marginBottom: Theme.spacing.lg,
    textAlign: 'center',
  },
  stepsList: {
    gap: Theme.spacing.base,
  },
  stepItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    backgroundColor: Theme.colors.surface.glass,
    padding: Theme.spacing.base,
    borderRadius: Theme.radius.md,
    borderWidth: 1,
    borderColor: Theme.colors.border.subtle,
  },
  stepNumber: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: Theme.colors.accent.primary,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: Theme.spacing.md,
  },
  stepNumberText: {
    fontSize: Theme.typography.size.base,
    fontWeight: Theme.typography.weight.bold,
    color: '#FFFFFF',
  },
  stepContent: {
    flex: 1,
  },
  stepTitle: {
    fontSize: Theme.typography.size.base,
    fontWeight: Theme.typography.weight.semibold,
    color: Theme.colors.text.primary,
    marginBottom: Theme.spacing.xs,
  },
  stepDescription: {
    fontSize: Theme.typography.size.sm,
    color: Theme.colors.text.secondary,
    lineHeight: Theme.typography.lineHeight.normal * Theme.typography.size.sm,
  },
  footer: {
    alignItems: 'center',
    paddingTop: Theme.spacing.xl,
    gap: Theme.spacing.base,
  },
  footerText: {
    fontSize: Theme.typography.size.sm,
    color: Theme.colors.text.tertiary,
  },
  exitButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: Theme.spacing.sm,
    paddingHorizontal: Theme.spacing.base,
    borderRadius: Theme.radius.md,
    backgroundColor: `${Theme.colors.accent.error}1A`,
    gap: Theme.spacing.xs,
  },
  exitButtonText: {
    color: Theme.colors.accent.error,
    fontSize: Theme.typography.size.sm,
    fontWeight: Theme.typography.weight.medium,
  },
});