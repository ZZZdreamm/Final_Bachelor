import React, { useState, useRef, useCallback } from 'react';
import { View, Text, TouchableOpacity, Alert, ActivityIndicator, StyleSheet, Dimensions } from 'react-native';
import { CameraView, CameraType, useCameraPermissions } from 'expo-camera';
import * as Location from 'expo-location';
import * as ImagePicker from 'expo-image-picker';
import { Ionicons } from '@expo/vector-icons';
import { Gesture, GestureDetector } from 'react-native-gesture-handler';
import { useSharedValue, runOnJS } from 'react-native-reanimated';
import { Link, useRouter } from 'expo-router';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { Theme } from '@/constants/Theme';
import { TopBar, IconButton, BottomSheet, Card, Badge, LoadingOverlay } from '@/components/ui';
import type { SnapPoint } from '@/components/ui';
import { CameraGridOverlay, type GridType } from './CameraGridOverlay';
import { CameraSettingsSheet } from './CameraSettingsSheet';

const MAX_ZOOM_LEVEL = 0.5;
const { height: SCREEN_HEIGHT } = Dimensions.get('window');
const BOTTOM_SHEET_COLLAPSED_HEIGHT = SCREEN_HEIGHT * 0.225;

type LocationType = { latitude: number; longitude: number } | null;

interface ImageSourceSelectorProps {
  onImageSelected: (uri: string, location: LocationType) => void;
  isProcessingImage?: boolean;
  setIsProcessingImage: (isProcessing: boolean) => void;
  setImageWidth: (width: number | null) => void;
  setImageHeight: (height: number | null) => void;
}

export function ImageSourceSelector({ onImageSelected, isProcessingImage = false, setIsProcessingImage, setImageWidth, setImageHeight }: ImageSourceSelectorProps) {
  const router = useRouter();
  const insets = useSafeAreaInsets();
  
  // Calculate top bar height: safe area top + padding + content height + padding
  const topBarHeight = insets.top + Theme.spacing.md + Theme.layout.height.topBar + Theme.spacing.md;
  
  // Calculate available height for camera: screen height - top bar - bottom sheet
  const cameraAvailableHeight = SCREEN_HEIGHT - topBarHeight - BOTTOM_SHEET_COLLAPSED_HEIGHT;

  const [cameraPermission, requestCameraPermission] = useCameraPermissions();
  const [locationPermission, requestLocationPermission] = Location.useForegroundPermissions();
  const [mediaLibraryPermission, requestMediaLibraryPermission] =
    ImagePicker.useMediaLibraryPermissions();

  const [facing, setFacing] = useState<CameraType>('back');
  const [zoomState, setZoomState] = useState(0);
  const [gridType, setGridType] = useState<GridType>('none');
  const [sheetSnapPoint, setSheetSnapPoint] = useState<SnapPoint>('collapsed');
  const cameraRef = useRef<CameraView | null>(null);

  const currentZoom = useSharedValue(0);
  const startZoom = useSharedValue(0);

  const updateZoomState = useCallback((newZoom: number) => {
    setZoomState(newZoom);
  }, []);

  const pinchGesture = Gesture.Pinch()
    .onStart(() => {
      startZoom.value = currentZoom.value;
    })
    .onUpdate((event) => {
      let newZoom = startZoom.value + (event.scale - 1) * MAX_ZOOM_LEVEL;
      newZoom = Math.max(0, Math.min(MAX_ZOOM_LEVEL, newZoom));
      currentZoom.value = newZoom;
      runOnJS(updateZoomState)(newZoom);
    });

  const getCurrentLocation = async () => {
    if (!locationPermission?.granted) return null;
    try {
      const loc = await Location.getCurrentPositionAsync({
        accuracy: Location.Accuracy.Highest,
      });
      return {
        latitude: loc.coords.latitude,
        longitude: loc.coords.longitude,
      };
    } catch (error) {
      return null;
    }
  };

  const toggleCameraFacing = useCallback(() => {
    setFacing((current) => (current === 'back' ? 'front' : 'back'));
  }, []);

  const handleTakePicture = useCallback(async () => {
    if (!cameraRef.current) return;
    try {
      if (!locationPermission?.granted) {
        Alert.alert('Permission Required', 'Location is needed to tag the photo.');
        return;
      }
      setIsProcessingImage(true);
      const locationData = await getCurrentLocation();
 
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.95,
        skipProcessing: true,
      });

      if (photo && locationData) {
        setImageWidth(null);
        setImageHeight(null);
        onImageSelected(photo.uri, locationData);
      } else {
        Alert.alert('Error', 'Could not retrieve location data.');
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to capture image.');
    }
  }, [locationPermission, onImageSelected]);

  const handlePickFromGallery = useCallback(async () => {
    if (!mediaLibraryPermission?.granted) {
      const permission = await requestMediaLibraryPermission();
      if (!permission.granted) return;
    }

    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ['images'],
        allowsEditing: false,
        quality: 1,
      });

      if (!result.canceled && result.assets[0].uri) {
        const asset = result.assets[0];

        setImageWidth(asset.width);
        setImageHeight(asset.height);
        setIsProcessingImage(true);
        const locationData = await getCurrentLocation();

        if (!locationData && locationPermission?.granted) {
          Alert.alert('Warning', 'Could not fetch current location for this image.');
        }

        onImageSelected(result.assets[0].uri, locationData);
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to pick image from library.');
    }
  }, [
    mediaLibraryPermission,
    requestMediaLibraryPermission,
    locationPermission,
    onImageSelected,
  ]);

  // Permissions loading state
  if (!cameraPermission || !locationPermission || !mediaLibraryPermission) {
    return (
      <View style={styles.centerContainer}>
        <ActivityIndicator size="large" color={Theme.colors.accent.primary} />
      </View>
    );
  }

  // Permissions denied state
  if (!cameraPermission.granted || !locationPermission.granted) {
    return (
      <View style={styles.permissionsContainer}>
        <Card variant="elevated" padding="large" style={styles.permissionsCard}>
          <View style={styles.permissionsIcon}>
            <Ionicons
              name="shield-checkmark-outline"
              size={64}
              color={Theme.colors.accent.warning}
            />
          </View>

          <Text style={styles.permissionsTitle}>Permissions Required</Text>
          <Text style={styles.permissionsText}>
            This app needs camera and location access to capture and tag photos of buildings.
          </Text>

          <View style={styles.permissionsList}>
            <View style={styles.permissionItem}>
              <Ionicons
                name={cameraPermission.granted ? 'checkmark-circle' : 'close-circle'}
                size={24}
                color={
                  cameraPermission.granted
                    ? Theme.colors.accent.success
                    : Theme.colors.accent.error
                }
              />
              <View style={styles.permissionContent}>
                <Text style={styles.permissionTitle}>Camera Access</Text>
                <Text style={styles.permissionDescription}>
                  Required to capture photos of buildings
                </Text>
              </View>
            </View>

            <View style={styles.permissionItem}>
              <Ionicons
                name={locationPermission.granted ? 'checkmark-circle' : 'close-circle'}
                size={24}
                color={
                  locationPermission.granted
                    ? Theme.colors.accent.success
                    : Theme.colors.accent.error
                }
              />
              <View style={styles.permissionContent}>
                <Text style={styles.permissionTitle}>Location Access</Text>
                <Text style={styles.permissionDescription}>
                  Used to geotag photos for better building recognition
                </Text>
              </View>
            </View>
          </View>

          <View style={styles.permissionsActions}>
            {!cameraPermission.granted && (
              <TouchableOpacity
                style={[styles.permButton, styles.permButtonPrimary]}
                onPress={requestCameraPermission}
              >
                <Text style={styles.permButtonText}>Grant Camera</Text>
              </TouchableOpacity>
            )}
            {!locationPermission.granted && (
              <TouchableOpacity
                style={[styles.permButton, styles.permButtonPrimary]}
                onPress={requestLocationPermission}
              >
                <Text style={styles.permButtonText}>Grant Location</Text>
              </TouchableOpacity>
            )}
          </View>

          <Link href="/(tabs)" asChild>
            <TouchableOpacity style={styles.goBackButton}>
              <Text style={styles.goBackButtonText}>Go Back</Text>
            </TouchableOpacity>
          </Link>
        </Card>
      </View>
    );
  }

  // Camera view
  return (
    <View style={styles.container}>
      <TopBar
        title="Capture"
        variant="glass"
        onBack={() => router.back()}
      />

      <View style={[styles.cameraContainer, { top: topBarHeight, height: cameraAvailableHeight }]}>
       <GestureDetector gesture={pinchGesture}>
          <CameraView style={styles.camera} facing={facing} ref={cameraRef} zoom={zoomState} />
        </GestureDetector> 

        <View style={styles.overlayContainer} pointerEvents="none">
          <CameraGridOverlay gridType={gridType} />
        </View>
      </View>

      {zoomState > 0 && (
        <View style={styles.zoomIndicator}>
          <Badge
            text={`${(1 + zoomState * (1 / MAX_ZOOM_LEVEL) * 5).toFixed(1)}x`}
            variant="default"
            size="small"
          />
        </View>
      )}

      <BottomSheet
        snapPoint={sheetSnapPoint}
        onSnapPointChange={setSheetSnapPoint}
        showBackdrop={false}
      >
        {sheetSnapPoint === 'half' ? (
          <CameraSettingsSheet gridType={gridType} onGridTypeChange={setGridType} />
        ) : (
          <View style={styles.bottomSheetCollapsed}>
            <TouchableOpacity
              onPress={handlePickFromGallery}
              style={styles.galleryButtonInSheet}
            >
              <Ionicons name="images-outline" size={24} color={Theme.colors.text.primary} />
            </TouchableOpacity>

            <TouchableOpacity onPress={handleTakePicture} style={styles.captureButton}>
              <View style={styles.captureButtonInner} />
            </TouchableOpacity>

            <IconButton
              icon="camera-reverse-outline"
              onPress={toggleCameraFacing}
              size="medium"
              variant="ghost"
            />
          </View>
        )}
      </BottomSheet>

      <LoadingOverlay
        visible={isProcessingImage}
        message="Uploading Image..."
        variant="processing"
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Theme.colors.background.primary,
  },
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: Theme.colors.background.primary,
  },
  cameraContainer: {
    position: 'absolute',
    left: 0,
    right: 0,
    width: '100%',
  },
  camera: {
    flex: 1,
  },
  overlayContainer: {
    ...StyleSheet.absoluteFillObject,
    zIndex: 5,
  },
  permissionsContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: Theme.spacing.lg,
    backgroundColor: Theme.colors.background.primary,
  },
  permissionsCard: {
    width: '100%',
    maxWidth: 400,
    alignItems: 'center',
  },
  permissionsIcon: {
    width: 100,
    height: 100,
    borderRadius: 50,
    backgroundColor: `${Theme.colors.accent.warning}1A`,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: Theme.spacing.lg,
  },
  permissionsTitle: {
    fontSize: Theme.typography.size['2xl'],
    fontWeight: Theme.typography.weight.bold,
    color: Theme.colors.text.primary,
    marginBottom: Theme.spacing.md,
    textAlign: 'center',
  },
  permissionsText: {
    fontSize: Theme.typography.size.base,
    color: Theme.colors.text.secondary,
    textAlign: 'center',
    marginBottom: Theme.spacing.lg,
    lineHeight: Theme.typography.lineHeight.relaxed * Theme.typography.size.base,
  },
  permissionsList: {
    width: '100%',
    gap: Theme.spacing.base,
    marginBottom: Theme.spacing.lg,
  },
  permissionItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    padding: Theme.spacing.base,
    backgroundColor: Theme.colors.surface.glass,
    borderRadius: Theme.radius.md,
    gap: Theme.spacing.md,
  },
  permissionContent: {
    flex: 1,
  },
  permissionTitle: {
    fontSize: Theme.typography.size.base,
    fontWeight: Theme.typography.weight.semibold,
    color: Theme.colors.text.primary,
    marginBottom: Theme.spacing.xs,
  },
  permissionDescription: {
    fontSize: Theme.typography.size.sm,
    color: Theme.colors.text.secondary,
    lineHeight: Theme.typography.lineHeight.normal * Theme.typography.size.sm,
  },
  permissionsActions: {
    width: '100%',
    gap: Theme.spacing.md,
    marginBottom: Theme.spacing.base,
  },
  permButton: {
    padding: Theme.spacing.base,
    borderRadius: Theme.radius.md,
    alignItems: 'center',
  },
  permButtonPrimary: {
    backgroundColor: Theme.colors.accent.primary,
  },
  permButtonText: {
    color: '#FFFFFF',
    fontSize: Theme.typography.size.base,
    fontWeight: Theme.typography.weight.semibold,
  },
  goBackButton: {
    marginTop: Theme.spacing.sm,
  },
  goBackButtonText: {
    color: Theme.colors.accent.primary,
    fontSize: Theme.typography.size.base,
    fontWeight: Theme.typography.weight.medium,
  },
  zoomIndicator: {
    position: 'absolute',
    bottom: 180,
    alignSelf: 'center',
    zIndex: 10,
  },
  bottomSheetCollapsed: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingTop: Theme.spacing.md,
    paddingBottom: Theme.spacing.lg,
    paddingHorizontal: Theme.spacing.xl,
  },
  galleryButtonInSheet: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: Theme.colors.surface.glass,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: Theme.colors.border.default,
  },
  captureButton: {
    width: 72,
    height: 72,
    borderRadius: 36,
    borderWidth: 4,
    borderColor: '#FFFFFF',
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'transparent',
  },
  captureButtonInner: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: '#FFFFFF',
  },
});