import React, { useState } from 'react';
import { View, Image, StyleSheet, Dimensions } from 'react-native';
import Animated, { FadeIn, FadeOut } from 'react-native-reanimated';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Theme } from '@/constants/Theme';
import { TopBar, Button, Badge, LoadingOverlay, BottomSheet, IconButton } from '@/components/ui';
import type { SnapPoint } from '@/components/ui';
import { ModelOverlay } from './ModelOverlay';

const { height: SCREEN_HEIGHT } = Dimensions.get('window');
const BOTTOM_SHEET_COLLAPSED_HEIGHT = SCREEN_HEIGHT * 0.225;

interface ReviewPhotoViewProps {
  finalPhotoUri: string;
  location: { latitude: number; longitude: number } | null;
  apiResult: any;
  modelToRender: any;
  loadedModelScene: any;
  isSending: boolean;
  isImageLoading: boolean;
  setIsImageLoading: (isImageLoading: boolean) => void;
  backToCamera: () => void;
  clearModelOverlay: () => void;
  handleSendWithLocation: () => void;
  handleSendWithoutLocation: () => void;
  handleCancelSend: () => void;
  hasModelLoaded?: boolean;
  is3DMode?: boolean;
  onToggle3DMode?: () => void;
}

export const ReviewPhotoView: React.FC<ReviewPhotoViewProps> = ({
  finalPhotoUri,
  location,
  apiResult,
  modelToRender,
  loadedModelScene,
  isSending,
  isImageLoading,
  setIsImageLoading,
  backToCamera,
  clearModelOverlay,
  handleSendWithLocation,
  handleSendWithoutLocation,
  handleCancelSend,
  hasModelLoaded = false,
  is3DMode = false,
  onToggle3DMode,
}) => {
  const insets = useSafeAreaInsets();
  const [sheetSnapPoint, setSheetSnapPoint] = useState<SnapPoint>('collapsed');
  
  const topBarHeight = insets.top + Theme.spacing.md + Theme.layout.height.topBar;
  
  const isModelDataAvailable = apiResult && !!(modelToRender?.modelUrl || loadedModelScene);

  return (
    <View style={styles.container}>
      <View style={styles.topBarContainer}>
        <TopBar
          title={isModelDataAvailable ? 'Building Found' : 'Review Photo'}
          variant="glass"
          onBack={backToCamera}
          noPaddingBottom
          rightElement={
            location ? (
              <Badge
                text={`${location.latitude.toFixed(2)}, ${location.longitude.toFixed(2)}`}
                icon="location-outline"
                variant="info"
                size="small"
                maxWidth={100}
              />
            ) : null
          }
        />
      </View>

      <View style={[
        styles.contentContainer, 
        { 
            marginTop: topBarHeight, 
            marginBottom: BOTTOM_SHEET_COLLAPSED_HEIGHT 
        }
      ]}>
        
        <Image
          key={finalPhotoUri} 
          source={{ uri: finalPhotoUri }}
          style={styles.image}
          resizeMode="contain"
          onLoadEnd={() => setIsImageLoading(false)}
        />

        {isModelDataAvailable && (
          <Animated.View
            entering={FadeIn.duration(300)}
            exiting={FadeOut.duration(300)}
            style={{...styles.modelOverlayContainer, display: is3DMode ? 'flex' : 'none'}}
          >
            <ModelOverlay
              loadedScene={loadedModelScene}
            />
          </Animated.View>
        )}

        {hasModelLoaded && onToggle3DMode && (
          <Animated.View entering={FadeIn.duration(400).delay(200)} style={styles.toggleButtonContainer}>
            <IconButton
              icon={is3DMode ? 'image-outline' : 'cube-outline'}
              onPress={onToggle3DMode}
              size="large"
              variant="glass"
              style={[
                styles.toggleButton,
                is3DMode && { backgroundColor: Theme.colors.accent.success },
              ]}
            />
          </Animated.View>
        )}
      </View>

      <LoadingOverlay
        visible={isImageLoading || isSending}
        message={isSending ? 'Analyzing Image...' : 'Loading...'}
        subMessage={isSending ? 'Searching for building match' : undefined}
        variant={isSending ? 'processing' : 'default'}
        showCancelButton={isSending}
        onCancel={handleCancelSend}
      />

      <BottomSheet
        snapPoint={sheetSnapPoint}
        onSnapPointChange={setSheetSnapPoint}
        showBackdrop={false}
      >
        <View style={styles.bottomSheetContent}>
          {isModelDataAvailable ? (
            <View style={styles.buttonsContainer}>
              <Button
                title="Retake Photo"
                onPress={backToCamera}
                variant="secondary"
                size="large"
                icon="camera-outline"
                fullWidth
              />
            </View>
          ) : (
            <View style={styles.buttonsContainer}>
              <View style={styles.primaryActionsRow}>
                <Button
                  title="Without Location"
                  onPress={handleSendWithoutLocation}
                  variant="secondary"
                  size="large"
                  icon="cloud-upload-outline"
                  disabled={isSending || isImageLoading}
                  loading={isSending}
                  style={styles.flexButton}
                />
                <Button
                  title="With Location"
                  onPress={handleSendWithLocation}
                  variant="primary"
                  size="large"
                  icon="send-outline"
                  disabled={isSending || isImageLoading || !location}
                  loading={isSending}
                  style={styles.flexButton}
                />
              </View>

              <Button
                title="Retake Photo"
                onPress={backToCamera}
                variant="secondary"
                size="large"
                icon="camera-outline"
                disabled={isSending}
                fullWidth
              />
            </View>
          )}
        </View>
      </BottomSheet>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000', 
  },
  topBarContainer: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    zIndex: 100,
  },
  contentContainer: {
    flex: 1, 
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#000',
    overflow: 'hidden',
  },
  image: {
    width: '100%',
    height: '100%',
  },
  modelOverlayContainer: {
    ...StyleSheet.absoluteFillObject,
    zIndex: 10,
  },
  toggleButtonContainer: {
    position: 'absolute',
    bottom: Theme.spacing.md,
    right: Theme.spacing.base,
    zIndex: 20,
  },
  toggleButton: {
    ...Theme.shadows.strong,
  },
  bottomSheetContent: {
    paddingTop: Theme.spacing.md,
    paddingBottom: Theme.spacing.lg,
  },
  buttonsContainer: {
    gap: Theme.spacing.md,
  },
  primaryActionsRow: {
    flexDirection: 'row',
    gap: Theme.spacing.sm,
  },
  flexButton: {
    flex: 1,
  },
});