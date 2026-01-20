import React from 'react';
import { GestureHandlerRootView } from "react-native-gesture-handler";
import { ImageSourceSelector } from '@/components/photo/ImageSourceSelector';
import { ReviewPhotoView } from '@/components/photo/ReviewPhotoView';
import ImageCropper from "@/components/photo/ImageCropper";
import { Toast } from '@/components/ui';
import { usePhotoWorkflow } from '@/hooks/usePhotoWorkflow';

function PhotoPageContent() {
    const {
        originalPhotoUri, finalPhotoUri, location,
        handleImageSelected, handleCropComplete, backToCamera,
        isSending, isImageLoading, setIsImageLoading,
        isProcessingImage, setIsProcessingImage,
        apiResult, modelToRender, loadedModelScene, clearModelOverlay,
        handleSendWithLocation, handleSendWithoutLocation,
        handleCancelSend,
        is3DMode, toggle3DMode,
        toastVisible, toastMessage, toastType, hideToast,
        imageWidth, imageHeight, setImageWidth, setImageHeight
    } = usePhotoWorkflow();

    return (
        <>
            {!originalPhotoUri && (
                <ImageSourceSelector
                    onImageSelected={handleImageSelected}
                    isProcessingImage={isProcessingImage}
                    setIsProcessingImage={setIsProcessingImage}
                    setImageWidth={setImageWidth}
                    setImageHeight={setImageHeight}
                />
            )}

            {originalPhotoUri && !finalPhotoUri && (
                <ImageCropper
                    originalUri={originalPhotoUri}
                    imageWidth={imageWidth}
                    imageHeight={imageHeight}
                    onRetake={backToCamera}
                    onCropComplete={handleCropComplete}
                    onImageLoaded={() => setIsProcessingImage(false)}
                />
            )}

            {finalPhotoUri && (
                <ReviewPhotoView
                    finalPhotoUri={finalPhotoUri}
                    location={location}
                    apiResult={apiResult}
                    modelToRender={modelToRender}
                    loadedModelScene={loadedModelScene}
                    isSending={isSending}
                    isImageLoading={isImageLoading}
                    setIsImageLoading={setIsImageLoading}
                    backToCamera={backToCamera}
                    clearModelOverlay={clearModelOverlay}
                    handleSendWithLocation={handleSendWithLocation}
                    handleSendWithoutLocation={handleSendWithoutLocation}
                    handleCancelSend={handleCancelSend}
                    is3DMode={is3DMode}
                    onToggle3DMode={toggle3DMode}
                    hasModelLoaded={!!loadedModelScene}
                />
            )}

            <Toast
                visible={toastVisible}
                message={toastMessage}
                type={toastType}
                onHide={hideToast}
            />
        </>
    );
}

export default function PhotoPage() {
    return (
        <GestureHandlerRootView style={{ flex: 1 }}>
            <PhotoPageContent />
        </GestureHandlerRootView>
    );
}