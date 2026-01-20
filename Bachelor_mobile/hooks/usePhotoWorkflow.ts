import { useState, useCallback } from 'react';
import { GLTFLoader } from "three-stdlib";
import RNFS from 'react-native-fs';
import { Buffer } from "buffer";
import { ModelToRenderType, sendToApiWithLocation, sendToApiWithoutLocation } from '@/utils/photo/modelUtils';
import type { ToastType } from '@/components/ui';

type LocationType = { latitude: number; longitude: number } | null;

export function usePhotoWorkflow() {
    const [location, setLocation] = useState<LocationType>(null);

    const [originalPhotoUri, setOriginalPhotoUri] = useState<string | null>(null);
    const [finalPhotoUri, setFinalPhotoUri] = useState<string | null>(null);

    const [isSending, setIsSending] = useState(false);
    const [isImageLoading, setIsImageLoading] = useState(false);
    const [isProcessingImage, setIsProcessingImage] = useState(false);
    const [apiResult, setApiResult] = useState<any>(null);
    const [modelToRender, setModelToRender] = useState<ModelToRenderType | null>(null);
    const [loadedModelScene, setLoadedModelScene] = useState<any>(null);
    const [is3DMode, setIs3DMode] = useState(false);
    const [imageWidth, setImageWidth] = useState<number | null>(null);
    const [imageHeight, setImageHeight] = useState<number | null>(null);

    const [toastVisible, setToastVisible] = useState(false);
    const [toastMessage, setToastMessage] = useState('');
    const [toastType, setToastType] = useState<ToastType>('info');

    const showToast = useCallback((message: string, type: ToastType = 'info') => {
        setToastMessage(message);
        setToastType(type);
        setToastVisible(true);
    }, []);

    const hideToast = useCallback(() => {
        setToastVisible(false);
    }, []);

    const handleImageSelected = useCallback((uri: string, loc: LocationType) => {
        setIsProcessingImage(true);
        setOriginalPhotoUri(uri);
        setLocation(loc);
        setFinalPhotoUri(null);
        setApiResult(null);
        setModelToRender(null);
        setLoadedModelScene(null);
        setIs3DMode(false);
    }, []);

    const handleCropComplete = useCallback((croppedUri: string) => {
        setFinalPhotoUri(croppedUri);
        setIsImageLoading(true);
    }, []);

    const backToCamera = useCallback(() => {
        setOriginalPhotoUri(null);
        setFinalPhotoUri(null);
        setIsImageLoading(false);
        setLocation(null);
        setApiResult(null);
        setModelToRender(null);
        setLoadedModelScene(null);
        setIs3DMode(false);
    }, []);

    const clearModelOverlay = useCallback(() => {
        setApiResult(null);
        setModelToRender(null);
        setLoadedModelScene(null);
        setIs3DMode(false);
    }, []);

    const toggle3DMode = useCallback(() => setIs3DMode(prev => !prev), []);

    const loadModelData = useCallback(async (modelData: ModelToRenderType) => {
        setLoadedModelScene(null);
        const loader = new GLTFLoader();

        const success = (gltf: any) => {
            setLoadedModelScene(gltf.scene);
            setIs3DMode(true);
            showToast('3D model loaded successfully', 'success');
        }
        const error = () => {
            showToast('Failed to parse the 3D model data', 'error');
            setLoadedModelScene(null);
            setIs3DMode(false);
        };

        setFinalPhotoUri(modelData.modifiedImage)

        if (modelData.modelUrl) {
            loader.load(modelData.modelUrl, success, undefined, error);
        } else if (modelData.modelData) {
            try {
                const buffer = Buffer.from(modelData.modelData);
                const base64Data = buffer.toString('base64');
                const tempFilePath = `${RNFS.DocumentDirectoryPath}/${modelData.filename}`;
                const fileURI = `file://${tempFilePath}`;

                const exists = await RNFS.exists(fileURI);
                if (!exists) {
                    await RNFS.writeFile(tempFilePath, base64Data, 'base64');
                }
                setLoadedModelScene(fileURI);
                setIs3DMode(true);
            } catch (e) {
                showToast('Failed to save or load the 3D model file', 'error');
                setLoadedModelScene(null);
                setIs3DMode(false);
            }
        }
    }, [showToast]);

    const handleCancelSend = useCallback(() => {
        setIsSending(false);
        clearModelOverlay();
    }, [clearModelOverlay]);

    const handleSendApiCall = useCallback(async (apiCall: any) => {
        if (!finalPhotoUri) return;
        try {
            setIsSending(true);
            clearModelOverlay();

            let result;
            if (apiCall === sendToApiWithLocation) {
                if(!location) {
                    showToast("Location missing", 'error');
                    setIsSending(false);
                    return;
                }
                result = await sendToApiWithLocation(finalPhotoUri, location);
            } else {
                result = await sendToApiWithoutLocation(finalPhotoUri);
            }

            setModelToRender(result.modelToRender);
            setApiResult(result.apiResult);
            await loadModelData(result.modelToRender);

        } catch (error: any) {
            showToast(error.message || 'An error occurred', 'error');
        } finally {
            setIsSending(false);
        }
    }, [finalPhotoUri, location, loadModelData, clearModelOverlay, showToast]);

    return {
        originalPhotoUri, finalPhotoUri, location,
        handleImageSelected, handleCropComplete, backToCamera,
        isSending, isImageLoading, setIsImageLoading,
        isProcessingImage, setIsProcessingImage,
        apiResult, modelToRender, loadedModelScene, clearModelOverlay,
        handleSendWithLocation: () => handleSendApiCall(sendToApiWithLocation),
        handleSendWithoutLocation: () => handleSendApiCall(sendToApiWithoutLocation),
        handleCancelSend,
        is3DMode, toggle3DMode,
        toastVisible, toastMessage, toastType, hideToast,
        imageWidth, imageHeight, setImageWidth, setImageHeight
    };
}
