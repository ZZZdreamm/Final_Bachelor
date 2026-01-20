import { Platform } from 'react-native';
import { Buffer } from 'buffer';
import { axiosInstance } from "@/api/axios";

export type ModelToRenderType = {
    modelUrl: string | null; // modelUrl is for web (data URI)
    modelData: ArrayBuffer | null; // modelData is for native (ArrayBuffer)
    contentType: string;
    filename: string;
    modifiedImage: string;
};

/**
 * Processes the binary response (ArrayBuffer) from the API into a usable format.
 */
export function processModelResponse(response: any): ModelToRenderType {
    const { model, image } = response.data;

    const contentType = model.media_type;
    const filename = model.filename || 'model.glb';
    const base64Data = model.data_base64;

    const imageUri = `data:${image.media_type};base64,${image.data_base64}`;

    if (Platform.OS === 'web') {
        const modelUrl = `data:${contentType};base64,${base64Data}`;

        return {
            modelUrl,
            modelData: null,
            contentType,
            filename,
            modifiedImage: imageUri
        };
    } else {
        const buffer = Buffer.from(base64Data, 'base64');

        const arrayBuffer = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);

        return {
            modelUrl: null,
            modelData: arrayBuffer,
            contentType,
            filename,
            modifiedImage: imageUri
        };
    }
}

async function sendPhotoToApi(
    finalPhotoUri: string,
    location: { latitude: number; longitude: number } | null,
    shouldFilterLocation: boolean
) {
    try {
        const formData = new FormData();
        let fileToAppend: any;

        if (Platform.OS === 'web') {
            const response = await fetch(finalPhotoUri);
            const blob = await response.blob();
            fileToAppend = blob;
        } else {
            fileToAppend = {
                uri: finalPhotoUri,
                type: 'image/jpeg',
                name: 'cropped_photo.jpg',
            } as any;
        }

        formData.append('building_image', fileToAppend);

        if (location && shouldFilterLocation) {
            formData.append('location', `${location.latitude},${location.longitude}`);
        }

        const response = await axiosInstance.post("buildings_search/find/", formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });

        if (response.status >= 200 && response.status < 300) {
            return processModelResponse(response);
        } else {
            throw new Error(`API call failed with status: ${response.status}`);
        }
    } catch (error: any) {
        if (error.response) {
            const status = error.response.status;
            const detail = error.response.data?.detail || 'Unknown error occurred';

            if (status === 404) {
                if (detail.includes('No building detected')) {
                    throw new Error('No building detected in the image. Please try taking a clearer photo.');
                } else if (detail.includes('No building from database nearby')) {
                    throw new Error('No matching building found nearby. Try a different location or send without location.');
                }
                throw new Error(detail);
            } else if (status === 400) {
                throw new Error('Invalid image. Please try taking another photo.');
            } else if (status === 500) {
                throw new Error('Server error. Please try again later.');
            } else {
                throw new Error(detail || `Request failed with status ${status}`);
            }
        } else if (error.request) {
            throw new Error('Network error. Please check your internet connection.');
        } else {
            throw new Error(error.message || 'An unexpected error occurred.');
        }
    }
}


export async function sendToApiWithLocation(
    finalPhotoUri: string,
    location: { latitude: number; longitude: number }
): Promise<{ modelToRender: ModelToRenderType, apiResult: any }> {
    const modelToRender = await sendPhotoToApi(finalPhotoUri, location, true);

    return {
        modelToRender,
        apiResult: {
            status: 'Model Downloaded (With Location Filter)',
            httpStatus: 200 
        }
    };
}

export async function sendToApiWithoutLocation(
    finalPhotoUri: string
): Promise<{ modelToRender: ModelToRenderType, apiResult: any }> {
    const modelToRender = await sendPhotoToApi(finalPhotoUri, null, false);

    return {
        modelToRender,
        apiResult: {
            status: 'Model Downloaded (No Location Filter)',
            httpStatus: 200 
        }
    };
}