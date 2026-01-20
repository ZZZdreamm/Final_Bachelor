import React, {useRef} from 'react';
import {View, StyleSheet} from 'react-native';

import {Camera, DefaultLight, FilamentScene, FilamentView, Model, useCameraManipulator} from "react-native-filament";
import {Gesture, GestureDetector} from "react-native-gesture-handler";


const FilamentModelViewer: React.FC<{ uri: string }> = ({ uri }) => {
    const isPinching = useRef(false);
    const lastScale = useRef(1);

    const cameraManipulator = useCameraManipulator({
        orbitHomePosition: [0, 0, -2],
        targetPosition: [0, 0, 0],
        orbitSpeed: [-0.005, -0.005],
        zoomSpeed: [0.05],
    });

    const panGesture = Gesture.Pan()
        .maxPointers(1)
        .activeOffsetX([-10, 10])
        .activeOffsetY([-10, 10])
        .onBegin((e) => {
            if (!isPinching.current) {
                cameraManipulator?.grabBegin(e.x, e.y, false);
            }
        })
        .onUpdate((e) => {
            if (!isPinching.current) {
                cameraManipulator?.grabUpdate(e.x, e.y);
            }
        })
        .onEnd(() => {
            cameraManipulator?.grabEnd();
        });

    const pinchGesture = Gesture.Pinch()
        .onStart(() => {
            isPinching.current = true;
            lastScale.current = 1;

            cameraManipulator?.grabEnd();
        })
        .onUpdate((e) => {
            const delta = lastScale.current - e.scale;
            lastScale.current = e.scale;

            const ZOOM_SENSITIVITY = 10;
            cameraManipulator?.scroll(e.focalX, e.focalY, delta * ZOOM_SENSITIVITY);
        })
        .onEnd(() => {
            isPinching.current = false;
        })
        .onFinalize(() => {
            isPinching.current = false;
        });

    const gestures = Gesture.Simultaneous(panGesture, pinchGesture);

    return (
        <GestureDetector gesture={gestures}>
            <View style={{ flex: 1 }}>
                <FilamentView style={{ flex: 1 }}>
                    <Camera cameraManipulator={cameraManipulator} />

                    <DefaultLight />

                    <Model
                        source={{ uri: uri }}
                        scale={[0.5, 0.5, 0.5]}
                        position={[0, 0, 0]}
                        transformToUnitCube
                    />
                </FilamentView>
            </View>
        </GestureDetector>
    );
};

export const ModelLoader: React.FC<{ modelUri: string }> = ({ modelUri }) => {
    return (
        <View style={StyleSheet.absoluteFill}>
            {modelUri && (
                <FilamentScene>
                        <FilamentModelViewer uri={modelUri} />
                </FilamentScene>
            )}
        </View>
    );
};
