import React from 'react';
import { View, ActivityIndicator, StyleSheet } from 'react-native';
import { ModelLoader } from "@/components/photo/ModelLoader";

interface ModelOverlayProps {
    loadedScene: any;
}

export const ModelOverlay: React.FC<ModelOverlayProps> = ({ loadedScene }) => {
    const isLoading = !loadedScene;

    return (
        <View style={localStyles.container}>

            <View style={localStyles.tintLayer}>
                {isLoading ? (
                    <View style={localStyles.loadingContainer}>
                        <ActivityIndicator size="large" color="#32CD32" />
                    </View>
                ) : (
                    <View style={localStyles.modelWrapper}>
                        <ModelLoader modelUri={loadedScene}/>
                    </View>
                )}
            </View>
        </View>
    );
};

const localStyles = StyleSheet.create({
    container: {
        flex: 1,
        // Ramka na zewnątrz
        borderWidth: 2,
        borderColor: '#32CD32',

        // WARSTWA 1: Solidny kolor, który odcina widoczność zdjęcia pod spodem.
        backgroundColor: 'white',

        overflow: 'hidden',
    },
    tintLayer: {
        flex: 1,
        width: '100%',
        height: '100%',

        // WARSTWA 2: Ciemny filtr nałożony na białe tło
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
    },
    loadingContainer: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
    },
    modelWrapper: {
        flex: 1,
        width: '100%',
        height: '100%',
    }
});