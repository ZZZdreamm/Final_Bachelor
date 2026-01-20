const { getDefaultConfig } = require('expo/metro-config');

// Get the default configuration for an Expo project
const config = getDefaultConfig(__dirname);


config.resolver.assetExts.push('glb', 'gltf');

module.exports = config;