const {
    createRunOncePlugin,
    withAndroidManifest
} = require('@expo/config-plugins');

const AR_CORE_META_DATA = 'com.google.ar.core';

function setArCoreMetaData(androidManifest) {
    // Check if the application tag exists
    if (!androidManifest.manifest.application) {
        return androidManifest;
    }

    const application = androidManifest.manifest.application[0];

    // Initialize meta-data array if it doesn't exist
    if (!application['meta-data']) {
        application['meta-data'] = [];
    }
    // Check if the AR Core meta-data already exists and update it
    let hasArCore = false;
    for (const metaData of application['meta-data']) {
        if (metaData['$']['android:name'] === AR_CORE_META_DATA) {
            metaData['$']['android:value'] = 'required';
            metaData['$']['tools:replace'] = 'android:value';
            hasArCore = true;
            break;
        }
    }

    // If it doesn't exist, add the required meta-data tag
    if (!hasArCore) {
        application['meta-data'].push({
            $: {
                'android:name': AR_CORE_META_DATA,
                'android:value': 'required',
                'tools:replace': 'android:value',
            },
        });
    }

    // Ensure xmlns:tools is present in the manifest tag
    androidManifest.manifest['$']['xmlns:tools'] = 'http://schemas.android.com/tools';

    return androidManifest;
}

const withArCore = (config) => {
    return withAndroidManifest(config, (config) => {
        config.modResults = setArCoreMetaData(config.modResults);
        return config;
    });
};

module.exports = createRunOncePlugin(withArCore, 'with-ar-core', '1.0.0');