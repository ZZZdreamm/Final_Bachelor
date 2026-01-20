module.exports = ({ config }) => {
    return {
        ...config,
        ios: {
            bundleIdentifier: "com.zzzdream.mobilepw",
            supportsTablet: true,

            infoPlist: {
                ITSAppUsesNonExemptEncryption: false
            }
        },
        android: {
            package: "com.zzzdream.mobilepw",
            enableProguardInReleaseBuilds: true,
            enableShrinkResourcesInReleaseBuilds: true
        },
        extra: {
            // API_ADDRESS: 'https://zzzdream95-bachelor.hf.space/',
            API_ADDRESS: process.env.NODE_ENV === 'production'
                ? 'https://zzzdream95-bachelor.hf.space/'
                : 'http://192.168.0.93:8000/',
            eas: {
                projectId: "1fbf60c2-3086-41a6-b208-023a556b610c"
            }
        }
    };
};