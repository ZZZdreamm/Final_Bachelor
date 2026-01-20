# Story Build - Mobile App

A React Native mobile application that captures building photos and discovers their 3D models using computer vision and AI. Built with Expo Router and Filament 3D viewer.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the App](#running-the-app)
- [Project Structure](#project-structure)
- [Pages & Screens](#pages--screens)
  - [Home Screen](#home-screen)
  - [Photo Workflow Screen](#photo-workflow-screen)
- [Navigation](#navigation)
- [API Integration](#api-integration)

## Overview

Story Build is a mobile application that allows users to capture photos of buildings and automatically retrieve their 3D models. The app uses AI-powered building recognition with optional GPS location matching to provide accurate 3D visualizations.

**Workflow:**
1. Capture or select a building photo
2. Crop the image to focus on the building
3. Send to API for processing (with or without GPS location)
4. View the building as an interactive 3D model

## Features

- **Camera Capture**: Live camera preview with pinch-to-zoom, grid overlays, and camera settings
- **Gallery Selection**: Choose existing photos from device gallery
- **Image Cropping**: Interactive crop interface with pan, zoom, and precise controls
- **GPS Integration**: Optional location-based building matching
- **3D Model Viewer**: Interactive 3D model rendering using Filament
- **Dark Theme**: Optimized dark UI with smooth animations


## Getting Started

### Prerequisites

- Node.js 18+ and npm
- Expo CLI
- For Android: Android Studio with SDK
- Development Build installed on your device

### Installation

```bash
cd Bachelor_mobile
npm install
```

### Running the App

#### Development Mode

```bash
npm run start
```

#### Using App

1. Install development build on your phone
2. Run `npm start`
3. Click `A` on your computer connected with your device through cabel to run it on device

## Project Structure

```
Bachelor_mobile/
├── app/                         # Expo Router pages (file-based routing)
│   ├── _layout.tsx              # Root layout with theme provider
│   └── (tabs)/                  # Tab-based navigation group
│       ├── _layout.tsx          # Tab navigation configuration
│       ├── index.tsx            # Home screen
│       └── photo.tsx            # Photo workflow screen
│
├── components/                   # Reusable components
│   ├── photo/                    # Photo & 3D components
│   │   ├── ImageSourceSelector.tsx
│   │   ├── ImageCropper.tsx
│   │   ├── ReviewPhotoView.tsx
│   │   ├── ModelOverlay.tsx
│   │   ├── ModelLoader.tsx
│   │   ├── CameraGridOverlay.tsx
│   │   └── CameraSettingsSheet.tsx
│   └── ui/                      # Shared UI components
│       ├── Button.tsx
│       ├── IconButton.tsx
│       ├── TopBar.tsx
│       ├── Badge.tsx
│       ├── Card.tsx
│       ├── BottomSheet.tsx
│       ├── LoadingOverlay.tsx
│       ├── Toast.tsx
│       ├── ThemedText.tsx
│       └── ThemedView.tsx
│
├── hooks/                       # Custom React hooks
│   ├── usePhotoWorkflow.ts      # Main photo workflow state machine
│   └── useThemeColor.ts         # Theme color utilities
│
├── api/                         # API integration
│   └── axios.ts                 # Axios instance configuration
│
├── utils/                        # Utility functions
│   └── photo/
│       ├── modelUtils.ts        # API calls & 3D model processing
│       └── cropUtils.ts         # Image cropping utilities
│
├── constants/                   # App constants & design tokens
│   ├── Theme.ts                 # Colors, typography, spacing
│   └── Colors.ts                # Theme color definitions
│
├── assets/                      # Static assets
│   ├── fonts/                   # Custom fonts
│   └── images/                  # Icons and images
│
└── config files                 # Configuration files
    ├── app.json                 # Expo configuration
    ├── app.config.ts            # Runtime configuration
    ├── eas.json                 # EAS build config
    ├── tsconfig.json            # TypeScript config
    └── metro.config.js          # Metro bundler config
```

## Pages & Screens

### Home Screen
**Location**: [app/(tabs)/index.tsx](app/(tabs)/index.tsx)

The landing page that introduces users to the app.

**Features:**
- Welcome message and app description
- Visual 3-step workflow guide:
  1. Capture a building photo
  2. Crop and send for processing
  3. View building in 3D
- Call-to-action button navigating to Photo screen
- Exit button

---

### Photo Workflow Screen
**Location**: [app/(tabs)/photo.tsx](app/(tabs)/photo.tsx)

A multi-state screen that handles the complete photo-to-3D workflow.

**State Machine:**
The screen renders different components based on workflow state managed by `usePhotoWorkflow` hook:

#### State 1: Image Source Selection
**Component**: `ImageSourceSelector`

**Features:**
- Live camera preview with rear/front camera toggle
- Camera controls:
  - Flip camera (front/rear)
  - Pinch-to-zoom gesture
  - Camera settings bottom sheet
- Image picker from gallery
- Capture photo
- GPS location permission handling
- Photo permissions handling

**User Flow:**
1. Camera preview loads automatically
2. User can adjust zoom, toggle grid, or flip camera
3. Press capture button OR select from gallery
4. Proceeds to cropping state

#### State 2: Image Cropping
**Component**: `ImageCropper`

**Features:**
- Interactive pan, zoom, and crop gestures
- Visual crop guides with grid overlay
- Real-time crop preview
- Retake option (returns to camera)
- Crop completion with final dimensions

**User Flow:**
1. Image loads with default crop area
2. User adjusts crop area with gestures
3. Press "Crop" to confirm OR "Retake" to go back
4. Proceeds to review state

#### State 3: Review & Send
**Component**: `ReviewPhotoView`

**Features:**
- Cropped image preview
- GPS location display badge
- Two sending options:
  - **Send with Location**: API matches nearby buildings using GPS
  - **Send without Location**: General building recognition
- Loading overlay during API processing
- 3D model display (when available)
- Toggle between 2D photo and 3D model views

**User Flow:**
1. Review cropped image
2. Choose send option (with or without location)
3. API processes image and returns 3D model
4. View interactive 3D model OR see error message
5. Can reset to take another photo

#### State 4: 3D Model Display
**Component**: `ModelOverlay`

**Features:**
- Interactive 3D model viewer
- Pan and pinch gestures for model manipulation
- Toggle back to 2D photo view
- Model rotation and zoom controls
- Smooth loading transitions
- Image with overlaid 3D model over it

**User Flow:**
1. 3D model loads automatically after API success
2. User can interact with model (rotate, zoom)
3. Toggle to see photo with 3D model overlaid over it
4. Reset workflow to capture another building

---

## Navigation

### Navigation Structure

The app uses **Expo Router** for file-based routing:

```
Root Stack
└── (tabs) Group
    ├── index → Home Screen
    └── photo → Photo Workflow Screen
```

### Navigation Flow

```
Home Screen (/)
    │
    └── "Take Photo" button
        │
        ↓
Photo Workflow Screen (/photo)
    │
    ├── State 1: Camera/Gallery Selection
    ├── State 2: Image Cropping
    ├── State 3: Review & Send
    └── State 4: 3D Model Display
```

## API Integration

### Base Configuration

**File**: [api/axios.ts](api/axios.ts)

```typescript
const API_BASE_URL = Constants.expoConfig?.extra?.API_ADDRESS || 'http://localhost:8000';
```

**Environment URLs:**
- **Production**: `https://zzzdream95-bachelor.hf.space/`
- **Development**: `http://192.168.0.93:8000/`

## Permissions

The app requires the following permissions:

| Permission | Usage | Required |
|------------|-------|----------|
| Camera | Capture building photos | Yes |
| Location | GPS-based building matching | Yes |
| Media Library | Select photos from gallery | Yes |
| File System | Save temporary 3D models | Yes |

---
